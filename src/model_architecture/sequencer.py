import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

from model_data import normalization


class SequenceDatasetMultiTask(Dataset):
    """Dataset class for the encoder-decoder LSTM (and BNN/Hybrid).

    Per-window z-score normalization
    --------------------------------
    Each item is a window of `sequence_length + forecast_steps` rows.
    The window is z-scored INSIDE this dataset using statistics computed
    from its own PAST portion (rows ``[0, sequence_length)``) and
    applied to the whole window.  Computing stats on past-only avoids
    leaking the future portion into the normalization, while still
    giving the future rows the same scale as the past so the encoder
    sees a consistent representation.

    The columns excluded from the z-score are listed in
    `model_data.normalization` (cyclic time encodings, geo /
    categorical features, the target itself, image references).  For
    those columns the raw value is returned unchanged.

    Window layout (post-normalization)
    ----------------------------------
        x[: sequence_length]      = real past HRRR + real past NYSM
        x[sequence_length:]       = real future HRRR + persisted last
                                    past NYSM
    The trailing-64-cols persistence overwrite (4 stations * 16 NYSM
    features) is applied AFTER normalization, so the persisted NYSM
    values land in the same z-score space as the past.

    `target_error` is intentionally NEVER normalized:
        * it is excluded from `features` upstream (in
          `prepare_lstm_data*.py`), so it is not present in `self.X` and
          can't be touched by the z-score block;
        * `self.y` is pulled directly from the raw `target_error`
          column and is returned as-is from `__getitem__`;
        * `normalization.SKIP_EXACT` also lists `target_error` as a
          defense-in-depth safeguard so that even if it ever ended up
          in `features`, the mask would pass it through unchanged.
    Keeping the target in raw error units lets the model learn the
    actual error distribution (its scale, skew, tails) directly.

    Optional fields
    ---------------
    `valid_times`  : list of pandas Timestamps aligned 1-1 with the
                     rows of `dataframe`.  Required if
                     `return_valid_times=True`.
    `return_valid_times` : if True, `__getitem__` additionally returns
                     a tensor of int64 ns-epoch timestamps for the
                     `forecast_steps` future rows.  This is what the
                     BNN's and Hybrid's `.predict()` expect.
    """

    def __init__(
        self,
        dataframe,
        target,
        features,
        sequence_length,
        forecast_steps,
        device,
        metvar,
        nwp_model="HRRR",
        valid_times=None,
        return_valid_times=False,
    ):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.forecast_steps = forecast_steps
        self.device = device
        self.metvar = metvar
        self.nwp_model = nwp_model
        self.return_valid_times = return_valid_times
        self.y = torch.tensor(dataframe[target].values).float().to(device)
        self.X = torch.tensor(dataframe[features].values).float().to(device)

        normalize_mask_np = normalization.get_normalize_mask(features)
        self.normalize_mask = torch.tensor(
            normalize_mask_np, dtype=torch.bool, device=device
        )

        if return_valid_times:
            if valid_times is None:
                raise ValueError(
                    "valid_times must be provided when return_valid_times=True"
                )
            self.valid_times_int = torch.tensor(
                [pd.Timestamp(t).value for t in valid_times], dtype=torch.int64
            ).to(device)
        else:
            self.valid_times_int = None

    def __len__(self):
        return self.X.shape[0]

    def _build_x_y(self, i):
        """Shared logic: return the (normalized, persisted) x window and y."""
        x_start = i
        x_end = i + (self.sequence_length + self.forecast_steps)
        y_start = i + self.sequence_length
        y_end = y_start + self.forecast_steps

        x = self.X[x_start:x_end, :].clone()
        y = self.y[y_start:y_end].unsqueeze(1)

        if x.shape[0] < (self.sequence_length + self.forecast_steps):
            _x = torch.zeros(
                (
                    (self.sequence_length + self.forecast_steps) - x.shape[0],
                    self.X.shape[1],
                ),
                device=self.device,
            )
            x = torch.cat((x, _x), 0)

        if y.shape[0] < self.forecast_steps:
            _y = torch.zeros(
                (self.forecast_steps - y.shape[0], 1),
                device=self.device,
            )
            y = torch.cat((y, _y), 0)

        # ---- Per-window z-score (past-only stats) -------------------
        past = x[: self.sequence_length, :]
        mean = torch.nanmean(past, dim=0, keepdim=True)
        past_filled = torch.where(torch.isnan(past), mean.expand_as(past), past)
        std = past_filled.std(dim=0, unbiased=False, keepdim=True)
        std = torch.where(std < 1e-8, torch.ones_like(std), std)

        x_norm = (x - mean) / std
        mask = self.normalize_mask.unsqueeze(0)
        x = torch.where(mask, x_norm, x)

        # ---- Persistence overwrite (after normalization) ------------
        x[-self.forecast_steps :, -int(4 * 16) :] = x[
            -int(self.forecast_steps + 1), -int(4 * 16) :
        ].clone()

        return x, y

    def _maybe_get_valid_times(self, i):
        """Return the int64 ns valid_times for the future window, padded
        with zeros if we ran past the end of the dataframe."""
        v_start = i + self.sequence_length
        v_end = v_start + self.forecast_steps
        v = self.valid_times_int[v_start:v_end]
        if v.shape[0] < self.forecast_steps:
            pad = torch.zeros(
                self.forecast_steps - v.shape[0],
                dtype=torch.int64,
                device=self.device,
            )
            v = torch.cat([v, pad])
        return v

    def __getitem__(self, i):
        x, y = self._build_x_y(i)
        if self.return_valid_times:
            v = self._maybe_get_valid_times(i)
            return x, y, v
        return x, y


class SequenceDatasetMultiTaskHybrid(SequenceDatasetMultiTask):
    """Hybrid (LSTM + ViT) dataset.

    On top of `SequenceDatasetMultiTask` this also returns a per-item
    image tensor `P` (shape ``(num_image_stations, h, w, c)``) loaded
    from `.npy` files referenced by columns in `image_list_cols`.

    Following the original FSDP hybrid, the images are pulled from a
    SINGLE row of the dataframe per item (the first future timestep,
    `i + sequence_length`) so the ViT sees one snapshot per inference
    rather than a time series of images.
    """

    def __init__(
        self,
        dataframe,
        target,
        features,
        sequence_length,
        forecast_steps,
        device,
        metvar,
        image_list_cols,
        nwp_model="HRRR",
        valid_times=None,
        return_valid_times=False,
        image_dtype=torch.float32,
    ):
        super().__init__(
            dataframe=dataframe,
            target=target,
            features=features,
            sequence_length=sequence_length,
            forecast_steps=forecast_steps,
            device=device,
            metvar=metvar,
            nwp_model=nwp_model,
            valid_times=valid_times,
            return_valid_times=return_valid_times,
        )
        self.image_list_cols = list(image_list_cols)
        self.image_dtype = image_dtype
        # Stash image paths per row in a plain Python list of lists so
        # we can lazy-load on the CPU side; loading every image up front
        # would blow GPU memory.
        self.P_ls = dataframe[self.image_list_cols].values.tolist()

    def _load_images_for_row(self, row_paths):
        images = []
        for img_path in row_paths:
            if img_path is None or (isinstance(img_path, float) and np.isnan(img_path)):
                # Missing image -> zeros of the same shape as the first
                # successfully loaded image.  We delay shape inference
                # until we hit a real one.
                images.append(None)
                continue
            arr = np.load(img_path).astype(np.float32)
            images.append(torch.from_numpy(arr))

        # Backfill missing entries with zeros of the right shape.
        ref = next((t for t in images if t is not None), None)
        if ref is None:
            raise FileNotFoundError(
                "All radiometer images for this row were missing; "
                "cannot infer image shape."
            )
        for k, t in enumerate(images):
            if t is None:
                images[k] = torch.zeros_like(ref)

        stacked = torch.stack(images).to(self.image_dtype).to(self.device)
        return stacked

    def __getitem__(self, i):
        x, y = self._build_x_y(i)

        # First future timestep determines the image snapshot used
        # by the ViT.
        idx = min(i + self.sequence_length, len(self.P_ls) - 1)
        P = self._load_images_for_row(self.P_ls[idx])

        if self.return_valid_times:
            v = self._maybe_get_valid_times(i)
            return x, P, y, v
        return x, P, y
