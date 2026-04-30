# -*- coding: utf-8 -*-
"""Hybrid (LSTM + ViT) data prep.

Wraps `prepare_lstm_data` (training, pandas) and
`prepare_lstm_data_rapids` (inference, cuDF) with the bookkeeping the
ViT side needs: one path-per-neighbor-station to the per-hour radiometer
``.npy`` snapshot.

The image columns are named `image_path_{station}` so the hybrid
sequencer can yank them out via `df[image_list_cols]`.
"""

import os
from typing import List, Tuple

import pandas as pd

from model_data import prepare_lstm_data, prepare_lstm_data_rapids


PROFILER_IMAGE_ROOT = os.environ.get(
    "PROFILER_IMAGE_ROOT",
    "/home/aevans/nwp_bias/src/machine_learning/data/profiler_images",
)


def _image_path_for(stid: str, vt: pd.Timestamp, root: str) -> str:
    """Build the canonical profiler-image path for a station/time.

    The producer (data_cleaning/build_profiler_images.py) saves files
    as `{root}/{year}/{stid}/{stid}_{year}_{MMDDHH}.npy`.
    """
    vt = pd.Timestamp(vt)
    year = vt.year
    fmt = vt.strftime("%m%d%H")
    return os.path.join(root, str(year), stid, f"{stid}_{year}_{fmt}.npy")


def _attach_image_columns(
    lstm_df: pd.DataFrame,
    valid_times: List[pd.Timestamp],
    stations: List[str],
    image_root: str,
    require_exists: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    """Add `image_path_{stid}` columns to `lstm_df` and return the
    updated dataframe plus the new column names."""
    image_list_cols: List[str] = []
    if len(valid_times) != len(lstm_df):
        # Truncate / pad valid_times to match the dataframe length.
        # `prepare_lstm_data` returns one valid_time per surviving row
        # but if any downstream step dropped rows the lengths can drift;
        # being defensive here saves a head-scratching mismatch later.
        valid_times = valid_times[: len(lstm_df)]

    for stid in stations:
        col = f"image_path_{stid}"
        paths = []
        for vt in valid_times:
            path = _image_path_for(stid, vt, image_root)
            if require_exists and not os.path.exists(path):
                path = None
            paths.append(path)
        lstm_df[col] = paths
        image_list_cols.append(col)

    return lstm_df, image_list_cols


def prepare_hybrid_data(
    nysm_df,
    hrrr_df,
    station,
    metvar,
    fh,
    train=False,
    image_root: str = PROFILER_IMAGE_ROOT,
    require_image_exists: bool = True,
):
    """Pandas / training variant.  Mirrors `prepare_lstm_data` but also
    returns the list of image-path column names."""
    lstm_df, features, stations, target, valid_times = (
        prepare_lstm_data.prepare_lstm_data(
            nysm_df, hrrr_df, station, metvar, fh=fh, train=train
        )
    )
    lstm_df, image_list_cols = _attach_image_columns(
        lstm_df,
        valid_times,
        stations,
        image_root=image_root,
        require_exists=require_image_exists if train else False,
    )
    if train and require_image_exists:
        # During training we want to drop rows that have no profiler
        # image for any of their neighbor stations - the ViT can't run
        # on missing inputs.  Inference path tolerates missing images
        # (they get zero-filled in the sequencer).
        before = len(lstm_df)
        mask = lstm_df[image_list_cols].notna().all(axis=1)
        lstm_df = lstm_df[mask].reset_index(drop=True)
        # Re-align valid_times.
        valid_times = [vt for vt, ok in zip(valid_times, mask.tolist()) if ok]
        if len(lstm_df) != before:
            print(
                f"[prepare_hybrid_data] dropped {before - len(lstm_df)} rows "
                "with at least one missing radiometer image"
            )
    return lstm_df, features, stations, target, valid_times, image_list_cols


def prepare_hybrid_data_rapids(
    nysm_df,
    hrrr_df,
    station,
    metvar,
    fh,
    now=None,
    sequence_length: int = 30,
    train: bool = False,
    image_root: str = PROFILER_IMAGE_ROOT,
):
    """cuDF / inference variant.  Mirrors `prepare_lstm_data_rapids`
    and additionally returns image-path column names.

    We force `require_image_exists=False` at inference because the
    most-recent radiometer snapshot may not have landed on disk yet;
    missing images are zero-filled by the sequencer."""
    lstm_df, features, stations, target, valid_times = (
        prepare_lstm_data_rapids.prepare_lstm_data(
            nysm_df,
            hrrr_df,
            station,
            metvar,
            fh,
            now=now,
            sequence_length=sequence_length,
            train=train,
        )
    )
    # `lstm_df` returned from the rapids prep is a pandas DataFrame
    # (the function .to_pandas()'s the cuDF before calling get_error).
    if not isinstance(lstm_df, pd.DataFrame):
        lstm_df = lstm_df.to_pandas()

    lstm_df, image_list_cols = _attach_image_columns(
        lstm_df,
        valid_times,
        stations,
        image_root=image_root,
        require_exists=False,
    )
    return lstm_df, features, stations, target, valid_times, image_list_cols
