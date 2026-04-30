"""Encoder-decoder LSTM building blocks.

This module defines the three classes used by both the LSTM-only and
the BNN training / inference pipelines:

* `ShallowRegressionLSTM_encode`
    Stacked LSTM that consumes the past + future feature window and
    returns the final `(hidden, cell)` tuple to be handed to the
    decoder.

* `ShallowRegressionLSTM_decode`
    Stacked LSTM + 2-layer MLP head that autoregressively rolls out
    `forecast_steps` predictions from the encoder hidden state.

* `ShallowLSTM_seq2seq_multi_task`
    Composite seq2seq module that owns one encoder and one decoder
    and exposes `train_model`, `test_model`, and `predict` so the
    outer training loops stay short.

The encoder is fed an `x` tensor of shape
``(batch, sequence_length + forecast_steps, num_features)``.  In the
sequencer the future portion contains real future HRRR forecasts with
the trailing-64 NYSM columns persisted from the last past row; see
`model_architecture/sequencer.py` for the leak-prevention rationale.
"""

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F


torch.autograd.set_detect_anomaly(True)


class ShallowRegressionLSTM_encode(nn.Module):
    """Stacked-LSTM encoder.

    Returns the final `(h, c)` hidden state of a `num_layers`-deep LSTM
    after consuming the entire input window.
    """

    def __init__(self, num_sensors, hidden_units, num_layers, mlp_units, device):
        super().__init__()
        self.num_sensors = num_sensors  # number of input features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device
        self.mlp_units = mlp_units

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x):
        x = x.to(self.device)
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(self.device)
        _, (hn, cn) = self.lstm(x, (h0, c0))
        return hn, cn


class ShallowRegressionLSTM_decode(nn.Module):
    """Stacked-LSTM decoder with a 2-layer MLP head.

    Per call: takes a single timestep's input + a hidden state, runs
    one LSTM step, and projects the output through a `LeakyReLU` MLP
    back to `num_sensors` features.  The outer module (`train_model`,
    `predict`) calls this `forecast_steps` times in a recursive loop.
    """

    def __init__(self, num_sensors, hidden_units, num_layers, mlp_units, device):
        super().__init__()
        self.num_sensors = num_sensors  # number of input/output features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device
        self.mlp_units = mlp_units

        self.lstm = nn.LSTM(
            input_size=self.num_sensors,
            hidden_size=self.hidden_units,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(
            in_features=self.hidden_units,
            out_features=self.num_sensors,
            bias=False,
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_units, self.mlp_units),
            nn.LeakyReLU(),
            nn.Linear(self.mlp_units, self.num_sensors),
        )

    def forward(self, x, hidden):
        x = x.to(self.device)
        out, hidden = self.lstm(x, hidden)

        # pass through mlp
        outn = self.mlp(out)
        return outn, hidden


class ShallowLSTM_seq2seq_multi_task(nn.Module):
    """LSTM encoder-decoder seq2seq model.

    Parameters
    ----------
    num_sensors : int
        Number of input/output features per timestep.
    hidden_units : int
        Width of the LSTM hidden state in both encoder and decoder.
    num_layers : int
        Stack depth shared by encoder and decoder.
    mlp_units : int
        Width of the decoder's hidden MLP layer.
    device : torch.device
        Compute device.
    num_stations : int
        Currently unused; reserved for a multi-head per-station decoder
        variant that may be re-enabled in future versions.

    The class also exposes three loop helpers:

    * `train_model(...)`   - one training epoch (recursive or
                             teacher-forced rollout)
    * `test_model(...)`    - one validation epoch
    * `predict(...)`       - inference rollout, returning concatenated
                             outputs across all batches
    """

    def __init__(
        self, num_sensors, hidden_units, num_layers, mlp_units, device, num_stations
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device
        self.mlp_units = mlp_units

        # Shared encoder
        self.encoder = ShallowRegressionLSTM_encode(
            num_sensors=num_sensors,
            hidden_units=hidden_units,
            num_layers=num_layers,
            mlp_units=mlp_units,
            device=device,
        )

        self.decoder = ShallowRegressionLSTM_decode(
            num_sensors=num_sensors,
            hidden_units=hidden_units,
            num_layers=num_layers,
            mlp_units=mlp_units,
            device=device,
        )

    def train_model(
        self,
        data_loader,
        loss_func,
        optimizer,
        epoch,
        training_prediction,
        teacher_forcing_ratio,
        dynamic_tf=True,
        decay_rate=0.02,
    ):
        """Run one training epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Yields `(X, y)` per batch where
            ``X.shape == (B, sequence_length + forecast_steps, F)`` and
            ``y.shape == (B, forecast_steps, 1)``.
        loss_func : Callable[[Tensor, Tensor], Tensor]
        optimizer : torch.optim.Optimizer
        epoch : int
            Used only for the printed log line.
        training_prediction : {"recursive", "teacher_forcing"}
            "recursive" feeds the previous decoder output back in;
            "teacher_forcing" mixes ground-truth `y` with the decoder
            output at probability `teacher_forcing_ratio`.
        teacher_forcing_ratio : float
        dynamic_tf : bool
            If True, decay `teacher_forcing_ratio` after every batch.
        decay_rate : float
            Per-batch decrement applied to `teacher_forcing_ratio`.
        """
        num_batches = len(data_loader)
        total_loss = 0
        self.train()

        for batch_idx, batch in enumerate(data_loader):
            gc.collect()
            # Skip the batch if it is None or if X or y are None
            if batch is None:
                continue
            X, y = batch

            X, y = X.to(self.device), y.to(self.device)

            # Encoder forward pass (shared)
            encoder_hidden = self.encoder(X)

            # Initialize outputs tensor
            outputs = torch.zeros(y.size(0), y.size(1), X.size(2)).to(self.device)

            decoder_input = X[:, -1, :].unsqueeze(1)
            decoder_hidden = encoder_hidden

            for t in range(y.size(1)):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden
                )
                outputs[:, t, :] = decoder_output.squeeze(1)

                if training_prediction == "recursive":
                    decoder_input = decoder_output
                elif training_prediction == "teacher_forcing":
                    if torch.rand(1).item() < teacher_forcing_ratio:
                        # Determine the padding required
                        padding_dim = (
                            X.shape[-1] - y.shape[-1]
                        )  # Difference in feature dimensions
                        if padding_dim > 0:
                            # Pad the last dimension of y to match X
                            y = F.pad(y, (0, padding_dim), mode="constant", value=0)
                        decoder_input = y[:, t, :].unsqueeze(1)
                    else:
                        decoder_input = decoder_output

            optimizer.zero_grad()
            loss = loss_func(outputs, y)
            loss.backward()
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)
            optimizer.step()
            total_loss += loss.item()

            if dynamic_tf and teacher_forcing_ratio > 0:
                teacher_forcing_ratio = max(0, teacher_forcing_ratio - decay_rate)

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch}], Loss: {avg_loss:.4f}")
        return avg_loss

    def test_model(self, data_loader, loss_function, epoch):
        """Run one validation / test epoch under `torch.no_grad`."""
        num_batches = len(data_loader)
        total_loss = 0
        self.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                gc.collect()
                if batch is None:
                    continue
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)

                encoder_hidden = self.encoder(X)
                outputs = torch.zeros(y.size(0), y.size(1), X.size(2)).to(self.device)
                decoder_input = X[:, -1, :].unsqueeze(1)
                decoder_hidden = encoder_hidden

                for t in range(y.size(1)):
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden
                    )
                    outputs[:, t, :] = decoder_output.squeeze(1)
                    decoder_input = decoder_output

                total_loss += loss_function(outputs, y).item()

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch}], Test Loss: {avg_loss:.4f}")
        return avg_loss

    def predict(self, data_loader):
        """Run inference and return all per-batch outputs concatenated.

        Returns a tensor of shape
        ``(N_total_items, forecast_steps, num_sensors)``.
        """
        num_batches = len(data_loader)
        all_outputs = []
        self.eval()

        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(data_loader):
                gc.collect()
                X, y = X.to(self.device), y.to(self.device)

                encoder_hidden = self.encoder(X)
                outputs = torch.zeros(y.size(0), y.size(1), X.size(2)).to(self.device)
                decoder_input = X[:, -1, :].unsqueeze(1)
                decoder_hidden = encoder_hidden

                for t in range(y.size(1)):
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden
                    )
                    outputs[:, t, :] = decoder_output.squeeze(1)
                    decoder_input = decoder_output

                all_outputs.append(outputs)
        all_outputs = torch.cat(all_outputs, dim=0)
        return all_outputs
