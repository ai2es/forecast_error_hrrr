"""Hybrid LSTM + Vision-Transformer (ViT) seq2seq model.

The model fuses two complementary encoders:

* `ShallowRegressionLSTM_encode` consumes the past + future
  HRRR/NYSM feature window and returns a hidden state.
* `VisionTransformer` consumes a stack of radiometer profiler images
  for the same stations / timesteps and returns a per-(station, time)
  embedding.

The two embeddings are concatenated and decoded by
`ShallowRegressionLSTM_decode` to produce one forecast-error value
per future timestep.
"""

import gc
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(".")

from model_architecture import encode_decode_lstm, hybrid_vit_encoder

torch.autograd.set_detect_anomaly(False)


class LSTM_Encoder_Decoder_with_ViT(nn.Module):
    """LSTM (time-series) + ViT (images) -> LSTM decoder.

    All hyper-parameters are explicit constructor arguments to make
    the model fully configurable from the training script
    (`engine_hybrid_training.py`).

    Parameters
    ----------
    num_sensors, hidden_units, num_layers, mlp_units, device, num_stations
        Same meaning as `ShallowLSTM_seq2seq_multi_task`.
    past_timesteps, future_timesteps : int
        Number of past / future steps the ViT sees.  Together they
        define the temporal axis of its position embeddings.
    pos_embedding, time_embedding : tensor or callable
        Spatial / temporal embeddings handed straight to the ViT.
    vit_num_layers, num_heads, hidden_dim, mlp_dim, output_dim,
    dropout, attention_dropout : ViT hyper-parameters.
    """

    def __init__(
        self,
        num_sensors,
        hidden_units,
        num_layers,
        mlp_units,
        device,
        num_stations,
        past_timesteps,
        future_timesteps,
        pos_embedding,
        time_embedding,
        vit_num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        output_dim,
        dropout,
        attention_dropout,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.mlp_units = mlp_units
        self.device = device
        self.num_stations = num_stations
        self.past_timesteps = past_timesteps
        self.future_timesteps = future_timesteps
        self.pos_embedding = pos_embedding
        self.time_embedding = time_embedding
        self.vit_num_layers = vit_num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout

        # LSTM encoder
        self.encoder = encode_decode_lstm.ShallowRegressionLSTM_encode(
            num_sensors=num_sensors,
            hidden_units=hidden_units,
            num_layers=num_layers,
            mlp_units=mlp_units,
            device=device,
        )

        # ViT encoder
        self.ViT = hybrid_vit_encoder.VisionTransformer(
            stations=num_stations,
            past_timesteps=past_timesteps,
            future_timesteps=future_timesteps,
            pos_embedding=pos_embedding,
            time_embedding=time_embedding,
            num_layers=vit_num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            num_classes=output_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        # LSTM decoder
        self.decoder = encode_decode_lstm.ShallowRegressionLSTM_decode(
            num_sensors=num_sensors,
            hidden_units=hidden_units,
            num_layers=num_layers,
            mlp_units=mlp_units,
            device=device,
        )

        # Project the ViT output to the LSTM hidden width.  These
        # constants assume the default hyper-parameter set used in
        # `engine_hybrid_training.py` and may need to be re-derived
        # from `vit_num_layers * hidden_dim` if you change them.
        self.hidden_proj = nn.Linear(2688, 1728)

        # Fuse the two encoder embeddings down to a single LSTM-shaped
        # hidden state (concat -> Linear -> LeakyReLU -> Dropout).
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_units * 2, hidden_units),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
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
        """One training epoch on `(X, P, y)` batches.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Yields `(X, P, y)` triples (use
            `SequenceDatasetMultiTaskHybrid`).
        loss_func, optimizer, epoch, training_prediction,
        teacher_forcing_ratio, dynamic_tf, decay_rate :
            See `ShallowLSTM_seq2seq_multi_task.train_model`.
        """
        num_batches = len(data_loader)
        total_loss = 0
        self.train()

        for batch_idx, batch in enumerate(data_loader):
            X, P, y = batch
            X, P, y = X.to(self.device), P.to(self.device), y.to(self.device)
            # --- Encoders ---
            encoder_hidden = self.encoder(X)
            encoder_hidden_profiler = self.ViT(P)

            # Expand ViT encoding to match (num_layers, batch, hidden_size)
            encoder_hidden_profiler = encoder_hidden_profiler.repeat(
                encoder_hidden[0].shape[0], 1, 1
            )

            # Optionally project ViT encoding
            encoder_hidden_profiler = self.hidden_proj(encoder_hidden_profiler)

            # Concatenate LSTM and ViT encodings
            combined = torch.cat([encoder_hidden[0], encoder_hidden_profiler], dim=-1)

            # Apply fusion MLP
            pass_hidden = self.fusion_mlp(combined)

            # Decoder init
            outputs = torch.zeros(y.size(0), y.size(1), X.size(2)).to(self.device)
            decoder_input = X[:, -1, :].unsqueeze(1)
            decoder_hidden = pass_hidden, encoder_hidden[1]

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
        """Validation epoch on `(X, P, y)` batches under `torch.no_grad`."""
        num_batches = len(data_loader)
        total_loss = 0
        self.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                X, P, y = batch
                X, P, y = X.to(self.device), P.to(self.device), y.to(self.device)
                # --- Encoders ---
                encoder_hidden = self.encoder(X)
                encoder_hidden_profiler = self.ViT(P)

                # Expand ViT encoding to match (num_layers, batch, hidden_size)
                encoder_hidden_profiler = encoder_hidden_profiler.repeat(
                    encoder_hidden[0].shape[0], 1, 1
                )

                # Optionally project ViT encoding
                encoder_hidden_profiler = self.hidden_proj(encoder_hidden_profiler)

                # Concatenate LSTM and ViT encodings
                combined = torch.cat(
                    [encoder_hidden[0], encoder_hidden_profiler], dim=-1
                )

                # Apply fusion MLP
                pass_hidden = self.fusion_mlp(combined)

                # Decoder init
                outputs = torch.zeros(y.size(0), y.size(1), X.size(2)).to(self.device)
                decoder_input = X[:, -1, :].unsqueeze(1)
                decoder_hidden = pass_hidden, encoder_hidden[1]
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
        """Inference rollout on `(X, P, y, v)` batches.

        Returns `(all_outputs, valid_times)` concatenated across the
        loader.  Use `SequenceDatasetMultiTaskHybrid(...,
        return_valid_times=True)` to produce 4-tuple batches.
        """
        num_batches = len(data_loader)
        all_outputs = []
        all_valid_times = []
        self.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                X, P, y, v = batch
                X, P, y = X.to(self.device), P.to(self.device), y.to(self.device)
                # --- Encoders ---
                encoder_hidden = self.encoder(X)
                encoder_hidden_profiler = self.ViT(P)

                # Expand ViT encoding to match (num_layers, batch, hidden_size)
                encoder_hidden_profiler = encoder_hidden_profiler.repeat(
                    encoder_hidden[0].shape[0], 1, 1
                )

                # Optionally project ViT encoding
                encoder_hidden_profiler = self.hidden_proj(encoder_hidden_profiler)

                # Concatenate LSTM and ViT encodings
                combined = torch.cat(
                    [encoder_hidden[0], encoder_hidden_profiler], dim=-1
                )

                # Apply fusion MLP
                pass_hidden = self.fusion_mlp(combined)

                # Decoder init
                outputs = torch.zeros(y.size(0), y.size(1), X.size(2)).to(self.device)
                decoder_input = X[:, -1, :].unsqueeze(1)
                decoder_hidden = pass_hidden, encoder_hidden[1]
                for t in range(y.size(1)):
                    decoder_output, decoder_hidden = self.decoder(
                        decoder_input, decoder_hidden
                    )
                    outputs[:, t, :] = decoder_output.squeeze(1)
                    decoder_input = decoder_output

                all_outputs.append(outputs)
                all_valid_times.append(v)

        all_outputs = torch.cat(all_outputs, dim=0)
        valid_times = torch.cat(all_valid_times, axis=0)  # (N, H)
        return all_outputs, valid_times
