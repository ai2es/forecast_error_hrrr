"""Bayesian-LSTM model components.

This module wraps `encode_decode_lstm` with a small Bayesian feed-
forward head (`SimpleBNN`) that outputs `(mu, log_var)` for every
forecast step.  Training optimises a Gaussian negative-log-likelihood
on `(mu, log_var, y)` (the variational KL term is exposed via
`SimpleBNN.kl()` for users who want to add an ELBO regulariser).

Uncertainty decomposition
-------------------------
At inference, `ShallowLSTM_seq2seq_multi_task_bnn.predict` runs
`mc_samples` Monte-Carlo passes through the Bayesian head.  Using
the law of total variance,

    Var[y | x]  =  E_theta[ Var[y | x, theta] ]   +   Var_theta[ E[y | x, theta] ]
                  `------- aleatoric -------'        `------ epistemic ------'

it returns the predictive mean alongside both components separately,
so downstream code can reason about *data noise* (aleatoric) and
*model uncertainty* (epistemic) independently.

Key classes
-----------
* `BayesianLinearCustom`
    Fully-connected layer with Gaussian variational posteriors over
    weight and bias, sampled via the local-reparameterisation trick.
    Provides a closed-form `kl()` against a zero-mean prior.

* `SimpleBNN`
    2-hidden-layer MLP built from `BayesianLinearCustom` layers, with
    two output heads: `mu` (predicted mean) and `log_var` (predicted
    log-variance).  Operates per-timestep on the LSTM output stream.

* `ShallowLSTM_seq2seq_multi_task_bnn`
    LSTM encoder + decoder followed by `SimpleBNN`.  Mirrors the
    plain `ShallowLSTM_seq2seq_multi_task` API but produces a
    predictive mean and decomposed predictive uncertainty
    (epistemic + aleatoric) at inference time.

Helpers
-------
* `nll_loss_gaussian(mu, log_var, target)` - pointwise Gaussian NLL
* `compute_ucr_torch(samples, y_true, alpha)` - empirical coverage
"""

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(".")

from model_architecture import encode_decode_lstm


class BayesianLinearCustom(nn.Module):
    """Linear layer with a Gaussian variational posterior on every weight.

    Parameters
    ----------
    in_features, out_features : int
        Standard `nn.Linear` shapes.
    prior_sigma : float
        Standard deviation of the zero-mean Gaussian prior used in the
        closed-form KL.
    init_mu_std : float
        Std of the Normal initialiser applied to each posterior mean.
    init_rho : float
        Initial value of the unconstrained-scale parameter rho.  The
        actual posterior std is `softplus(rho)`.
    """

    def __init__(
        self, in_features, out_features, prior_sigma=0.5, init_mu_std=0.1, init_rho=-3.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = float(prior_sigma)

        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).normal_(0.0, init_mu_std)
        )
        self.weight_rho = nn.Parameter(
            torch.empty(out_features, in_features).fill_(init_rho)
        )
        self.bias_mu = nn.Parameter(torch.empty(out_features).normal_(0.0, init_mu_std))
        self.bias_rho = nn.Parameter(torch.empty(out_features).fill_(init_rho))

        self.register_buffer("_eps", torch.tensor(1e-8))

    def forward(self, x):
        weight_sigma = F.softplus(self.weight_rho) + self._eps
        bias_sigma = F.softplus(self.bias_rho) + self._eps

        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)

        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps

        return F.linear(x, weight, bias)

    def kl(self):
        """Closed-form KL between the variational posterior and a
        zero-mean Gaussian prior with std `self.prior_sigma`."""
        weight_sigma = F.softplus(self.weight_rho) + self._eps
        bias_sigma = F.softplus(self.bias_rho) + self._eps
        prior_var = self.prior_sigma**2

        kl_w = (
            torch.log(self.prior_sigma / weight_sigma)
            + (weight_sigma**2 + self.weight_mu**2) / (2.0 * prior_var)
            - 0.5
        )
        kl_b = (
            torch.log(self.prior_sigma / bias_sigma)
            + (bias_sigma**2 + self.bias_mu**2) / (2.0 * prior_var)
            - 0.5
        )

        return kl_w.sum() + kl_b.sum()


def nll_loss_gaussian(mu, log_var, target, reduction="mean"):
    """Pointwise Gaussian negative log-likelihood.

    Per element: ``0.5 * (log(2*pi) + log_var + (target - mu)^2 / exp(log_var))``.

    Parameters
    ----------
    mu, log_var, target : Tensor
        All of shape `(B, T, S)`.
    reduction : {"mean", "sum", None}
        Aggregation across the batch dimension after summing time and
        feature dimensions.
    """
    var = torch.exp(log_var)
    se = (target - mu) ** 2
    const = 0.5 * math.log(2 * math.pi)
    nll = 0.5 * (log_var + se / (var + 1e-12)) + const
    # sum over time and sensors
    nll = nll.sum(dim=(1, 2))  # shape (B,)
    if reduction == "mean":
        return nll.mean()
    elif reduction == "sum":
        return nll.sum()
    elif reduction is None:
        return nll
    else:
        return nll.mean()


def compute_ucr_torch(samples, y_true, alpha=0.90):
    """Empirical predictive coverage (UCR).

    Computes the fraction of `y_true` values that fall inside the
    central `alpha` predictive interval estimated from `samples`.

    Parameters
    ----------
    samples : Tensor of shape `(B, mc, T, S)`
        MC samples drawn from the predictive distribution.
    y_true : Tensor of shape `(B, T, S)`.
    alpha : float in (0, 1)
        Nominal coverage of the interval (default 0.90).
    """
    # quantiles along mc dimension
    lower_q = (1 - alpha) / 2.0
    upper_q = 1.0 - lower_q

    lower = torch.quantile(samples, lower_q, dim=1)  # (B, T, S)
    upper = torch.quantile(samples, upper_q, dim=1)  # (B, T, S)

    covered = (y_true >= lower) & (y_true <= upper)
    coverage_fraction = covered.float().mean().item()
    return coverage_fraction


class SimpleBNN(nn.Module):
    """2-hidden-layer Bayesian MLP head with `(mu, log_var)` output.

    Designed to be applied at every timestep of the LSTM decoder
    output stream.  Expects an input of shape `(B, T, feat)` and
    returns two tensors of shape `(B, T, output_dim)` (`mu`, `log_var`).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the LSTM output features.
    hidden_dim : int
        Width of the two intermediate Bayesian layers.
    output_dim : int
        Number of target features per timestep.
    prior_sigma : float
        Std of the Gaussian prior used by every `BayesianLinearCustom`.
    clamp_input : float
        Inputs are clamped to ``[-clamp_input, clamp_input]`` before
        the first FC layer to stop occasional spikes from blowing up
        the variational posterior.
    """

    def __init__(
        self, input_dim, hidden_dim, output_dim, prior_sigma=0.5, clamp_input=50.0
    ):
        super().__init__()
        self.fc1 = BayesianLinearCustom(input_dim, hidden_dim, prior_sigma=prior_sigma)
        self.fc2 = BayesianLinearCustom(hidden_dim, hidden_dim, prior_sigma=prior_sigma)
        self.fc_mu = BayesianLinearCustom(
            hidden_dim, output_dim, prior_sigma=prior_sigma
        )
        self.fc_logvar = BayesianLinearCustom(
            hidden_dim, output_dim, prior_sigma=prior_sigma
        )

        self.clamp_input = float(clamp_input)
        self.logvar_min = -10.0
        self.logvar_max = 10.0

    def forward(self, lstm_outputs):
        # lstm_outputs: (B, T, feat)
        B, T, feat = lstm_outputs.shape
        lstm_flat = lstm_outputs.reshape(B * T, feat)

        # clamp inputs
        lstm_flat = torch.clamp(lstm_flat, -self.clamp_input, self.clamp_input)

        h = F.softplus(self.fc1(lstm_flat))
        h = F.softplus(self.fc2(h))

        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        log_var = torch.clamp(log_var, self.logvar_min, self.logvar_max)

        # NaN safety: convert to finite numbers (prefer raising during debug)
        if torch.isnan(mu).any() or torch.isnan(log_var).any():
            mu = torch.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
            log_var = torch.nan_to_num(
                log_var,
                nan=self.logvar_min,
                posinf=self.logvar_max,
                neginf=self.logvar_min,
            )

        mu = mu.view(B, T, -1)
        log_var = log_var.view(B, T, -1)

        return mu, log_var

    def kl(self):
        """Total KL divergence summed across every Bayesian sub-layer."""
        kl_sum = 0.0
        for m in self.modules():
            if isinstance(m, BayesianLinearCustom):
                kl_sum = kl_sum + m.kl()
        return kl_sum


class ShallowLSTM_seq2seq_multi_task_bnn(nn.Module):
    """LSTM encoder-decoder + Bayesian MLP head.

    Same forward shape as `ShallowLSTM_seq2seq_multi_task` but the
    decoder's output stream is fed through `SimpleBNN`, yielding a
    predictive `(mu, log_var)` per timestep.

    Parameters
    ----------
    num_sensors : int
        Number of input/output features per timestep.
    hidden_units : int
        Width of the LSTM hidden state.
    num_layers : int
        Stack depth of the encoder/decoder.
    mlp_units : int
        Width of the decoder's MLP head.
    device : torch.device
    num_stations : int
        Reserved for a future per-station decoder variant; unused.
    seq_len : int
        Length of the encoder input window (sequence_length +
        forecast_steps).  Currently informational only.
    input_dim : int
        Feature width fed into the BNN head (typically equal to
        `num_sensors`).
    """

    def __init__(
        self,
        num_sensors,
        hidden_units,
        num_layers,
        mlp_units,
        device,
        num_stations,
        seq_len,
        input_dim,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device
        self.mlp_units = mlp_units
        self.seq_len = seq_len
        self.input_dim = input_dim

        # Shared encoder
        self.encoder = encode_decode_lstm.ShallowRegressionLSTM_encode(
            num_sensors=num_sensors,
            hidden_units=hidden_units,
            num_layers=num_layers,
            mlp_units=mlp_units,
            device=device,
        )

        self.decoder = encode_decode_lstm.ShallowRegressionLSTM_decode(
            num_sensors=num_sensors,
            hidden_units=hidden_units,
            num_layers=num_layers,
            mlp_units=mlp_units,
            device=device,
        )

        self.bnn = SimpleBNN(
            input_dim=input_dim,
            hidden_dim=hidden_units,
            output_dim=1,
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

        The Gaussian NLL is computed internally on `(mu, log_var, y)`,
        so the `loss_func` argument is accepted for API parity with
        `ShallowLSTM_seq2seq_multi_task.train_model` but is unused.
        Likewise, `training_prediction` and `teacher_forcing_ratio`
        are kept for signature compatibility; the BNN currently uses
        a fully recursive rollout.
        """
        num_batches = len(data_loader)
        total_loss = 0.0
        self.train()

        for batch_idx, batch in enumerate(data_loader):
            if batch is None:
                continue

            X, y = batch
            X, y = X.to(self.device), y.to(self.device)

            # Encoder pass: fold the entire input window into the
            # initial decoder hidden state.
            encoder_hidden = self.encoder(X)

            outputs = torch.zeros(y.size(0), y.size(1), X.size(2)).to(self.device)

            # Decoder warm-start: feed the last encoder timestep as
            # the first decoder input, then roll out recursively.
            decoder_input = X[:, -1, :].unsqueeze(1)
            decoder_hidden = encoder_hidden

            for t in range(y.size(1)):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden
                )
                outputs[:, t, :] = decoder_output.squeeze(1)
                decoder_input = decoder_output

            # Per-timestep Bayesian head -> (mu, log_var).
            mu, log_var = self.bnn(outputs)

            loss = nll_loss_gaussian(mu, log_var, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch}], Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def test_model(self, data_loader, loss_function, epoch):
        """Run one validation epoch (Gaussian NLL, no gradient)."""
        num_batches = len(data_loader)
        total_loss = 0
        self.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
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

                # bnn
                mu, log_var = self.bnn(outputs)
                # 🔹 Compute loss

                loss = nll_loss_gaussian(mu, log_var, y)
                total_loss += loss.item()

        # 🔹 Average loss across batches
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch}], Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def predict(self, data_loader, mc_samples=30, return_samples=False):
        """Run inference and return decomposed predictive uncertainty.

        The encoder + decoder are deterministic given fixed weights, so
        we run them ONCE per batch and cache the LSTM output stream.
        We then run the Bayesian head `mc_samples` times on that cached
        stream, with each call drawing a fresh sample from every
        `BayesianLinearCustom`'s variational posterior.

        Using the law of total variance, the per-sample
        `(mu_s, log_var_s)` outputs are decomposed into:

        * `epistemic_var = Var_s[ mu_s ]`
            spread of the predictive means across MC samples - this is
            the *model* uncertainty (it shrinks if you give the model
            more training data).
        * `aleatoric_var = E_s[ exp(log_var_s) ]`
            mean of the predicted observation variance - this is the
            *data* noise the model believes is irreducible at this
            input (it does NOT shrink with more data).

        The total predictive variance is simply
        `epistemic_var + aleatoric_var`.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Yields `(X, y, v)` triples (use
            `SequenceDatasetMultiTask(..., return_valid_times=True)`).
        mc_samples : int
            Number of Monte-Carlo draws from the variational posterior
            (default 30).  Higher values give a less noisy estimate of
            `epistemic_var` at the cost of a linear increase in BNN
            forward passes; 20-50 is the usual sweet spot.
        return_samples : bool
            If True, additionally return the raw `(N, mc, output_dim)`
            tensors of per-sample `mu` and `log_var` from the final
            forecast step.  Useful for `compute_ucr_torch` or any
            downstream analysis that needs the full predictive
            ensemble.  Defaults to False to keep memory low.

        Returns
        -------
        mu : Tensor of shape `(N, output_dim)`
            Predictive mean at the most-recent forecast step.
        epistemic_var : Tensor of shape `(N, output_dim)`
            Variance of the per-sample means (model uncertainty).
        aleatoric_var : Tensor of shape `(N, output_dim)`
            Mean of the per-sample predicted variances (data noise).
        valid_times : Tensor of shape `(N, ...)`
            Per-item valid times as returned by the dataset.
        mu_samples, logvar_samples : Tensor of shape `(N, mc, output_dim)`
            Only returned when `return_samples=True`.
        """
        all_outputs = []
        all_valid_times = []
        self.eval()

        # Pass 1: run the deterministic encoder/decoder once per batch
        # and cache the LSTM output stream.
        with torch.no_grad():
            for batch_idx, (X, y, v) in enumerate(data_loader):
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
                all_valid_times.append(v)

        all_outputs = torch.cat(all_outputs, dim=0)
        valid_times = torch.cat(all_valid_times, axis=0)

        # Pass 2: MC-sample the Bayesian head.  `BayesianLinearCustom.forward`
        # re-samples weights on every call regardless of train/eval mode, so
        # each iteration produces a different `(mu, log_var)`.
        mus = []
        logvars = []
        with torch.no_grad():
            for _ in range(mc_samples):
                m, lv = self.bnn(all_outputs)
                mus.append(m)
                logvars.append(lv)

        # Stack along a new MC axis: (N, mc, T, output_dim).
        mus = torch.stack(mus, dim=1)
        logvars = torch.stack(logvars, dim=1)

        # Take the most-recent forecast step (t=-1) to match the caller
        # contract used by the inference engine.
        mus = mus[:, :, -1, :]          # (N, mc, output_dim)
        logvars = logvars[:, :, -1, :]  # (N, mc, output_dim)

        mu_pred = mus.mean(dim=1)
        epistemic_var = mus.var(dim=1, unbiased=False)
        aleatoric_var = torch.exp(logvars).mean(dim=1)

        if return_samples:
            return (
                mu_pred,
                epistemic_var,
                aleatoric_var,
                valid_times,
                mus,
                logvars,
            )
        return mu_pred, epistemic_var, aleatoric_var, valid_times