# Inference AI2ES Forecast Error

Sequence-to-sequence machine-learning models that predict the **forecast
error** of an NWP model (currently HRRR) for three meteorological
variables at every site of a mesonet:

- `t2m`     — 2 m air temperature error
- `u_total` — total wind-speed error
- `tp`      — total precipitation error

The error is defined as `target_error = NWP - mesonet observation`.
The trained networks consume the past 30 hours of HRRR forecasts and
mesonet observations, plus the future HRRR forecasts out to a
configurable forecast hour (1–18), and emit a per-station, per-hour
prediction of `target_error`.

Three model families ship in this repo:

| Model  | Architecture                                       | Output |
|--------|----------------------------------------------------|--------|
| LSTM   | Encoder–decoder LSTM seq2seq                       | point estimate of `target_error` |
| BNN    | LSTM seq2seq + Bayesian MLP head                   | predictive mean + decomposed epistemic / aleatoric variance |
| Hybrid | LSTM seq2seq + Vision Transformer (radiometer images) | point estimate of `target_error` (multi-modal) |

If you build on this work, please cite the project (see
[Citation](#citation)).

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [Installation](#installation)
3. [Data inputs](#data-inputs)
4. [Quickstart](#quickstart)
5. [Detailed pipeline](#detailed-pipeline)
   1. [Cleaning](#1-cleaning)
   2. [Training](#2-training)
   3. [Inference](#3-inference)
   4. [Evaluation](#4-evaluation)
6. [Output schemas](#output-schemas)
7. [Configuration via environment variables](#configuration-via-environment-variables)
8. [Architecture notes](#architecture-notes)
9. [Citation](#citation)
10. [License](#license)

---

## Repository layout

```
src/
├── pipeline.py                    # Hourly real-time inference (LSTM)
├── training_pipeline.py           # Top-level LSTM training driver
├── clean_lstm_data.py             # Refresh cleaned input parquets
├── eval_model_metrics.py          # Score predictions vs realised error
│
├── lstm_s2s_engine.py             # LSTM inference
├── engine_lstm_training.py        # LSTM training
├── bnn_s2s_engine.py              # BNN inference (mu, epi/ale uncertainty)
├── engine_bnn_training.py         # BNN training
├── hybrid_s2s_engine.py           # Hybrid (LSTM + ViT) inference
├── engine_hybrid_training.py      # Hybrid training
├── hybrid_fsdp_training.py        # Legacy FSDP multi-GPU reference
│
├── model_architecture/
│   ├── encode_decode_lstm.py      # Shared LSTM encoder + decoder + seq2seq head
│   ├── sequencer.py               # PyTorch Dataset: windowing, per-window
│   │                              # z-score, NYSM persistence
│   ├── bnn.py                     # BayesianLinearCustom + SimpleBNN +
│   │                              # ShallowLSTM_seq2seq_multi_task_bnn
│   ├── hybrid_vit_lstm.py         # LSTM + ViT fusion model
│   ├── hybrid_vit_encoder.py      # Vision-Transformer encoder
│   └── hybrid_fsdp.py             # FSDP-aware Hybrid (legacy reference)
│
├── model_data/
│   ├── prepare_lstm_data.py       # Pandas data prep for LSTM training
│   ├── prepare_lstm_data_rapids.py# cuDF data prep for LSTM/BNN inference
│   ├── prepare_hybrid_data.py     # Adds radiometer image-path columns
│   ├── normalization.py           # Feature mask used by the sequencer
│   ├── encode.py                  # Sin/cos cyclic time encodings
│   ├── get_error.py               # Computes the target_error column
│   ├── get_closest_nysm_stations.py # 4-station nearest-neighbour cluster
│   ├── hrrr_data.py / hrrr_data_rapids.py  # Read HRRR parquets
│   └── nysm_data.py / nysm_data_rapids.py  # Read NYSM parquets
│
├── data_cleaning/
│   ├── forecast_hr_parquet_builder.py     # Stitch per-init-time HRRR
│   ├── all_models_comparison_to_mesos_lstm.py # Align HRRR with NYSM
│   ├── get_resampled_nysm_data.py         # Resample raw 5-min NYSM
│   ├── build_profiler_images.py           # Per-hour radiometer .npy cache
│   └── output_post_process.py             # Reserved for future post-proc
│
└── visuals/                                # Plotting helpers (notebooks)
```

---

## Installation

The project targets **Python 3.9–3.12** with **PyTorch ≥ 2.0** and
**RAPIDS** (cuDF / CuPy) for the GPU-accelerated inference path.

### Conda (recommended, GPU)

```bash
conda create -n forecast_err python=3.10
conda activate forecast_err

# RAPIDS — pick the channel/version matching your CUDA toolkit
conda install -c rapidsai -c conda-forge -c nvidia \
    cudf=24.06 cuml=24.06 cupy

# PyTorch — pick the build matching your CUDA version (see pytorch.org)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Everything else
pip install pandas numpy scipy scikit-learn xarray cfgrib metpy \
    matplotlib seaborn cartopy geopandas pyarrow
```

### CPU-only / training-only

If you only need to train the LSTM (no real-time inference), the
RAPIDS dependencies are optional. Install plain PyTorch + pandas and
skip the `*_rapids.py` modules.

---

## Data inputs

You need three data sources on the local filesystem before any of
the pipelines will run:

1. **HRRR forecasts** (per init time, per forecast hour).
   Cleaned parquets are produced by upstream scripts and live under
   `/home/aevans/ai2es/cleaned/HRRR/{year}/{month}/`.

   Downloading raw HRRR is out of scope for this repo — see Lauriana
   Gaudet's companion notebooks
   <https://github.com/lgaudet/AI2ES-notebooks> (specifically
   `s3_download.ipynb` and `cleaning_bible-NYS-subset.ipynb`).

2. **NYSM observations** (5-minute netCDF, 1-hour and 3-hour parquet).
   Request data from the New York State Mesonet:
   <https://www.nysmesonet.org/weather/requestdata>.
   Resampled parquets are written by
   `data_cleaning/get_resampled_nysm_data.py`.

3. **Radiometer profiler images** *(only required for the Hybrid model)*.
   Per-station, per-hour numpy snapshots produced by
   `data_cleaning/build_profiler_images.py` and stored at
   `{root}/{year}/{stid}/{stid}_{year}_{MMDDHH}.npy`.

If your filesystem layout differs from the defaults baked into the
modules, override them via the
[environment variables](#configuration-via-environment-variables)
listed below. The current code also references a few absolute paths
inside `/home/aevans/...` for metadata CSVs (e.g. station→climate-
division mapping and land-use clusters) — search the codebase for
those paths and point them at your own copies.

---

## Quickstart

### 1. Refresh cleaned input parquets for the current month

```bash
cd src
python clean_lstm_data.py --fh 6 --year 2025 --month 4 --day 30 --model hrrr
```

This stitches HRRR + NYSM into the per-`fh` parquets that everything
downstream consumes. Run this hourly (or once per day, depending on
data freshness needs). For the full 1–18 forecast-hour sweep, call
`clean_lstm_data.main(now)` from your own scheduler.

### 2. Train one model family

```bash
# LSTM (point estimate)
python engine_lstm_training.py --clim_div "Hudson Valley" --device_id 0

# BNN (uncertainty-aware)
python engine_bnn_training.py --clim_div "Hudson Valley" --device_id 0

# Hybrid (multi-modal, requires radiometer images)
python engine_hybrid_training.py --clim_div "Hudson Valley" --device_id 0
```

Each script trains one model per `(climate_division, station, metvar,
forecast_hour)` combination and writes the encoder / decoder (and BNN
or ViT) weights under `MODELS/`, `MODELS/BNN/`, or `MODELS/HYBRID/`.

### 3. Run inference for the current hour

```bash
# Hourly cron-style real-time scoring (LSTM)
python pipeline.py

# Per-fh inference (any model)
python lstm_s2s_engine.py   --fh 6 --device_id 0
python bnn_s2s_engine.py    --fh 6 --device_id 0 --mc_samples 30
python hybrid_s2s_engine.py --fh 6 --device_id 0
```

Output parquets land under `FINAL_OUTPUT/` (LSTM), `FINAL_OUTPUT/BNN/`,
or `FINAL_OUTPUT/HYBRID/` — see [Output schemas](#output-schemas).

### 4. Evaluate predictions against realised error

After predictions for `valid_time = now` are on disk, compute the
realised error and append it to the per-(`fh`, `metvar`) error
parquets:

```bash
python eval_model_metrics.py
```

---

## Detailed pipeline

### 1. Cleaning

| Script                                              | Purpose |
|-----------------------------------------------------|---------|
| `data_cleaning/forecast_hr_parquet_builder.py`      | Stitch the 24 per-init-time HRRR parquets into a single per-day, per-`fh` parquet keyed by `valid_time`. |
| `data_cleaning/all_models_comparison_to_mesos_lstm.py` | Interpolate the gridded NWP fields to NYSM site lat/lons, mask out water grid cells, and emit the cleaned per-station HRRR parquet that `prepare_lstm_data*` consumes. |
| `data_cleaning/get_resampled_nysm_data.py`          | Resample raw 5-minute NYSM netCDFs to 1 h and 3 h hourly parquets, deriving `td` (dew point) and `mslp` (mean sea-level pressure) via MetPy. |
| `data_cleaning/build_profiler_images.py`            | (Hybrid only) Build the per-station, per-hour `.npy` radiometer image cache. |

`clean_lstm_data.py` is a thin orchestrator that runs (1) and (2) for
a given `fh` / date / month.

### 2. Training

All three training drivers share the same shape:

1. Pick climate division(s) → list of stations.
2. For each station, for each `metvar`, for each forecast hour
   (visited in random order so partial failures still cover the range):
   1. Build the wide HRRR + NYSM dataframe via the appropriate
      `prepare_*_data.py`.
   2. Wrap it in `model_architecture.sequencer.SequenceDatasetMultiTask`
      (or `SequenceDatasetMultiTaskHybrid` for the Hybrid model).
   3. Build the model, optionally warm-starting from existing weights.
   4. Train with `AdamW` + `ReduceLROnPlateau` + early stopping
      (patience 10, factor 0.1, max 50 epochs).
   5. Save the best-on-train weights to `MODELS/` (or model-specific
      subdir).

Loss functions:

| Model  | Loss |
|--------|------|
| LSTM   | `OutlierFocusedLoss(alpha=2.0)` — MAE up-weighted for outliers |
| BNN    | Gaussian NLL on `(mu, log_var)` (computed inside `bnn.train_model`) |
| Hybrid | `OutlierFocusedLoss(alpha=2.0)` |

For multi-GPU training there is a legacy FSDP reference in
`hybrid_fsdp_training.py` + `model_architecture/hybrid_fsdp.py`; it
depends on helper modules that aren't bundled here, so use it as a
guide rather than a runnable script.

### 3. Inference

All three inference drivers follow the same loop:

1. Load the year's NYSM parquet and the per-`fh` HRRR parquet on the
   GPU (`cuDF`).
2. For each NYSM station × `metvar`:
   1. Build the inference dataframe via `prepare_lstm_data_rapids`
      (or `prepare_hybrid_data_rapids`). At inference, HRRR is loaded
      for `[now − 29h, now + fh]` and NYSM only for `[now − 29h, now]`,
      then merged with a left join so the future HRRR rows survive
      with NaN NYSM columns.
   2. Wrap in the sequencer (NYSM persistence overwrites the future
      NaNs with the last known observation; per-window z-score is
      applied with past-only stats).
   3. Build the model and load the matching weights.
   4. Call `model.predict(...)`.
3. Append the prediction(s) to the per-(`fh`, `metvar`) output parquet.

The `pipeline.py` script wraps the LSTM cleaning + inference into one
hourly call. Equivalent wrappers for BNN / Hybrid are easy to add by
mirroring `pipeline.py` and swapping the engine import.

### 4. Evaluation

`eval_model_metrics.py` reads the prediction parquets, recomputes
`target_error` against the realised NYSM observation at `valid_time =
now`, and writes `(actual, model_output, difference)` rows to
`FINAL_OUTPUT/fh{fh}_{metvar}_error_metrics.parquet`. Plot helpers
under `src/visuals/` ingest these parquets to render the per-station
MAE map, the time-of-day / time-of-year heatmaps, and the bulk
scatter density plots.

---

## Output schemas

All output parquets are indexed by `(valid_time, stid)`.

### LSTM (`FINAL_OUTPUT/fh{fh}_{metvar}_inference_out.parquet`)

| column         | type    | meaning |
|----------------|---------|---------|
| `valid_time`   | datetime | inference anchor (start of the forecast horizon) |
| `stid`         | str     | NYSM station id |
| `model_output` | float   | predicted `target_error` (NWP units) |

### BNN (`FINAL_OUTPUT/BNN/fh{fh}_{metvar}_bnn_inference_out.parquet`)

| column           | type      | meaning |
|------------------|-----------|---------|
| `valid_time`     | datetime  | inference anchor |
| `stid`           | str       | NYSM station id |
| `mu`             | list[float] | predictive mean (`output_dim` per row) |
| `epistemic_var`  | list[float] | model uncertainty (variance of MC means) |
| `aleatoric_var`  | list[float] | data noise (mean predicted variance) |
| `total_var`      | list[float] | `epistemic_var + aleatoric_var` |

A 95 % predictive interval is `mu ± 1.96 * sqrt(total_var)`. Increase
`--mc_samples` for a less noisy `epistemic_var` estimate
(20–50 typical, ≥ 100 for offline calibration).

### Hybrid (`FINAL_OUTPUT/HYBRID/fh{fh}_{metvar}_hybrid_inference_out.parquet`)

| column         | type    | meaning |
|----------------|---------|---------|
| `valid_time`   | datetime |  |
| `stid`         | str     |  |
| `model_output` | float   | predicted `target_error` (NWP units) |

### Evaluation (`FINAL_OUTPUT/fh{fh}_{metvar}_error_metrics.parquet`)

| column         | type    | meaning |
|----------------|---------|---------|
| `valid_time`   | datetime |  |
| `stid`         | str     |  |
| `actual`       | float   | realised `target_error` from NYSM |
| `model_output` | float   | model prediction at the same time |
| `difference`   | float   | `model_output − actual` |

---

## Configuration via environment variables

Override the default filesystem layout without editing code:

| Variable                | Used by                          | Default |
|-------------------------|----------------------------------|---------|
| `LSTM_MODEL_DIR`        | `lstm_s2s_engine.py`, `engine_lstm_training.py` | `/home/aevans/inference_ai2es_forecast_err/MODELS` |
| `LSTM_OUTPUT_DIR`       | `lstm_s2s_engine.py`, `eval_model_metrics.py`   | `/home/aevans/inference/FINAL_OUTPUT` |
| `BNN_MODEL_DIR`         | `bnn_s2s_engine.py`              | `${LSTM_MODEL_DIR}/BNN` |
| `BNN_OUTPUT_DIR`        | `bnn_s2s_engine.py`              | `${LSTM_OUTPUT_DIR}/BNN` |
| `HYBRID_MODEL_DIR`      | `hybrid_s2s_engine.py`           | `${LSTM_MODEL_DIR}/HYBRID` |
| `HYBRID_OUTPUT_DIR`     | `hybrid_s2s_engine.py`           | `${LSTM_OUTPUT_DIR}/HYBRID` |
| `PROFILER_IMAGE_ROOT`   | `prepare_hybrid_data.py`         | `/home/aevans/nwp_bias/src/machine_learning/data/profiler_images` |

A few legacy paths (NYSM metadata CSVs, climate-division lookup,
station-cluster parquet) are still hard-coded inside the modules
under `model_data/`. Search for `/home/aevans/` and point them at
your own copies if you are running outside the original environment.

---

## Architecture notes

### Sequencer & windowing

`model_architecture/sequencer.py` is the data-flow heart of all three
models. Each `__getitem__` returns a window of
`sequence_length + forecast_steps` rows (default `30 + fh`) where:

- `x[: sequence_length]` is the real past HRRR + real past NYSM.
- `x[sequence_length :]` is the real future HRRR with the last 64
  NYSM columns *persisted* from the most recent past observation
  (the model never sees future NYSM truth).

This persistence pattern is intentional architecture — it lets the
encoder consume future NWP forecasts while keeping the NYSM future
unseen.

### Per-window z-score normalization

Z-scoring is performed inside the sequencer using statistics computed
**only from each window's past portion** and applied to the entire
window. This means:

- Train and inference share the *exact* same normalization with no
  persisted stats files.
- No future leakage into the per-window mean / std.
- Cyclic time encodings, geographic features, image paths, and the
  target itself are skipped (see `model_data/normalization.py`).

The target `target_error` is **never** normalized so the model can
learn the true error distribution.

### BNN uncertainty decomposition

`model_architecture/bnn.py` runs `mc_samples` Monte-Carlo passes
through the Bayesian head per inference call. Using the law of total
variance,

```
Var[y | x]  =  E_θ[ Var[y | x, θ] ]   +   Var_θ[ E[y | x, θ] ]
              `------ aleatoric ------'   `------ epistemic ------'
```

it returns the predictive mean alongside both components separately.

- Aleatoric uncertainty captures *data noise* the model thinks is
  irreducible at this input — it does not shrink with more training
  data.
- Epistemic uncertainty captures *model uncertainty* — it shrinks as
  the model sees more data.

---

## Citation

If this code, or models trained from it, contribute to your research,
please cite the project. \\ Please cite this paper: Evans, D. A., Sulia, K. J., Bassill, N. P., Thorncroft, C. D., Rothenberger, J. C., & Gaudet, L. C. (2025). Predicting Forecast Error for the HRRR Using LSTM Neural Networks: A Comparative Study Using New York and Oklahoma State Mesonets. ArXiv. https://arxiv.org/abs/2512.14898 \\
reference this GitHub
repository and the
[NSF AI2ES institute](https://www.ai2es.org/).)

This work was supported by AI2ES — the NSF AI Institute for
Research on Trustworthy AI in Weather, Climate, and Coastal
Oceanography.

## License

[MIT License](LICENSE) — Copyright (c) 2025 David Aaron Evans.
