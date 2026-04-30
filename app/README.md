# Forecast Error App

A self-contained, GPU-ready suite of Jupyter notebooks that lets you train and
run a forecast-error correction model for **any region in the continental US** —
not just New York State.

---

## Quick-start (5 steps)

```bash
# 1. Install dependencies (from the repo root)
pip install herbie-data requests folium ipywidgets pyyaml scikit-learn \
            pandas pyarrow metpy torch

# 2. Edit the bounding box and date range in config.yaml
#    (see "Configuration reference" below)

# 3. Open the notebooks in order
jupyter lab app/notebooks/
```

Then run the four notebooks in sequence:

| Notebook | What it does |
|---|---|
| `01_setup_data.ipynb` | Download ASOS + HRRR, QC, build station clusters |
| `02_train.ipynb` | Train the model on your date range |
| `03_evaluate.ipynb` | Score the test set, visualise errors |
| `04_inference.ipynb` | Run real-time or retrospective predictions |

---

## Directory layout

```
app/
├── config.yaml              ← edit this before running anything
├── utils/
│   ├── config_loader.py     load + validate config.yaml
│   ├── bbox_filter.py       spatial filter for station metadata
│   └── engine_bridge.py     glue between app/ and src/ engines
├── data/
│   ├── fetch_hrrr.py        download HRRR via herbie-data
│   ├── fetch_asos.py        download ASOS via IEM REST API
│   └── build_station_clusters.py  nearest-neighbour lookup parquet
└── notebooks/
    ├── 01_setup_data.ipynb
    ├── 02_train.ipynb
    ├── 03_evaluate.ipynb
    └── 04_inference.ipynb
```

All outputs go to `app/output/` (configurable):

```
app/output/
├── hrrr_raw/            raw grib2 files (can be deleted after staging)
├── parquets/
│   ├── hrrr_data/fh{N}/ cleaned HRRR parquets per forecast hour
│   └── mesonet/         hourly observation parquets
└── models/
    ├── lookups/         station metadata + cluster parquet
    └── *.pth            trained encoder / decoder weights
    results/             inference output + error metric parquets
```

---

## Configuration reference

Every user-facing setting lives in `app/config.yaml`.

### Bounding box

```yaml
bbox:
  lat_min: 40.0
  lat_max: 43.5
  lon_min: -80.0
  lon_max: -71.5
```

Set these to any rectangular region you have data for.  The app
automatically finds all ASOS stations inside the box and downloads
HRRR grid points nearest to each of them.

### Data sources

```yaml
data:
  mesonet_source: asos       # "asos" or "local"
  forecast_hours: [1, 3, 6, 12, 18]
  metvars: [t2m, u_total, tp]
```

**`mesonet_source: asos`** — downloads observations from the
[IEM ASOS network](https://mesonet.agron.iastate.edu/request/asos/).
No credentials required.

**`mesonet_source: local`** — use pre-existing hourly parquets (e.g.
a private mesonet or already-downloaded ASOS data).  Set
`mesonet_local_dir` to the directory containing files named
`mesonet_1H_obs_{YYYY}.parquet`.  The files must have a multi-index
`(station, time_1H)` and the columns listed under
"Local mesonet schema" below.

**`forecast_hours`** — any subset of 1–18.  Each value trains and
runs a separate model, so narrow this list if you only care about
specific lead times.

**`metvars`** — variables the model predicts the error for.
Supported values: `t2m` (2 m temperature), `u_total` (wind speed),
`tp` (precipitation).

### Model type

```yaml
model: lstm    # lstm | bnn | hybrid
```

| Value | Architecture | Output |
|---|---|---|
| `lstm` | Encoder-decoder LSTM | Point forecast of error |
| `bnn` | LSTM + Bayesian MLP head | Mean + epistemic + aleatoric variance |
| `hybrid` | LSTM + Vision Transformer | Point forecast (requires radiometer images) |

### Training split

```yaml
training:
  start_date: "2023-01-01"
  end_date:   "2024-12-31"
  train_frac: 0.70
  val_frac:   0.15
  test_frac:  0.15
```

The three fractions must sum to 1.0.  The date range is divided
*chronologically* — train comes first, then validation, then test.
No shuffling is applied.

### Hyper-parameters

```yaml
training:
  epochs:          50
  batch_size:      1000
  num_layers:      3
  learning_rate:   5.0e-5
  weight_decay:    0.0
  sequence_length: 30     # hours of past observations fed to the encoder
  device_id:       0      # GPU index (0 = first GPU)
  mc_samples:      30     # BNN only: Monte-Carlo samples for uncertainty
```

---

## Changing the target variable

The model predicts `target_error = observed_value − HRRR_forecast`.
To switch the target variable, change `data.metvars` in `config.yaml`.

If you want to add a new meteorological variable that is not yet in
`src/model_data/get_error.py`, you need to add one entry to the
`VARS_DICT` mapping in that file:

```python
VARS_DICT = {
    "t2m":     ("t2m",  "tair"),    # (HRRR column, ASOS/mesonet column)
    "u_total": ("u_total", "wspd_sonic"),
    "tp":      ("tp",   "precip_total"),
    "my_var":  ("hrrr_col_name", "obs_col_name"),  # ← add here
}
```

No other source changes are needed.

---

## Changing HRRR variables

The set of HRRR fields extracted from each grib2 file is controlled by
`_SEARCH_SPECS` at the top of `app/data/fetch_hrrr.py`.  Each entry is
a dict with a `search` key (a herbie/cfgrib search string) and a `var`
key (the output column name):

```python
_SEARCH_SPECS = [
    {"search": ":TMP:2 m above",   "var": "t2m"},
    {"search": ":UGRD:10 m above", "var": "u10"},
    # add more fields here, e.g.:
    {"search": ":SNOD:surface",    "var": "snowd"},
]
```

The variable name you choose for `var` must match the column name that
`src/model_data/prepare_lstm_data.py` expects (or be in the
`columns_drop_hrrr` drop list if it is not needed as a feature).

---

## Using a local mesonet instead of ASOS

1. Set `data.mesonet_source: local` in `config.yaml`.
2. Set `data.mesonet_local_dir` to the folder containing your parquets.
3. Set `data.station_meta_csv` to a CSV with columns `station`, `lat`, `lon`.

### Local mesonet schema

Your hourly parquets must have a multi-index `(station, time_1H)` and
these columns (fill unavailable columns with `-999`):

| Column | Description |
|---|---|
| `lat`, `lon`, `elev` | Station coordinates and elevation (m) |
| `tair` | 2 m air temperature (°C) |
| `ta9m` | 9 m air temperature (°C) — fill with `-999` if absent |
| `td` | Dew point temperature (°C) |
| `relh` | Relative humidity (%) |
| `srad` | Downwelling shortwave radiation (W m⁻²) — fill with `-999` if absent |
| `pres` | Station pressure (hPa) |
| `mslp` | Mean sea-level pressure (hPa) |
| `wspd_sonic` | Instantaneous wind speed (m s⁻¹) |
| `wspd_sonic_mean` | Mean wind speed over the hour (m s⁻¹) |
| `wmax_sonic` | Maximum wind gust over the hour (m s⁻¹) |
| `wdir_sonic` | Wind direction (degrees) |
| `precip_total` | Accumulated precipitation over the hour (mm) |
| `snow_depth` | Snow depth (mm) — fill with `-999` if absent |

The `time_1H` index must be timezone-naive UTC timestamps.

---

## Tips

- **Resume-friendly**: `fetch_hrrr.py` and `fetch_asos.py` skip
  already-written parquets.  Kill a long download and restart without
  re-downloading.
- **Reduce scope for testing**: set `forecast_hours: [1]` and
  `metvars: [t2m]` in `config.yaml` to do a single fast end-to-end run.
- **GPU requirement**: the training and inference engines require a
  CUDA GPU.  The data-setup notebook (`01_setup_data.ipynb`) and the
  evaluation notebook (`03_evaluate.ipynb`) run fine on CPU.
- **Raw grib2 files**: the `hrrr_raw_dir` folder can grow very large.
  Once `01_setup_data.ipynb` has finished staging all parquets you can
  delete it to reclaim disk space.
- **BNN uncertainty**: the `bnn` model outputs three uncertainty channels
  per station per forecast step.  `03_evaluate.ipynb` renders histograms
  and `04_inference.ipynb` renders per-channel maps automatically when
  `model: bnn` is set in `config.yaml`.
