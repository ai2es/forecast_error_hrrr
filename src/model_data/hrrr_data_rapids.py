"""Read concatenated HRRR forecast parquets via cuDF (GPU).

GPU-accelerated counterpart to `model_data.hrrr_data`.  Used by the
inference engines (`lstm_s2s_engine.py`, `bnn_s2s_engine.py`,
`hybrid_s2s_engine.py`) when reading large per-forecast-hour parquets
for the latest year.

Returns a pandas DataFrame so downstream code that expects pandas
(e.g. `prepare_lstm_data_rapids`'s preliminary helpers) continues to
work uniformly.
"""

import gc
import os

import cudf
import numpy as np


# Years of cleaned HRRR parquets to load.  Add new years here as more
# data becomes available.
YEARS = ["2023", "2024", "2025"]


def read_hrrr_data(fh, year):
    """Read HRRR parquets for forecast hour `fh` (GPU-accelerated).

    Parameters
    ----------
    fh : str
        Two-digit forecast hour string (e.g. "06").
    year : int
        Currently informational; the actual years loaded are taken
        from the module-level `YEARS` constant.

    Returns
    -------
    pandas.DataFrame
        Concatenated HRRR forecasts for all months / years available
        on disk for `fh`, returned in pandas form for downstream
        compatibility.  Missing values are filled with `-999`.
    """
    savedir = f"/home/aevans/nwp_bias/src/machine_learning/data/hrrr_data/fh{fh}/"

    hrrr_fcast_and_error = []
    for y in YEARS:
        for month in np.arange(1, 13):
            str_month = str(month).zfill(2)
            filename = (
                f"{savedir}HRRR_{y}_{str_month}_direct_compare_to_nysm_sites_"
                "mask_water.parquet"
            )
            if os.path.exists(filename):
                df = cudf.read_parquet(filename).reset_index()
                hrrr_fcast_and_error.append(df)
                gc.collect()

    if not hrrr_fcast_and_error:
        return cudf.DataFrame().to_pandas()

    hrrr_fcast_and_error_df = cudf.concat(hrrr_fcast_and_error)
    hrrr_fcast_and_error_df = hrrr_fcast_and_error_df.reset_index(drop=True).fillna(
        -999
    )

    # Older parquets carried an unused `new_tp` column.
    if "new_tp" in hrrr_fcast_and_error_df.columns:
        hrrr_fcast_and_error_df = hrrr_fcast_and_error_df.drop(columns="new_tp")

    return hrrr_fcast_and_error_df.to_pandas()
