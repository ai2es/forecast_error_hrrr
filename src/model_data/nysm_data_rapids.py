"""Load NYSM observation parquets via cuDF (GPU).

GPU-accelerated counterpart to `model_data.nysm_data`.  Returns a
pandas DataFrame so downstream code that expects pandas continues to
work uniformly.
"""

import cudf
import numpy as np


# Years of NYSM hourly parquets to load.
YEARS = np.arange(2023, 2026)


def load_nysm_data(start_year):
    """Read and concatenate hourly NYSM observation parquets (GPU).

    Parameters
    ----------
    start_year : int
        Currently informational; actual years come from `YEARS`.

    Returns
    -------
    pandas.DataFrame
        Concatenated NYSM observations.  `snow_depth` / `ta9m` missing
        values sentinel-filled with `-999`; other NaN rows dropped.
    """
    nysm_path = "/home/aevans/nwp_bias/data/nysm/"
    nysm_1H = []

    for year in YEARS:
        df = cudf.read_parquet(f"{nysm_path}nysm_1H_obs_{year}.parquet")
        df = df.reset_index()
        df = df.rename(columns={"time_1H": "valid_time"})
        nysm_1H.append(df)

    nysm_1H_obs = cudf.concat(nysm_1H)

    # Sentinel-fill the columns that may legitimately be missing.
    nysm_1H_obs["snow_depth"] = nysm_1H_obs["snow_depth"].fillna(-999)
    nysm_1H_obs["ta9m"] = nysm_1H_obs["ta9m"].fillna(-999)

    nysm_1H_obs = nysm_1H_obs.dropna()
    return nysm_1H_obs.to_pandas()
