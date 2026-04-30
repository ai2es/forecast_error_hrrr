"""Load NYSM observation parquets (pandas).

The hourly NYSM observation parquets are produced upstream by
`data_cleaning/get_resampled_nysm_data.py` and live in `nysm_path`.
This module concatenates all years in `YEARS` into one DataFrame.

For the GPU-accelerated inference path, see
`model_data.nysm_data_rapids`.
"""

import numpy as np
import pandas as pd


# Years of NYSM hourly parquets to load.
YEARS = np.arange(2023, 2026)


def load_nysm_data(start_year):
    """Read and concatenate hourly NYSM observation parquets.

    Parameters
    ----------
    start_year : int
        Currently informational; the actual years loaded are taken
        from the module-level `YEARS` constant.  The argument is kept
        for API parity with the cuDF variant.

    Returns
    -------
    pandas.DataFrame
        Concatenated NYSM observations.  `snow_depth` and `ta9m`
        missing values are sentinel-filled with `-999`; any other rows
        with NaNs are dropped.  The `time_1H` column is renamed to
        `valid_time` to match HRRR's convention.
    """
    nysm_path = "/home/aevans/nwp_bias/data/nysm/"

    nysm_1H = []
    for year in YEARS:
        df = pd.read_parquet(f"{nysm_path}nysm_1H_obs_{year}.parquet")
        df.reset_index(inplace=True)
        df = df.rename(columns={"time_1H": "valid_time"})
        nysm_1H.append(df)

    nysm_1H_obs = pd.concat(nysm_1H)

    # Sentinel-fill columns that legitimately can be missing
    # (e.g. snow depth in summer, 9m temperature for older sites).
    nysm_1H_obs.fillna({"snow_depth": -999}, inplace=True)
    nysm_1H_obs.fillna({"ta9m": -999}, inplace=True)

    # Drop rows with any other missing values.
    nysm_1H_obs.dropna(inplace=True)
    return nysm_1H_obs
