"""Read concatenated HRRR forecast parquets (pandas).

The cleaned HRRR parquets are produced upstream by
``data_cleaning/all_models_comparison_to_mesos_lstm.main`` and live
under ``data/hrrr_data/fh{fh}/``.  This module concatenates all
months found across `YEARS` into a single in-memory dataframe and
returns it, ready to be filtered by date in
`engine_lstm_training.main`.

For the GPU-accelerated inference path, see
`model_data.hrrr_data_rapids` which mirrors this module but uses
cuDF.
"""

import gc
import os

import numpy as np
import pandas as pd


# Years of cleaned HRRR parquets to load.  Add new years here as more
# data becomes available.
YEARS = ["2023", "2024", "2025"]


def read_hrrr_data(fh, year):
    """Read and concatenate HRRR parquets for forecast hour `fh`.

    Parameters
    ----------
    fh : str
        Two-digit forecast hour string (e.g. "06").
    year : int
        Currently informational; the actual years loaded are taken
        from the module-level `YEARS` constant.  Kept in the signature
        for API parity with the `_rapids` variant.

    Returns
    -------
    pandas.DataFrame
        Concatenated HRRR forecasts for all months / years available
        on disk for `fh`.  Missing values are filled with `-999`.
    """
    savedir = f"/home/aevans/nwp_bias/src/machine_learning/data/hrrr_data/fh{fh}/"

    hrrr_fcast_and_error = []
    for y in YEARS:
        for month in np.arange(1, 13):
            str_month = str(month).zfill(2)
            path = (
                f"{savedir}HRRR_{y}_{str_month}_direct_compare_to_nysm_sites_"
                "mask_water.parquet"
            )
            if os.path.exists(path):
                hrrr_fcast_and_error.append(pd.read_parquet(path))
            gc.collect()

    hrrr_fcast_and_error_df = pd.concat(hrrr_fcast_and_error)
    hrrr_fcast_and_error_df = hrrr_fcast_and_error_df.reset_index().fillna(-999)

    # Older parquets used to carry an unused `new_tp` column.
    if "new_tp" in hrrr_fcast_and_error_df.columns:
        hrrr_fcast_and_error_df = hrrr_fcast_and_error_df.drop(columns="new_tp")

    return hrrr_fcast_and_error_df
