import pandas as pd
import os
import numpy as np
import gc


def read_hrrr_data(fh, year):
    """
    Reads and concatenates parquet files containing forecast and error data for HRRR weather models
    for the years 2018 to 2022.

    Returns:
        pandas.DataFrame: of hrrr weather forecast information for each NYSM site.
    """

    years = [str(int(year - 1)), str(int(year))]
    savedir = f"/home/aevans/nwp_bias/src/machine_learning/data/hrrr_data/fh{fh}/"

    # create empty lists to hold dataframes for each model
    hrrr_fcast_and_error = []

    # loop over years and read in parquet files for each model
    for year in years:
        for month in np.arange(1, 13):
            str_month = str(month).zfill(2)
            if (
                os.path.exists(
                    f"{savedir}HRRR_{year}_{str_month}_direct_compare_to_nysm_sites_mask_water.parquet"
                )
                == True
            ):
                hrrr_fcast_and_error.append(
                    pd.read_parquet(
                        f"{savedir}HRRR_{year}_{str_month}_direct_compare_to_nysm_sites_mask_water.parquet"
                    )
                )
            else:
                continue
            gc.collect()

    # concatenate dataframes for each model
    hrrr_fcast_and_error_df = pd.concat(hrrr_fcast_and_error)
    hrrr_fcast_and_error_df = hrrr_fcast_and_error_df.reset_index().fillna(-999)

    if "new_tp" in hrrr_fcast_and_error_df.columns:
        hrrr_fcast_and_error_df = hrrr_fcast_and_error_df.drop(columns="new_tp")

    # return dataframes for each model
    return hrrr_fcast_and_error_df
