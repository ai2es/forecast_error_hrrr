import cudf
import os
import numpy as np
import gc


def read_hrrr_data(fh, year):
    years = ["2023", "2024", "2025"]
    savedir = f"/home/aevans/nwp_bias/src/machine_learning/data/hrrr_data/fh{fh}/"

    hrrr_fcast_and_error = []

    for year in years:
        for month in np.arange(1, 13):
            str_month = str(month).zfill(2)
            filename = f"{savedir}HRRR_{year}_{str_month}_direct_compare_to_nysm_sites_mask_water.parquet"
            if os.path.exists(filename):
                df = cudf.read_parquet(filename).reset_index()
                hrrr_fcast_and_error.append(df)
                gc.collect()

    if not hrrr_fcast_and_error:
        return cudf.DataFrame().to_pandas()  # return empty if nothing found

    hrrr_fcast_and_error_df = cudf.concat(hrrr_fcast_and_error)
    hrrr_fcast_and_error_df = hrrr_fcast_and_error_df.reset_index(drop=True).fillna(
        -999
    )

    if "new_tp" in hrrr_fcast_and_error_df.columns:
        hrrr_fcast_and_error_df = hrrr_fcast_and_error_df.drop(columns="new_tp")

    return (
        hrrr_fcast_and_error_df.to_pandas()
    )  # maintain pandas downstream compatibility
