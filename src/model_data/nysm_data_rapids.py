import cudf
import numpy as np
import pandas


def load_nysm_data(start_year):
    nysm_path = "/home/aevans/nwp_bias/data/nysm/"
    nysm_1H = []

    for year in np.arange(2023, 2026):  # can replace with dynamic range if needed
        df = cudf.read_parquet(f"{nysm_path}nysm_1H_obs_{year}.parquet")
        df = df.reset_index()
        df = df.rename(columns={"time_1H": "valid_time"})
        nysm_1H.append(df)

    nysm_1H_obs = cudf.concat(nysm_1H)

    # Fill specific missing values
    nysm_1H_obs["snow_depth"] = nysm_1H_obs["snow_depth"].fillna(-999)
    nysm_1H_obs["ta9m"] = nysm_1H_obs["ta9m"].fillna(-999)

    # Drop all remaining nulls
    nysm_1H_obs = nysm_1H_obs.dropna()

    return nysm_1H_obs.to_pandas()  # convert back to pandas for compatibility
