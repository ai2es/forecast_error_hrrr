import sys

sys.path.append("..")

from datetime import datetime
import pandas as pd
import numpy as np
import statistics as st

from model_data import (
    nysm_data,
    hrrr_data,
    encode,
    get_closest_nysm_stations,
    get_error,
)


def create_geo_dict(geo_df, c, df1):
    geo_dict = dict(zip(geo_df["station"], geo_df[c]))

    # Map the 'station' values from df1 to the corresponding values in geo_dict
    df1[c] = df1["station"].map(geo_dict)
    return df1


def columns_drop_hrrr(df):
    df = df.drop(
        columns=[
            # "level_0",
            "index",
            "lead time",
            "lsm",
            "latitude",
            "longitude",
            "time",
            "orog",
        ]
    )
    return df


def columns_drop_nysm(df):
    df = df.drop(
        columns=[
            "ta9m",
            "td",
            "relh",
            "srad",
            "pres",
            "mslp",
            "wspd_sonic_mean",
            "wspd_sonic",
            "wmax_sonic",
            "wdir_sonic",
            "snow_depth",
            "precip_total",
        ]
    )
    return df


def add_suffix(master_df, station):
    cols = ["valid_time", "time"]
    master_df = master_df.rename(
        columns={c: c + f"_{station}" for c in master_df.columns if c not in cols}
    )
    return master_df


def dataframe_wrapper(stations, df):
    master_df = df[df["station"] == stations[0]]
    master_df = add_suffix(master_df, stations[0])
    for station in stations[1:]:
        df1 = df[df["station"] == station]
        df1 = add_suffix(df1, station)
        master_df = master_df.merge(
            df1, on="valid_time", suffixes=(None, f"_{station}")
        )
    return master_df


def prepare_lstm_data(nysm_df, hrrr_df, station, metvar, train=False):
    # Filter NYSM data to match valid times from HRRR data
    hrrr_df = hrrr_df.sort_values("valid_time")
    hrrr_df["valid_time"] = pd.to_datetime(hrrr_df["valid_time"])

    # Get the latest available time (current hour)
    current_time = hrrr_df["valid_time"].max()

    if train == False:
        # Filter the previous 30 hours including the current hour
        mask = (hrrr_df["valid_time"] >= current_time - pd.Timedelta(hours=29)) & (
            hrrr_df["valid_time"] <= current_time
        )
        filtered_times = hrrr_df.loc[mask, "valid_time"]
        # Convert to list if needed
        mytimes = filtered_times.tolist()
        nysm_df = nysm_df[nysm_df["valid_time"].isin(mytimes)]
    else:
        # Filter NYSM data to match valid times from HRRR data
        mytimes = hrrr_df["valid_time"].tolist()

    # load geo cats
    geo_df = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/lstm_clusters.csv")
    stations = get_closest_nysm_stations.get_closest_stations_csv(station)

    hrrr_df1 = hrrr_df[hrrr_df["station"].isin(stations)]
    nysm_df1 = nysm_df[nysm_df["station"].isin(stations)]
    geo_df1 = geo_df[geo_df["station"].isin(stations)]

    hrrr_df1["lulc_cat"] = geo_df1["lulc_cat"]
    hrrr_df1["elev_cat"] = geo_df1["elev_cat"]
    hrrr_df1["slope_cat"] = geo_df1["slope_cat"]

    # add geo columns
    hrrr_df1 = create_geo_dict(geo_df, "lulc_cat", hrrr_df1)
    hrrr_df1 = create_geo_dict(geo_df, "elev_cat", hrrr_df1)
    hrrr_df1 = create_geo_dict(geo_df, "slope_cat", hrrr_df1)

    # format for LSTM
    hrrr_df1 = columns_drop_hrrr(hrrr_df1)

    master_df = dataframe_wrapper(stations, hrrr_df1)
    master_df2 = dataframe_wrapper(stations, nysm_df1)

    # combine HRRR + NYSM data on time
    master_df = master_df.merge(master_df2, on="valid_time", suffixes=(None, f"_xab"))

    # Calculate the error using NWP data.
    # options are {
    # t2m, mslma, tp, u_total
    # }
    the_df = get_error.nwp_error(metvar, station, master_df)

    return the_df


def main(now):
    year = now.year
    nysm_ = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
    stations = nysm_["stid"].unique()
    nysm_df = nysm_data.load_nysm_data(year)
    nysm_df = nysm_df.reset_index(drop=True)
    outpath = "/home/aevans/inference/FINAL_OUTPUT"
    for fh in np.arange(1, 19):
        for metvar in ["t2m", "u_total", "tp"]:
            guess_df = pd.read_parquet(
                f"{outpath}/fh{fh}_{metvar}_inference_out.parquet"
            ).reset_index()
            error_df = pd.read_parquet(
                f"{outpath}/fh{fh}_{metvar}_error_metrics.parquet"
            ).reset_index()
            outs = []
            for station in stations:
                hrrr_df = hrrr_data.read_hrrr_data(str(fh).zfill(2), year)
                known_df = prepare_lstm_data(nysm_df, hrrr_df, station, metvar)
                known_df = known_df[known_df["valid_time"] == now]
                known = known_df["target_error"].iloc[0]
                filtered_lstm = guess_df[guess_df["valid_time"] == now]
                lstm_output = filtered_lstm["model_output"].iloc[0]
                diff = lstm_output - known
                # save output
                outs.append(
                    {
                        "valid_time": now,
                        "stid": station,
                        "actual": known,
                        "model_output": lstm_output,
                        "difference": diff,
                    }
                )
            child = pd.DataFrame(outs)
            error_df = pd.concat([error_df, child], ignore_index=True)
            error_df.set_index(["valid_time", "stid"], inplace=True)
            error_df.to_parquet(f"{outpath}/fh{fh}_{metvar}_error_metrics.parquet")


if __name__ == "__main__":
    now = datetime.now()
    now = now.replace(minute=0, second=0, microsecond=0)
    # try:
    main(now)
    # except Exception as e:
    print("🔥 Runtime Error:", str(e))
