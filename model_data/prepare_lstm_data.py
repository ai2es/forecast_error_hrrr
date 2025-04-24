import sys

sys.path.append("..")

import datetime
import pandas as pd
import numpy as np

from model_data import nysm_data, hrrr_data, encode, get_closest_nysm_stations


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


def prepare_lstm_data(nysm_df, hrrr_df, train=False):
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

    """
    NEED TO MAKE THIS FUNCTION JUST READING FROM A CSV, IT TAKES TOO LONG IN INFERENCE, get_closest_nysm_stations
    """
    stations = get_closest_nysm_stations.get_closest_stations(
        nysm_df, 6, station, "HRRR"
    )

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

    # fh2_, fh4_ = get_more_fh(fh, station, var, mytimes)
    # nysm_df1 = columns_drop_nysm(nysm_df1)
    master_df = dataframe_wrapper(stations, hrrr_df1)

    nysm_df1 = nysm_df1.drop(
        columns=[
            "index",
        ]
    )
    master_df2 = dataframe_wrapper(stations, nysm_df1)

    # combine HRRR + NYSM data on time
    master_df = master_df.merge(master_df2, on="valid_time", suffixes=(None, f"_xab"))

    # Calculate the error using NWP data.
    # options are {
    # t2m, mslma, tp, u_total
    # }
    the_df = get_error.nwp_error(var, station, master_df)
    valid_times = the_df["valid_time"].tolist()
    # encode day of year to be cylcic
    the_df = encode.encode(the_df, "valid_time", 366)
    # drop columns
    the_df = the_df[the_df.columns.drop(list(the_df.filter(regex="station")))]

    new_df = the_df.drop(columns="valid_time")

    # normalize data
    cols = [
        "valid_time_cos",
        "valid_time_sin",
        "valid_time_cos_clock",
        "valid_time_sin_clock",
        "lat",
        "lon",
        "elev",
        "lulc",
        "slope",
    ]
    for k, r in new_df.items():
        if k in cols or any(sub in k for sub in cols) or "images" in k:
            print(k)
            continue
        else:
            means = st.mean(new_df[k])
            stdevs = st.pstdev(new_df[k])
            new_df[k] = (new_df[k] - means) / stdevs

    # features = [c for c in new_df.columns if c != 'target_error']
    features = [
        c
        for c in new_df.columns
        if c != "target_error" and c != "valid_time" and "images" not in c
    ]
    # # get radiometer images for ViT
    # image_list_cols = [c for c in new_df.columns if "images" in c]
    lstm_df = new_df.copy()
    target_sensor = "target_error"
    forecast_lead = 0
    target = f"{target_sensor}"
    lstm_df.insert(loc=(0), column=target, value=lstm_df[target_sensor])
    lstm_df = lstm_df.drop(columns=[target_sensor]).fillna(0)

    return lstm_df, features, stations, target, valid_times
