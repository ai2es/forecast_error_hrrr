import sys

sys.path.append("..")

import datetime
import statistics as st
import pandas 
import cudf
import cupy as cp

from model_data import (
    encode,
    get_closest_nysm_stations,
    get_error,
)


# Collect station-wise frames
def add_suffix(df, station):
    cols = ["valid_time", "time"]
    return df.rename(
        columns={c: f"{c}_{station}" if c not in cols else c for c in df.columns}
    )


def dataframe_wrapper(stations, df):
    master_df = add_suffix(df[df["station"] == stations[0]], stations[0])
    for station in stations[1:]:
        df1 = add_suffix(df[df["station"] == station], station)
        master_df = master_df.merge(
            df1, on="valid_time", suffixes=(None, f"_{station}")
        )
    return master_df


def prepare_lstm_data(nysm_df, hrrr_df, station, metvar, train=False):
    # Convert to cuDF if not already
    if not isinstance(nysm_df, cudf.DataFrame):
        nysm_df = cudf.from_pandas(nysm_df)
    if not isinstance(hrrr_df, cudf.DataFrame):
        hrrr_df = cudf.from_pandas(hrrr_df)

    hrrr_df = hrrr_df.sort_values("valid_time")
    hrrr_df["valid_time"] = cudf.to_datetime(hrrr_df["valid_time"])

    current_time = hrrr_df["valid_time"].max()

    if not train:
        mask = (
            hrrr_df["valid_time"]
            >= current_time - cudf.utils.dtypes._timedelta_from_str("29h")
        ) & (hrrr_df["valid_time"] <= current_time)
        filtered_times = hrrr_df.loc[mask, "valid_time"]
        mytimes = filtered_times.to_pandas().tolist()
        nysm_df = nysm_df[nysm_df["valid_time"].isin(mytimes)]
    else:
        mytimes = hrrr_df["valid_time"].to_pandas().tolist()

    # Load geo cluster data
    geo_df = cudf.read_csv("/home/aevans/nwp_bias/src/landtype/data/lstm_clusters.csv")
    stations = get_closest_nysm_stations.get_closest_stations_csv(station)
    hrrr_df1 = hrrr_df[hrrr_df["station"].isin(stations)]
    nysm_df1 = nysm_df[nysm_df["station"].isin(stations)]
    geo_df1 = geo_df[geo_df["station"].isin(stations)]

    # Map geographic features
    for col in ["lulc_cat", "elev_cat", "slope_cat"]:
        geo_dict = dict(
            zip(geo_df[col].to_pandas().index, geo_df[col].to_pandas().values)
        )
        hrrr_df1[col] = (
            geo_df.set_index("station")[col]
            .loc[hrrr_df1["station"]]
            .reset_index(drop=True)
        )

    hrrr_df1 = hrrr_df1.drop(
        columns=["index", "lead time", "lsm", "latitude", "longitude", "time", "orog"]
    )

    master_df = dataframe_wrapper(stations, hrrr_df1)
    master_df2 = dataframe_wrapper(stations, nysm_df1)
    master_df = master_df.merge(master_df2, on="valid_time", suffixes=(None, f"_xab"))

    the_df = get_error.nwp_error(metvar, station, master_df.to_pandas())
    valid_times = the_df["valid_time"].tolist()
    the_df = encode.encode(the_df, "valid_time", 366)
    the_df = the_df[the_df.columns.drop(list(the_df.filter(regex="station")))]

    new_df = the_df.drop(columns="valid_time")
    new_df = cudf.from_pandas(new_df)  # Move back to cuDF

    # Normalize data
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

    for k in new_df.columns:
        if k in cols or any(sub in k for sub in cols) or "images" in k:
            continue
        col = new_df[k].dropna()
        if len(col) == 0:
            continue
        mean = col.mean()
        std = col.std(ddof=0)
        if std != 0:
            new_df[k] = (new_df[k] - mean) / std

    features = [c for c in new_df.columns if c != "target_error" and "images" not in c]
    target = "target_error"
    lstm_df = new_df.copy()
    lstm_df = lstm_df.dropna()

    return lstm_df.to_pandas(), features, stations, target, valid_times
