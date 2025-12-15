import sys

sys.path.append("..")

from datetime import timedelta, datetime
import statistics as st
import pandas as pd
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

    # Get max valid time (cuDF Timestamp)
    current_time = hrrr_df["valid_time"].max()

    if not train:
        # Create timedelta using pandas (this is okay for cuDF arithmetic)
        time_window = pd.Timedelta(hours=30)

        # Do the arithmetic with cuDF timestamps
        start_time = current_time - time_window

        # Create mask
        mask = (hrrr_df["valid_time"] >= start_time) & (
            hrrr_df["valid_time"] <= current_time
        )

        # Filter and convert to pandas to get list of timestamps
        filtered_times = hrrr_df.loc[mask, "valid_time"]
        mytimes = filtered_times.to_pandas().tolist()

        # Apply to NYSM DataFrame
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

    for k, r in new_df.items():
        if k in cols or any(sub in k for sub in cols) or "images" in k:
            continue
        else:
            means = st.mean(new_df[k])
            stdevs = st.pstdev(new_df[k])
            new_df[k] = (new_df[k] - means) / stdevs

    features = [c for c in new_df.columns if c != "target_error" and "images" not in c]
    target = "target_error"
    lstm_df = new_df.copy()
    # lstm_df = lstm_df.dropna()

    return lstm_df, features, stations, target, valid_times
