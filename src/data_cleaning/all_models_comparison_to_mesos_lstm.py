import glob
import multiprocessing as mp
import os
import warnings
import cfgrib
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import time
from metpy.units import units
from scipy import interpolate
from sklearn.neighbors import BallTree
from multiprocessing import Process
import argparse

from sklearn import preprocessing
from sklearn import utils

import matplotlib.pyplot as plt

import datetime as datetime
from datetime import timedelta
import gc
import time
import pickle
import cudf


def load_nysm_data(year):
    nysm_path = "/home/aevans/nwp_bias/data/nysm/"
    nysm_1H_obs = cudf.read_parquet(f"{nysm_path}nysm_1H_obs_{year}.parquet")
    nysm_1H_obs = nysm_1H_obs.fillna(-999)
    return nysm_1H_obs.to_pandas()  # ← convert back to pandas


def read_data_ny(model, month, year, fh):
    cleaned_data_path = f"/home/aevans/ai2es/lstm/{model.upper()}/fh_{fh}/"
    filelist = sorted(glob.glob(f"{cleaned_data_path}{year}/{month}/*.parquet"))

    li = []
    for filename in filelist:
        try:
            df_temp = cudf.read_parquet(filename)
            df_temp.reset_index(drop=True, inplace=True)
            li.append(df_temp)
        except Exception as e:
            print(f"Skipping file {filename}: {e}")
            continue

    df = cudf.concat(li)
    return df.to_pandas()  # ← convert back to pandas


def read_data_ny_v2(model, month, year, fh):
    cleaned_data_path = f"/home/aevans/ai2es/lstm/{model.upper()}/fh_{fh}/"
    filelist = sorted(glob.glob(f"{cleaned_data_path}{year}/{month}/*.parquet"))

    li = []
    for filename in filelist:
        try:
            df_temp = cudf.read_parquet(
                filename,
                columns=["valid_time", "time", "tp", "latitude", "longitude"],
            )
            df_temp.reset_index(drop=True, inplace=True)
            li.append(df_temp)
        except Exception as e:
            print(f"Skipping file {filename}: {e}")
            continue

    df = cudf.concat(li)
    return df.to_pandas()  # ← convert back to pandas


def make_dirs(save_dir, fh):
    if not os.path.exists(f"{save_dir}/"):
        os.makedirs(f"{save_dir}/")


def reformat_df(df):
    """
    Cleans and reformats a weather data DataFrame by:
    - Removing missing values
    - Resetting the index
    - Adjusting longitude values to the range [-180, 180]
    - Converting temperature variables from Kelvin to Celsius
    - Computing wind speed and direction
    - Calculating forecast lead time in hours
    - Setting 'valid_time' as the new index

    Parameters:
    df (pd.DataFrame): Input DataFrame with weather data.

    Returns:
    pd.DataFrame: Reformatted DataFrame with computed fields.
    """
    # Remove missing values
    df = df.dropna()

    # Reset index
    df = df.reset_index()

    # Normalize longitude values to [-180, 180] range
    df["longitude"] = ((df["longitude"] + 180) % 360) - 180

    # Convert temperature from Kelvin to Celsius
    df[["t2m", "d2m"]] = df[["t2m", "d2m"]] - 273.15

    # Compute wind speed and direction
    u10 = units.Quantity(df["u10"].values, "m/s")
    v10 = units.Quantity(df["v10"].values, "m/s")
    df["u_total"] = mpcalc.wind_speed(u10, v10).magnitude
    df["u_dir"] = mpcalc.wind_direction(u10, v10, convention="from").magnitude

    # Calculate lead time in hours
    lead_time_delta = df["valid_time"] - df["time"]
    df["lead time"] = (24.0 * lead_time_delta.dt.days) + divmod(
        lead_time_delta.dt.seconds, 3600
    )[0]

    # Set 'valid_time' as the index
    df = df.set_index("valid_time")

    return df


def interpolate_func_griddata(values, model_lon, model_lat, xnew, ynew):
    if np.mean(values) == np.nan:
        print("SOME VALS ARE NAN")
    vals = interpolate.griddata(
        (model_lon, model_lat), values, (xnew, ynew), method="linear"
    )
    if np.mean(values) == np.nan:
        print("SOME VALS ARE NAN")
    return vals


def datetime_convert(df, col):
    new_vals = []
    for i, _ in enumerate(df[col]):
        seconds = df[col].iloc[1] / 1e9  # Convert nanoseconds to seconds
        dt = datetime.datetime.utcfromtimestamp(seconds)
        new_vals.append(dt)
    df[col] = new_vals
    return df


def interpolate_model_data_to_nysm_locations_groupby(df_model, df_nysm, vars_to_interp):
    """
    Interpolates model grid data to New York State Mesonet (NYSM) site locations using `groupby` operations.

    This function:
    - Averages NYSM site locations by station.
    - Filters model data to a geographic subset covering New York State.
    - Interpolates selected variables from the model grid to NYSM site locations.
    - Returns a DataFrame with interpolated values for each valid time.

    **Note:**
    - This method is **not recommended for the HRRR grid**, as it is computationally slow.

    Parameters:
    df_model (pd.DataFrame): The input model data containing latitude, longitude, time, and weather variables.
    df_nysm (pd.DataFrame): NYSM station data with latitude (`lat`) and longitude (`lon`) information.
    vars_to_interp (list of str): List of variable names to interpolate.

    Returns:
    pd.DataFrame: A DataFrame with interpolated values at NYSM locations.
    """

    # Compute mean latitude & longitude for each NYSM station
    df_nysm = df_nysm.groupby("station").mean()[["lat", "lon"]]
    xnew = df_nysm["lon"]  # NYSM longitude values for interpolation
    ynew = df_nysm["lat"]  # NYSM latitude values for interpolation

    # Set time as index for model data
    df_model = df_model.set_index("time")

    # Extract model data within the geographic bounds of New York
    df_model_ny = df_model[
        (df_model.latitude >= 39.0)
        & (df_model.latitude <= 47.0)
        & (df_model.longitude <= -71.0)
        & (df_model.longitude >= -80.0)
    ]

    # Select a subset of model grid points for interpolation
    model_lon_lat_ny = df_model_ny[
        df_model_ny["valid_time"] == df_model_ny["valid_time"].unique().min()
    ]
    model_lon_ny = model_lon_lat_ny["longitude"].values
    model_lat_ny = model_lon_lat_ny["latitude"].values

    # Ensure model data includes only selected longitudes
    df_model = df_model_ny[df_model_ny["longitude"].isin(model_lon_ny)]
    df_model = df_model.reset_index()

    # Initialize an empty DataFrame for interpolated results
    df = pd.DataFrame()

    # Perform interpolation for each variable in `vars_to_interp`
    for v, var in enumerate(vars_to_interp):
        print(var)
        df[var] = df_model.groupby(["valid_time"])[var].apply(
            interpolate_func_griddata, model_lon_ny, model_lat_ny, xnew, ynew
        )

    # Expand lists into separate rows
    df_explode = df.apply(lambda col: col.explode())

    # Add NYSM site station mapping if latitude is available
    if "latitude" in df_explode.keys():
        print("Adding NYSM site column")
        nysm_sites = df_nysm.reset_index().station.unique()
        model_interp_lats = df_explode.latitude.unique()
        map_dict = {model_interp_lats[i]: nysm_sites[i] for i in range(len(nysm_sites))}
        df_explode["station"] = df_explode["latitude"].map(map_dict)

    # Convert datetime columns
    df_explode = datetime_convert(df_explode, "valid_time")
    df_explode = df_explode.drop(columns=["valid_time"])
    df_explode["valid_time"] = df_explode.index
    df_explode = datetime_convert(df_explode, "time")

    return df_explode


def get_locations_for_ball_tree(df, nysm_1H_obs):
    locations_a = df.reset_index()[["latitude", "longitude"]]
    locations_b = nysm_1H_obs[["lat", "lon"]].dropna().reset_index()
    # ball tree to find nysm site locations
    # locations_a ==> build the tree
    # locations_b ==> query the tree
    # Creates new columns converting coordinate degrees to radians.
    for column in locations_a[["latitude", "longitude"]]:
        rad = np.deg2rad(locations_a[column].values)
        locations_a[f"{column}_rad"] = rad

    for column in locations_b[["lat", "lon"]]:
        rad = np.deg2rad(locations_b[column].values)
        locations_b[f"{column}_rad"] = rad

    return locations_a, locations_b


def haversine(lon1, lat1, lon2, lat2):
    import math

    R = 6371000  # radius of Earth in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    meters = R * c  # output distance in meters
    km = meters / 1000.0  # output distance in kilometers

    km = round(km, 3)
    # print("distance in km: ", km)

    return km


def df_with_nysm_locations(df):
    """
    RAPIDS-accelerated version of the df_with_nysm_locations function.
    """
    with open(
        "/home/aevans/inference_ai2es_forecast_err/MODELS/lookups/hrrr_station_coords.pkl",
        "rb",
    ) as f:
        station_coords = pickle.load(f)

    # Convert the DataFrame to cuDF
    df_gpu = cudf.from_pandas(df)

    # Prepare station info as a cuDF
    station_list = []
    for station, (lat, lon) in station_coords.items():
        station_list.append({"station": station, "latitude": lat, "longitude": lon})
    station_df = cudf.DataFrame(station_list)

    # Merge on latitude and longitude
    df_merged = df_gpu.merge(station_df, on=["latitude", "longitude"], how="inner")

    # Set index
    df_merged = df_merged.set_index(["station", "valid_time"])

    # Convert back to pandas if needed
    return df_merged.to_pandas()


def redefine_precip_intervals(data, prev_fh, model):
    """
    Adjusts total precipitation ('tp') values in HRRR forecast data to represent hourly intervals.

    This function:
    - Shifts the `valid_time` in `prev_fh` forward by 1 hour.
    - Merges the `tp` values from `prev_fh` into `data` based on the aligned `valid_time`.
    - Computes hourly precipitation by subtracting `tp_prev_fh` (previous forecast hour) from `tp`.
    - Ensures no negative precipitation values by clipping at 0.
    - Drops unnecessary columns after merging.

    Parameters:
    data (pd.DataFrame): The main dataset containing HRRR forecast data.
    prev_fh (pd.DataFrame): The previous forecast hour's dataset, including `valid_time` and `tp`.

    Returns:
    pd.DataFrame: The modified dataset with adjusted hourly precipitation values.
    """

    if model == "GFS":
        # Shift `valid_time` forward by 1 hour in `prev_fh`
        prev_fh["valid_time"] = prev_fh["valid_time"] + pd.to_timedelta(3, unit="h")
    else:
        # Shift `valid_time` forward by 1 hour in `prev_fh`
        prev_fh["valid_time"] = prev_fh["valid_time"] + pd.to_timedelta(1, unit="h")

    # Merge `tp` from `prev_fh` into `data` based on `valid_time`
    data = pd.merge(
        data,
        prev_fh,
        on=["valid_time", "station"],
        how="left",
        suffixes=("", "_prev_fh"),
    )

    # Compute hourly precipitation by subtracting `tp_prev_fh` from `tp`
    data["tp"] = data["tp"] - data["tp_prev_fh"]

    # Ensure no negative precipitation values
    data["tp"] = data["tp"].clip(lower=0)

    # Drop unnecessary columns from the merged dataset
    data = data.loc[:, ~data.columns.str.contains("prev_fh")]
    # Set `station` and `valid_time` as the index for the final DataFrame
    data = data.set_index(["station", "valid_time"])

    return data


def mask_out_water(model, df_model):
    df_model = df_model.reset_index()
    # read in respective data
    # these files are hard coded since we only need land surface information that was not extracted from original files
    # within the data cleaning script
    indir = f"/home/aevans/nwp_bias/data/model_data/{model.upper()}/2018/01/"
    if model.upper() == "GFS":
        filename = "gfs_4_20180101_0000_003.grb2"
        ind = 42
        var = "landn"
    elif model.upper() == "NAM":
        filename = "namanl_218_20180101_0000_003.grb2"
        ind = 26
        var = "lsm"
    elif model.upper() == "HRRR":
        filename = "20180101_hrrr.t00z.wrfsfcf00.grib2"
        ind = 34
        var = "lsm"
    ds = cfgrib.open_datasets(f"{indir}{filename}")

    ds_tointerp = ds[ind]  # extract the data array that contains land surface class
    ds_tointerp = ds_tointerp.assign_coords(
        {"longitude": (((ds_tointerp.longitude + 180) % 360) - 180)}
    )
    if model.upper() == "GFS":
        ds_tointerp = ds_tointerp.sortby("longitude")
    df_tointerp = ds_tointerp.to_dataframe(dim_order=None).reset_index()

    # will need to use this dataframe & merge with data (that needs to be interpolated)
    # based on lat/lon values
    df_model_merge = df_model.merge(
        df_tointerp[["latitude", "longitude", var]], on=["latitude", "longitude"]
    )
    df_model_merge = df_model_merge[
        df_model_merge[var] == 1
    ]  # only return grid cells over land

    return df_model_merge


def main(month, year, model, fh, mask_water=True):
    """
    This function loads in the parquet data cleaned from the grib files and interpolates (GFS, NAM) or finds the nearest
    grid neighbor (HRRR) for each specified variable to each NYSM site location across NYS. It also calculates the
    precipitation accumulation over 1-h (HRRR, NAM when forecast hour <= 36) and 3-h (GFS, NAM when forecast hour > 36)
    increments. These data are saved as parquet files.

    The following parameters need to be passed into main():

    month (str) - integer corresponding to calendar month (e.g. '01' is January, '02' is Februrary, etc.)
    year (str) - the year of interest (e.g., '2020')
    model (str) - hrrr, nam, gfs
    init(str) - initilization time for model, '00' or '12' UTC
    mask_water (bool) - true to mask out grid cells over water before interpolation/nearest-neighbor, false to leave all grid cells available for interpolation/nearest-neighbor
    """
    start_time = time.time()
    savedir = f"/home/aevans/nwp_bias/src/machine_learning/data/{model}_data/fh{fh}/"
    model = model.upper()

    if model == "HRRR":
        pres = "mslma"
    else:
        pres = "prmsl"

    nysm_1H_obs = load_nysm_data(year)
    print("Loading NYSM Data")
    gc.collect()
    df_model_ny = read_data_ny(model, month, year, fh)
    gc.collect()
    print("Loading Model Data")

    if model == "HRRR" and fh != "01":
        print("Loading Previous Model Data")
        previous_fh_df = read_data_ny_v2(model, month, year, str(int(fh) - 1).zfill(2))
        gc.collect()

    if model == "NAM" and fh != "001":
        print("Loading Previous Model Data")
        previous_fh_df = read_data_ny_v2(model, month, year, str(int(fh) - 1).zfill(3))
        gc.collect()

    if model == "GFS" and fh != "003":
        print("Loading Previous Model Data")
        previous_fh_df = read_data_ny_v2(model, month, year, str(int(fh) - 3).zfill(3))
        gc.collect()
    if model == "HRRR":
        # drop some info that got carried over from xarray data array
        keep_vars = [
            "valid_time",
            "time",
            "latitude",
            "longitude",
            "t2m",
            "sh2",
            "d2m",
            "r2",
            "u10",
            "v10",
            "tp",
            pres,
            "orog",
            "tcc",
            "asnow",
            "cape",
            "dswrf",
            "dlwrf",
            "gh",
        ]
    else:
        # drop some info that got carried over from xarray data array
        keep_vars = [
            "valid_time",
            "time",
            "latitude",
            "longitude",
            "t2m",
            "sh2",
            "d2m",
            "r2",
            "u10",
            "v10",
            "tp",
            pres,
            "orog",
            "tcc",
            "cape",
            "cin",
            "dswrf",
            "dlwrf",
            "gh",
        ]

    if "x" in df_model_ny.keys():
        df_model_ny = df_model_ny.drop(
            columns=["x", "y"]
        )  # drop x & y if they're columns since reindex will fail with them in original index

    df_model_ny = df_model_ny.reset_index()[keep_vars]
    df_model_ny = reformat_df(df_model_ny)
    gc.collect()

    print("--- reformatting completed ---")
    if mask_water == True:
        print("masking water")
        # before interpolation or nearest neighbor methods, mask out any grid cells over water
        df_model_ny = mask_out_water(model, df_model_ny)
    print("Access Information closest to NYSM")
    if model in ["GFS", "NAM"]:
        print("interpolating variables")
        vars_to_interp = [
            "valid_time",
            "time",
            "latitude",
            "longitude",
            "t2m",
            "sh2",
            "d2m",
            "r2",
            "u10",
            "v10",
            "u_total",
            "u_dir",
            "tp",
            pres,
            "orog",
            "tcc",
            "cape",
            "cin",
            "dswrf",
            "dlwrf",
            "gh",
        ]

        indices_list_ny = get_ball_tree_indices_ny(df_model_ny, nysm_1H_obs)
        gc.collect()

        # nearest neighbor
        if model == "NAM":
            df_model_nysm_sites_nn, interpolate_stations = df_with_nysm_locations(
                df_model_ny, nysm_1H_obs, indices_list_ny
            )
            if fh != "001":
                # nn
                previous_fh_df_nn, interpolate_stations = df_with_nysm_locations(
                    previous_fh_df, nysm_1H_obs, indices_list_ny
                )
                previous_fh_df_nn.reset_index(inplace=True)
                # interpolation
                previous_fh_df = interpolate_model_data_to_nysm_locations_groupby(
                    previous_fh_df,
                    nysm_1H_obs,
                    ["valid_time", "time", "latitude", "longitude", "tp"],
                )
            gc.collect()
            # interpolation
            df_model_nysm_sites_interp = (
                interpolate_model_data_to_nysm_locations_groupby(
                    df_model_ny, nysm_1H_obs, vars_to_interp
                )
            )

        if model == "GFS":
            df_model_nysm_sites_nn, interpolate_stations = df_with_nysm_locations(
                df_model_ny, nysm_3H_obs, indices_list_ny
            )
            if fh != "003":
                previous_fh_df_nn, interpolate_stations = df_with_nysm_locations(
                    df_model_ny, nysm_3H_obs, indices_list_ny
                )
                previous_fh_df_nn.reset_index(inplace=True)
                # interpolation
                previous_fh_df = interpolate_model_data_to_nysm_locations_groupby(
                    previous_fh_df,
                    nysm_3H_obs,
                    ["valid_time", "time", "latitude", "longitude", "tp"],
                )
            gc.collect()

            # interpolation
            df_model_nysm_sites_interp = (
                interpolate_model_data_to_nysm_locations_groupby(
                    df_model_ny, nysm_3H_obs, vars_to_interp
                )
            )

        df_model_nysm_sites_nn["lead time"] = (
            df_model_nysm_sites_nn["lead time"].astype(float).round(0).astype(int)
        )

        df_model_nysm_sites_nn.reset_index(inplace=True)

        # join dataframes
        # Filter out rows where 'station' is not in interpolate_stations
        df_model_nysm_sites_nn = df_model_nysm_sites_nn[
            ~df_model_nysm_sites_nn["station"].isin(interpolate_stations)
        ]
        # Filter out rows where 'station' is in interpolate_stations
        df_model_nysm_sites_interp = df_model_nysm_sites_interp[
            df_model_nysm_sites_interp["station"].isin(interpolate_stations)
        ]

        df_model_nysm_sites = pd.concat(
            [df_model_nysm_sites_interp, df_model_nysm_sites_nn], axis=0
        )
        df_model_nysm_sites.set_index("time", inplace=True)
        gc.collect()

        if (model == "NAM" and fh != "001") or (model == "GFS" and fh != "003"):
            # join dataframes
            # Filter out rows where 'station' is not in interpolate_stations
            previous_fh_df_nn = previous_fh_df_nn[
                ~previous_fh_df_nn["station"].isin(interpolate_stations)
            ]
            # Filter out rows where 'station' is in interpolate_stations
            previous_fh_df = previous_fh_df[
                previous_fh_df["station"].isin(interpolate_stations)
            ]

            previous_fh_df = pd.concat([previous_fh_df, previous_fh_df_nn], axis=0)
            previous_fh_df.set_index("time", inplace=True)
            gc.collect()

    elif model == "HRRR":
        gc.collect()
        # indices_list_ny = get_ball_tree_indices_ny(df_model_ny, nysm_1H_obs)
        df_model_nysm_sites = df_with_nysm_locations(df_model_ny)
        if fh != "01":
            previous_fh_df = df_with_nysm_locations(previous_fh_df)

        # to avoid future issues, convert lead time to float, round, and then convert to integer
        # without rounding first, the conversion to int will round to the floor, leading to incorrect lead times
        df_model_nysm_sites["lead time"] = (
            df_model_nysm_sites["lead time"].astype(float).round(0).astype(int)
        )
        gc.collect()

    if model == "GFS" and fh != "003":
        gc.collect()
        print("Redefining Precip")
        previous_fh_df.reset_index(inplace=True)
        df_model_nysm_sites = redefine_precip_intervals(
            df_model_nysm_sites, previous_fh_df, model
        )

    if model == "HRRR" and fh != "01":
        gc.collect()
        print("Redefining Precip")
        previous_fh_df.reset_index(inplace=True)
        df_model_nysm_sites = redefine_precip_intervals(
            df_model_nysm_sites, previous_fh_df, model
        )
    if model == "NAM" and fh != "001":
        gc.collect()
        print("Redefining Precip")
        previous_fh_df.reset_index(inplace=True)
        df_model_nysm_sites = redefine_precip_intervals(
            df_model_nysm_sites, previous_fh_df, model
        )

    make_dirs(savedir, fh)
    df_model_nysm_sites = df_model_nysm_sites.fillna(0)
    gc.collect()
    # df_model_nysm_sites.set_index(inplace=True)
    if mask_water:
        df_model_nysm_sites.to_parquet(
            f"{savedir}/{model}_{year}_{month}_direct_compare_to_nysm_sites_mask_water.parquet"
        )
    else:
        df_model_nysm_sites.to_parquet(
            f"{savedir}/{model}_{year}_{month}_direct_compare_to_nysm_sites.parquet"
        )
    timer9 = time.time() - start_time
    print(f"Saving New Files For :: {model} : {year}--{month}")
    print("--- %s seconds ---" % (timer9))


####   END OF MAIN

if __name__ == "__main__":
    # # One at a time
    model = "hrrr"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--month", type=int, required=True, help="Month-- of year to grab data for"
    )
    parser.add_argument(
        "--year", type=int, required=True, help="Year-- to grab data for"
    )
    parser.add_argument(
        "--fh", type=int, required=True, help="Forecast Hour-- to grab data for"
    )
    args = parser.parse_args()

    main(str(args.month).zfill(2), args.year, model, str(args.fh).zfill(2))
