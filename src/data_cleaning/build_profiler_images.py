"""Build per-hour radiometer "profiler images" used by the Hybrid model.

Pipeline:

1. `get_raw_profiler_data(year, root)`
   Reads all available monthly radiometer netCDFs for a year, applies
   unit conversions / QC, and returns one tidy long-format DataFrame.
2. The main routine then walks each station / hour, pulls the
   `(time, range)` slice, stacks the variables into a single 2-D
   array, and writes it as
   ``{out_root}/{year}/{stid}/{stid}_{year}_{MMDDHH}.npy``.

Those `.npy` files are what
`model_data.prepare_hybrid_data._image_path_for` looks up when
attaching `image_path_{stid}` columns to the Hybrid training /
inference dataframes.

Run via:

    python build_profiler_images.py --year 2025
"""

import argparse
import gc
import glob
import os
from datetime import datetime, time

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import xarray as xr


def get_raw_profiler_data(year, radiometer_data_path):
    """Load and preprocess all monthly radiometer netCDFs for `year`.

    Parameters
    ----------
    year : int
    radiometer_data_path : str
        Root directory containing per-year subdirectories with monthly
        netCDF files.

    Returns
    -------
    df_nysm : pandas.DataFrame
        Tidy long-format radiometer observations.
    nysm_sites : numpy.ndarray
        Unique station ids present in the data.
    """
    # Construct path to the specific year's directory
    radiometer_data_path = f"{radiometer_data_path}/{year}/"

    # Find all subdirectories (months) within the year directory
    file_dirs = glob.glob(f"{radiometer_data_path}/*")
    file_dirs.sort()

    # Extract the available months from the directory names
    avail_months = [int(x.split("/")[-1]) for x in file_dirs]
    print(avail_months)
    # List to hold DataFrames for each month
    df_nysm_list = []

    # Loop through the available months and process each
    for x in range(avail_months[0], avail_months[-1] + 1):
        print("month index: ", x)
        # Open multiple NetCDF files for the current month and convert to DataFrame
        try:
            ds_nysm_month = xr.open_mfdataset(
                f"{radiometer_data_path}{str(x).zfill(2)}/*.nc"
            )
            df_nysm_list.append(ds_nysm_month.to_dataframe())
        except:
            continue

    # Concatenate all the monthly DataFrames into one
    df_nysm = pd.concat(df_nysm_list)
    df_nysm.reset_index(inplace=True)

    # Convert temperature and IR temperature from Kelvin to Celsius
    df_nysm["temperature"] = df_nysm["temperature"] - 273.13
    df_nysm["ir_temperature"] = df_nysm["ir_temperature"] - 273.13
    df_nysm["dewpoint"] = df_nysm["dewpoint"] - 273.13

    # Convert relative humidity to a percentage
    df_nysm["relative_humidity"] = df_nysm["relative_humidity"] / 100

    # Drop columns related to quality control or surface data
    df_nysm = df_nysm.drop(columns=df_nysm.filter(like="_qc").columns)
    df_nysm = df_nysm.drop(columns=df_nysm.filter(like="surface").columns)

    # Drop additional columns that are not needed
    drop_list = ["v", "w", "u", "velocity", "direction", "cnr", "rws"]
    df_nysm = df_nysm.drop(columns=drop_list)

    # Filter data to only include range <= 5000 meters
    df_nysm = df_nysm[df_nysm["range"] <= 5000]

    # Fill missing data with -999 (placeholder for missing data)
    df_nysm.fillna(-999, inplace=True)

    # Convert time column to datetime format
    df_nysm["time"] = pd.to_datetime(df_nysm["time"])

    # Get a list of unique station sites
    nysm_sites = df_nysm["station"].unique()

    return df_nysm, nysm_sites


def get_raw_profiler_data_rapids(year, radiometer_data_path, start_month):
    """GPU-accelerated counterpart to `get_raw_profiler_data`.

    Reads the previous + current month's radiometer netCDFs, performs
    unit conversions on GPU via cuDF, and returns a `(cudf.DataFrame,
    np.ndarray of stations)` tuple.
    """
    radiometer_data_path = f"{radiometer_data_path}/{year}/"
    file_dirs = sorted(glob.glob(f"{radiometer_data_path}/*"))

    avail_months = [int(x.split("/")[-1]) for x in file_dirs]

    df_list = []

    for x in [int(start_month - 1), start_month]:
        try:
            ds = xr.open_mfdataset(
                f"{radiometer_data_path}{str(x).zfill(2)}/*.nc",
                parallel=True,
            )
            pdf = ds.to_dataframe().reset_index()
            df_list.append(pdf)
        except:
            continue

    # CPU concat once
    pdf = pd.concat(df_list, ignore_index=True)

    # 🚀 Move to GPU
    df = cudf.DataFrame.from_pandas(pdf)

    # -------- Vectorized GPU operations -------- #

    df["temperature"] -= 273.13
    df["ir_temperature"] -= 273.13
    df["dewpoint"] -= 273.13
    df["relative_humidity"] /= 100

    # Drop unwanted columns
    df = df.drop(columns=[c for c in df.columns if "_qc" in c or "surface" in c], errors="ignore")

    drop_list = ["v", "w", "u", "velocity", "direction", "cnr", "rws"]
    df = df.drop(columns=drop_list, errors="ignore")

    # Filter range
    df = df[df["range"] <= 5000]

    df = df.fillna(-999)

    # Convert time once on GPU
    df["time"] = cudf.to_datetime(df["time"])

    nysm_sites = df["station"].unique().to_pandas().values

    return df, nysm_sites

def make_images(df):
    """Convert per-(time, range) observations into a 3D image stack.

    Each non-`time`/`range` variable becomes a `(height, width)` slice
    where `height = unique ranges` and `width = unique times`.  All
    slices are stacked along the last axis to produce a
    `(height, width, channels)` array.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format radiometer slice for a single station/hour.

    Returns
    -------
    numpy.ndarray
        Image of shape `(height, width, channels)`.
    """
    skip_list = ["time", "range"]  # Skip these columns when converting
    stacked_list = []

    for c in df.columns:
        if c in skip_list:
            continue
        else:
            print(c)
            # Pivot the data to have 'range' as index and 'time' as columns
            var_pivot = df.pivot(index="range", columns="time", values=c)
            var_array = var_pivot.to_numpy()

            stacked_list.append(var_array)  # Append to list instead of vstack

    # Stack along the third axis to create (height, width, channels)
    stacked_array = np.stack(stacked_list, axis=-1)

    print("Final stacked array shape:", stacked_array.shape)  # (h, w, c)
    return stacked_array

def make_images_rapids(df):
    """
    Converts a cuDF DataFrame into a 3D CuPy array 
    GPU-accelerated counterpart to `make_images`.

    Parameters
    ----------
    df : cudf.DataFrame

    Returns
    -------
    cupy.ndarray
        3D array `(height, width, channels)` on the GPU.
    """

    skip_list = ["time", "range"]
    stacked_list = []

    # Sort once for consistent pivot ordering (important!)
    df = df.sort_values(["range", "time"])

    for c in df.columns:
        if c in skip_list:
            continue

        # GPU pivot
        var_pivot = df.pivot(
            index="range",
            columns="time",
            values=c
        )

        # Convert directly to CuPy (stays on GPU)
        var_array = var_pivot.to_cupy()

        stacked_list.append(var_array)

    # Stack along channel dimension on GPU
    stacked_array = cp.stack(stacked_list, axis=-1)

    return stacked_array


def main(radiometer_data_path):
    """Build the per-station, per-hour `.npy` image cache.

    For every year in the loop, every station, every day, every hour:

    1. Filter the radiometer dataframe to that hour's slice.
    2. Pivot it into a `(height, width, channels)` numpy array via
       `make_images`.
    3. Save to ``{out}/{year}/{stid}/{stid}_{year}_{MMDDHH}.npy``.

    Parameters
    ----------
    radiometer_data_path : str
        Root directory containing per-year radiometer netCDFs.
    """
    save_path = "/home/aevans/nwp_bias/src/machine_learning/data/profiler_images"

    # Loop through the specified years and process data for each
    for yy in np.arange(2025, 2026):
        print("YEAR", yy)
        df_nysm, nysm_sites = get_raw_profiler_data(yy, radiometer_data_path)
        gc.collect()

        # Loop through the unique station sites
        for site in nysm_sites:
            print("compiling data for", site)
            df_filtered = df_nysm[df_nysm["station"] == site]
            df_filtered = df_filtered.drop(columns="station")
            gc.collect()

            # Extract unique days from the 'time' column
            unique_days = df_filtered["time"].dt.date.unique()

            # Loop through each day
            for d in unique_days:
                time_filtered_df = df_filtered[df_filtered["time"].dt.date == d]
                gc.collect()

                # Loop through each hour of the day (0-23)
                for t in np.arange(0, 24):
                    # Create a datetime object for the specific query time
                    query_time = datetime.combine(d, time(hour=int(t)))

                    # Filter the data for the specific hour
                    hr_df = time_filtered_df[
                        time_filtered_df["time"].dt.hour == query_time.hour
                    ]

                    # Generate an image array from the hourly data
                    image = make_images(hr_df)
                    gc.collect()

                    if not hr_df.empty:
                        # Extract the year and formatted date-time string for saving
                        year = hr_df["time"].iloc[0].year

                        print(year)
                        formatted_str = hr_df["time"].iloc[0].strftime("%m%d%H")
                        print(formatted_str)

                        if not os.path.exists(f"{save_path}/{year}/{site}/"):
                            os.makedirs(f"{save_path}/{year}/{site}/")

                        # Save the generated image as a numpy file
                        np.save(
                            f"{save_path}/{year}/{site}/{site}_{year}_{formatted_str}.npy",
                            image,
                        )
                        gc.collect()
                    print("saving data for", site, formatted_str)

def rapids_main(radiometer_data_path, year, month):
    """
    Main function to process raw profiler data for multiple years, filter it by station and time,
    and save the results as numpy image files.
    """

    save_path = "/home/aevans/nwp_bias/src/machine_learning/data/profiler_images"

    # Loop through the specified years and process data for each
    yy = year
    print("YEAR", yy)

    # df_nysm must be cudf.DataFrame
    df_nysm, nysm_sites = get_raw_profiler_data_rapids(yy, radiometer_data_path, month)

    gc.collect()

    # Loop through the unique station sites
    for site in nysm_sites:
        print("compiling data for", site)

        # GPU boolean filtering
        df_filtered = df_nysm[df_nysm["station"] == site]

        df_filtered = df_filtered.drop(columns="station")

        gc.collect()

        # Extract unique days from the 'time' column (move small result to CPU)
        unique_days = (
            df_filtered["time"]
            .dt.date
            .unique()
            .to_pandas()
        )

        # Loop through each day
        for d in unique_days:

            # GPU filtering (dt handled by cuDF)
            time_filtered_df = df_filtered[
                df_filtered["time"].dt.date == d
            ]

            gc.collect()

            # Loop through each hour of the day (0-23)
            for t in np.arange(0, 24):

                # Create datetime exactly as before
                query_time = datetime.combine(d, time(hour=int(t)))

                # GPU hour filtering
                hr_df = time_filtered_df[
                    time_filtered_df["time"].dt.hour == query_time.hour
                ]

                # Generate image (should return CuPy array)
                image_gpu = make_images_rapids(hr_df)

                gc.collect()

                # GPU-safe empty check (avoids Pandas fallback)
                if hr_df.shape[0] != 0:

                    # Extract metadata (single-row transfer only)
                    first_time = hr_df["time"].iloc[0]

                    year = int(first_time.year)
                    print(year)

                    formatted_str = first_time.strftime("%m%d%H")
                    print(formatted_str)

                    out_dir = f"{save_path}/{year}/{site}/"

                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)

                    # Convert to CPU only here
                    image_cpu = cp.asnumpy(image_gpu)

                    np.save(
                        f"{out_dir}/{site}_{year}_{formatted_str}.npy",
                        image_cpu,
                    )

                    gc.collect()

                print("saving data for", site, formatted_str)
### END OF MAIN


# Ensure the script runs only when executed directly, not when imported as a module
if __name__ == "__main__":
    radiometer_data_path = "/home/aevans/nysm/archive/profiler/netcdf/proc-range/"
    main(radiometer_data_path)

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--month",
    #     type=int,
    #     required=True,
    #     help="Month-- of year to grab data for",
    # )
    # parser.add_argument(
    #     "--year", type=int, required=True, help="Year-- to grab data for"
    # )
    # args = parser.parse_args()
    # rapids_main(radiometer_data_path, args.year, args.month)
