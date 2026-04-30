"""Resample raw 5-minute NYSM observations into hourly / 3-hourly parquets.

The raw NYSM netCDFs (5-minute observations) live under
`/home/aevans/nysm/archive/nysm/netcdf/proc/{year}/{month}/`.  This
module:

1. Reads two months of raw 5-minute data (the requested month and
   the one before it).
2. Computes derived columns (`td` dew point, `mslp` mean-sea-level
   pressure) using `metpy`.
3. Resamples each variable to the requested frequency:
   - `precip_total` : per-station 5-min diff -> sum over interval
   - `wspd_sonic`   : mean over interval (saved as `wspd_sonic_mean`)
   - everything else: top-of-the-hour samples
4. Appends the new rows to the existing per-year parquet at
   `/home/aevans/nwp_bias/data/nysm/nysm_{1H,3H}_obs_{year}.parquet`.

Run via the CLI:

    python get_resampled_nysm_data.py --year 2025 --month 4
"""

import argparse
import glob

import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import xarray as xr
from metpy.units import units


def get_raw_nysm_data(year, start_month):
    """Load the raw 5-minute NYSM netCDFs for the previous + current month.

    Returns
    -------
    df_nysm : pandas.DataFrame
        Long-format observations indexed by `(station, time_5M)`.
    nysm_sites : numpy.ndarray
        Unique station ids present in the dataframe.
    """
    # first, find the available months in the year directory
    nysm_path = f"/home/aevans/nysm/archive/nysm/netcdf/proc/{year}/"
    file_dirs = glob.glob(f"{nysm_path}/*")
    file_dirs.sort()
    df_nysm_list = []
    for x in [int(start_month - 1), start_month]:
        try:
            print(x)
            ds_nysm_month = xr.open_mfdataset(f"{nysm_path}{str(x).zfill(2)}/*.nc")
            df_nysm_list.append(ds_nysm_month.to_dataframe())
        except:
            continue
    df_nysm = pd.concat(df_nysm_list)
    temp = units.Quantity(df_nysm["tair"].values, "degC")
    relh = df_nysm["relh"].values / 100.0
    dewpoint = mpcalc.dewpoint_from_relative_humidity(temp, relh)
    df_nysm["td"] = mpcalc.dewpoint_from_relative_humidity(temp, relh).magnitude

    altimeter_value = units.Quantity(df_nysm["pres"].values, "hPa")
    height = units.Quantity(
        df_nysm["elev"].values + 1.5, "m"
    )  # + 1.5 to adjust for barometer height
    df_nysm["mslp"] = mpcalc.altimeter_to_sea_level_pressure(
        altimeter_value, height, temp
    )
    nysm_sites = df_nysm.reset_index()["station"].unique()

    return df_nysm, nysm_sites


def get_resampled_data(df, interval, method):
    """Resample `df` per station at `interval` using aggregation `method`.

    Parameters
    ----------
    df : pandas.DataFrame
        Indexed by `(station, time_5M)`.
    interval : str
        Pandas offset alias (e.g. `"1H"`, `"3H"`).
    method : str
        Aggregation name (`"mean"`, `"sum"`, `"max"`, ...).
    """
    return (
        df.reset_index()
        .set_index("time_5M")
        .groupby("station")
        .resample(interval, label="right")
        .apply(method)
        .rename_axis(index={"time_5M": f"time_{interval}"})
    )


def get_valid_time_data(df, hours_list, interval):
    """Pick the rows whose `time_5M` is on the top of the hour.

    `hours_list` controls which hours are kept (e.g. all 24 for 1H,
    every 3rd for 3H).
    """
    df = df.reset_index()
    # extract hourly observations at top of the hour in provided list
    df_return = df[
        (df["time_5M"].dt.hour.isin(hours_list)) & (df["time_5M"].dt.minute == 0)
    ]
    return df_return.set_index(["station", "time_5M"]).rename_axis(
        index={"time_5M": f"time_{interval}"}
    )


def get_resampled_precip_data(df, interval, method):
    """Resample 5-minute cumulative precipitation into per-interval totals.

    The raw `precip_total` column is monotonically increasing within
    each station; we take the per-station first-difference, then
    aggregate the per-step deltas into an interval sum.  Unrealistic
    spikes (>500 mm / 5 min) are dropped, and small negatives
    (e.g. gauge resets) are clipped to zero.
    """
    precip_diff = df.groupby("station").diff().reset_index()
    # remove unrealistic precipitation values (e.g., > 500 mm / 5 min)
    precip_diff.loc[precip_diff["precip_total"] > 500.0, "precip_total"] = np.nan
    precip_diff.loc[precip_diff["precip_total"] < 0.0, "precip_total"] = 0.0
    return (
        precip_diff.groupby(["station", pd.Grouper(freq=interval, key="time_5M")])[
            "precip_total"
        ]
        .apply(method)
        .rename_axis(index={"time_5M": f"time_{interval}"})
        .reset_index()
        .set_index(["station", f"time_{interval}"])
    )


def get_resampled_wind_data(df, interval, method):
    """Resample 5-minute sonic wind speed into per-interval aggregates.

    Returns a dataframe with column ``wspd_sonic_{method}`` indexed by
    ``(station, time_{interval})``.
    """
    df = df.reset_index()
    wind_resampled = (
        df.groupby(["station", pd.Grouper(freq=interval, key="time_5M")])["wspd_sonic"]
        .apply(method)
        .rename(f"wspd_sonic_{method}")
        .rename_axis(index={"time_5M": f"time_{interval}"})
        .reset_index()
        .set_index(["station", f"time_{interval}"])
    )
    return wind_resampled


def get_nysm_dataframe_for_resampled(df_nysm, freq):
    """Resample every NYSM variable to `freq` and concat into one frame.

    Parameters
    ----------
    df_nysm : pandas.DataFrame
        Raw 5-minute observations.
    freq : {"1H", "3H"}
    """
    nysm_vars = [
        "lat",
        "lon",
        "elev",
        "tair",
        "ta9m",
        "td",
        "relh",
        "srad",
        "pres",
        "mslp",
        "wspd_sonic",
        "wmax_sonic",
        "wdir_sonic",
        "precip_total",
        "snow_depth",
    ]
    if freq == "1H":
        hours_list = np.arange(0, 24)  # every hour
    elif freq == "3H":
        hours_list = np.arange(0, 24, 3)  # every 3 hours

    precip_dfs = []
    wind_dfs = []

    for var in nysm_vars:
        if var == "precip_total":
            precip_dfs.append(get_resampled_precip_data(df_nysm[var], freq, "sum"))
        elif var == "wspd_sonic":
            wind_resampled = get_resampled_wind_data(df_nysm[var], freq, "mean")
            wind_valid_time = get_valid_time_data(df_nysm[var], hours_list, freq)
            # Combine wind data with valid time data
            wind_dfs.append(wind_resampled)
            wind_dfs.append(wind_valid_time)
        else:
            wind_dfs.append(get_valid_time_data(df_nysm[var], hours_list, freq))

    precip_combined = pd.concat(precip_dfs, axis=1)
    wind_combined = pd.concat(wind_dfs, axis=1)

    # Concatenate precip and wind data frames
    nysm_obs = pd.concat([wind_combined, precip_combined], axis=1)

    return nysm_obs


def main(year, start_month):
    """Append two months of resampled NYSM data to the per-year parquets.

    The output parquets at
    ``/home/aevans/nwp_bias/data/nysm/nysm_{1H,3H}_obs_{year}.parquet``
    are read, the freshly resampled rows are appended, and the result
    is written back in place.

    Parameters
    ----------
    year : int
    start_month : int
        Month being refreshed (this and `start_month - 1` are read).
    """
    save_path = "/home/aevans/nwp_bias/data/nysm/"

    print("--- get_raw_nysm_data ---")
    df_nysm, nysm_sites = get_raw_nysm_data(year, start_month)

    # resample the data to 1H and 3H frequencies
    print("--- get_nysm_dataframe_for_resampled ---")
    nysm_1H_obs = get_nysm_dataframe_for_resampled(df_nysm, "1H")
    nysm_3H_obs = get_nysm_dataframe_for_resampled(df_nysm, "3H")

    og_parquet_1H = pd.read_parquet(f"{save_path}nysm_1H_obs_{year}.parquet")
    og_parquet_3H = pd.read_parquet(f"{save_path}nysm_3H_obs_{year}.parquet")

    save_1H = pd.concat([og_parquet_1H, nysm_1H_obs])
    save_3H = pd.concat([og_parquet_3H, nysm_3H_obs])

    save_1H.to_parquet(f"{save_path}nysm_1H_obs_{year}.parquet")
    save_3H.to_parquet(f"{save_path}nysm_3H_obs_{year}.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--month",
        type=int,
        required=True,
        help="Month-- of year to grab data for",
    )
    parser.add_argument(
        "--year", type=int, required=True, help="Year-- to grab data for"
    )
    args = parser.parse_args()
    main(args.year, args.month)
