"""Look up the four NYSM stations nearest to a target station.

Two helpers are provided:

* `get_closest_stations_csv(target_station)`
    Reads a pre-computed lookup parquet keyed by station id (built
    once by `main()` below) and returns the list of closest stations.

* `get_closest_stations(nysm_df, neighbors, target_station, nwp_model)`
    On-the-fly nearest-neighbour search using sklearn's `BallTree`
    with the Haversine metric.  Used to (re)build the lookup parquet.

The model architecture expects exactly 4 stations per cluster: the
target plus its three closest neighbours.
"""

import ast

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


# Path to the pre-computed station-cluster lookup parquet (one row
# per station, with `closest_stations` holding a Python literal of the
# four-station list).
LOOKUP_PARQUET = (
    "/home/aevans/inference_ai2es_forecast_err/MODELS/lookups/"
    "traingulate_nysm.parquet"
)

# Earth's radius (used to convert Haversine radians to kilometres).
EARTH_RADIUS_KM = 6378


def get_closest_stations_csv(target_station):
    """Return the cached list of closest stations for `target_station`.

    Returns an empty list and prints a warning if the station is not
    present in the lookup parquet.
    """
    look_up_df = pd.read_parquet(LOOKUP_PARQUET)
    row = look_up_df[look_up_df["station"] == target_station]
    if row.empty:
        print(f"[WARNING] No match found for station: {target_station}")
        return []

    # The `closest_stations` cell is stored as a string repr of a
    # Python list; literal_eval is safe here because the file is
    # produced by us from a controlled build step.
    byte_str = row.iloc[0]["closest_stations"]
    return ast.literal_eval(byte_str)


def get_closest_stations(nysm_df, neighbors, target_station, nwp_model):
    """Compute the `neighbors` closest NYSM stations to `target_station`.

    Parameters
    ----------
    nysm_df : pandas.DataFrame
        Station metadata with `station`, `lat`, `lon` columns.
    neighbors : int
        Number of neighbours to return (including the target).
    target_station : str
        Station id at the centre of the cluster.
    nwp_model : str
        Reserved for variants that may filter by NWP-specific
        availability; currently unused.

    Returns
    -------
    list[str]
        Station ids ordered from closest to farthest.
    """
    # `LKPL` is excluded because it lacks the data feeds needed for
    # the standard 4-station cluster.
    nysm_df = nysm_df[nysm_df["station"] != "LKPL"]

    lats = nysm_df["lat"].unique()
    lons = nysm_df["lon"].unique()

    locations_a = pd.DataFrame({"lat": lats, "lon": lons})
    for column in locations_a[["lat", "lon"]]:
        locations_a[f"{column}_rad"] = np.deg2rad(locations_a[column].values)

    locations_b = locations_a

    # BallTree with Haversine = great-circle distances on the sphere.
    ball = BallTree(locations_a[["lat_rad", "lon_rad"]].values, metric="haversine")

    distances, indices = ball.query(
        locations_b[["lat_rad", "lon_rad"]].values, k=neighbors
    )
    distances_km = distances * EARTH_RADIUS_KM

    indices_list = [indices[x][0:neighbors] for x in range(len(indices))]
    distances_list = [distances_km[x][0:neighbors] for x in range(len(distances_km))]
    stations = nysm_df["station"].unique()

    station_dict = {}
    for k, _ in enumerate(stations):
        station_dict[stations[k]] = (indices_list[k], distances_list[k])

    vals, _dists = station_dict.get(target_station)
    return [stations[v] for v in vals]


def main():
    """Rebuild the station-cluster lookup parquet from scratch.

    Run this once after the NYSM metadata changes (e.g. a new station
    was deployed or decommissioned).
    """
    climdf = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/nysm.csv")
    climdf = climdf.rename(
        columns={"lat [degrees]": "lat", "lon [degrees]": "lon", "stid": "station"}
    )
    stations = climdf["station"].unique()

    master_ls = []
    for s in stations:
        utilize_ls = get_closest_stations(climdf, 4, s, "HRRR")
        master_ls.append({"station": s, "closest_stations": utilize_ls})

    master_df = pd.DataFrame(master_ls)
    master_df.to_parquet(LOOKUP_PARQUET)
    return master_df


if __name__ == "__main__":
    main()
