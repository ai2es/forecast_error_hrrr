import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from sklearn import preprocessing
from sklearn import utils
import ast

def get_closest_stations_csv(target_station):
    # Load the lookup DataFrame
    look_up_df = pd.read_parquet('/home/aevans/inference_ai2es_forecast_err/MODELS/lookups/traingulate_nysm.parquet')

    # Filter the DataFrame for the target station
    row = look_up_df[look_up_df['station'] == target_station]

    if row.empty:
        print(f"[WARNING] No match found for station: {target_station}")
        return []

    # Decode byte string and parse it into a list
    byte_str = row.iloc[0]['closest_stations']
    closest = ast.literal_eval(byte_str.decode('utf-8'))

    return closest

def get_closest_stations(nysm_df, neighbors, target_station, nwp_model):
    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6378
    nysm_df = nysm_df[nysm_df["station"] != "LKPL"]

    lats = nysm_df["lat"].unique()
    lons = nysm_df["lon"].unique()

    locations_a = pd.DataFrame()
    locations_a["lat"] = lats
    locations_a["lon"] = lons

    for column in locations_a[["lat", "lon"]]:
        rad = np.deg2rad(locations_a[column].values)
        locations_a[f"{column}_rad"] = rad

    locations_b = locations_a

    ball = BallTree(locations_a[["lat_rad", "lon_rad"]].values, metric="haversine")

    # k: The number of neighbors to return from tree
    k = neighbors
    # Executes a query with the second group. This will also return two arrays.
    distances, indices = ball.query(locations_b[["lat_rad", "lon_rad"]].values, k=k)

    # Convert distances from radians to kilometers
    distances_km = distances * EARTH_RADIUS_KM

    # source info to creare a dictionary
    indices_list = [indices[x][0:k] for x in range(len(indices))]
    distances_list = [distances_km[x][0:k] for x in range(len(distances_km))]
    stations = nysm_df["station"].unique()

    # create dictionary
    station_dict = {}
    for k, _ in enumerate(stations):
        station_dict[stations[k]] = (indices_list[k], distances_list[k])

    utilize_ls = []
    vals, dists = station_dict.get(target_station)

    for v, d in zip(vals, dists):
        x = stations[v]
        utilize_ls.append(x)

    return utilize_ls



def main():
    climdf = pd.read_csv('/home/aevans/nwp_bias/src/landtype/data/nysm.csv')
    climdf = climdf.rename(columns={"lat [degrees]": "lat", "lon [degrees]": "lon", "stid": "station"})
    stations = climdf['station'].unique()

    master_ls = []

    for s in stations:
        utilize_ls = get_closest_stations(climdf, 5, s, "HRRR")
        master_ls.append({
            "station": s,
            "closest_stations": utilize_ls
        })

    master_df = pd.DataFrame(master_ls)
    master_df.to_parquet('/home/aevans/inference_ai2es_forecast_err/MODELS/lookups/traingulate_nysm.parquet')
    return master_df  # or print(master_df)


if __name__ == "__main__":
    main()


