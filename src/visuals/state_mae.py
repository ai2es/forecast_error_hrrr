"""Per-station MAE choropleth for the OKSM mesonet.

Renders a Cartopy / Lambert-Conformal map with each station drawn as
a scatter point sized and coloured by its mean-absolute-error value
(`grouped_df['mae']`).  Climate divisions are shaded behind the
points.

This script expects a pre-computed `grouped` Series (station_id ->
MAE) to exist in the importing namespace; it is intended to be
dropped into a notebook or REPL after such a Series has been built
from the inference-vs-truth comparison parquets produced by
`eval_model_metrics.py`.
"""

import cartopy.crs as crs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

clim_div = [
    "Northeast",
    "West Central",
    "Central",
    "East Central",
    "Southwest",
    "South Central",
    "Southeast",
    "Panhandle",
    "North Central",
    "Nan",
    "Nan1",
    "Nan2",
    "Nan3",
]


def create_xCITE_gif(grouped_df, clim_div=clim_div, logo=None):
    """Render the per-station MAE map.

    Parameters
    ----------
    grouped_df : pandas.DataFrame
        One row per station with columns `station`, `lat`, `lon`,
        `mae`.
    clim_div : list[str]
        Climate-division labels used in the legend.  Defaults to the
        OKSM (Oklahoma) divisions defined at module top.
    logo : optional
        Reserved for future use (currently unused).
    """
    font_size = 22

    # Create plot
    fig = plt.figure(figsize=(25, 16))
    ax = fig.add_subplot(
        1,
        1,
        1,
        projection=crs.LambertConformal(
            central_longitude=-98.0, standard_parallels=(30, 40)
        ),
    )

    # Load shapefile
    shapefile_path = "/home/aevans/nwp_bias/src/machine_learning/notebooks/data/GIS.OFFICIAL_CLIM_DIVISIONS.shp"
    gdf = gpd.read_file(shapefile_path)
    gdf_filtered = pd.concat([gdf.iloc[191:198], gdf.iloc[172:175]])
    gdf_filtered["category"] = np.arange(len(gdf_filtered))

    # Create legend for climate divisions
    division_patches = [
        mpatches.Patch(
            color=plt.cm.tab10(i / len(gdf_filtered)), alpha=0.3, label=clim_div[i]
        )
        for i in range(len(gdf_filtered) - 1)
    ]
    legend1 = ax.legend(
        handles=division_patches,
        loc="lower left",
        title="Climate Divisions",
        title_fontsize=font_size,
        fontsize=font_size,
    )
    ax.add_artist(legend1)

    # Set extent
    ax.set_extent([-103.0, -94.0, 33.0, 37.5], crs=crs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linestyle=":", zorder=1)
    ax.add_feature(cfeature.STATES.with_scale("50m"), linestyle=":", zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale("50m"), zorder=1)
    ax.gridlines(
        crs=crs.PlateCarree(),
        draw_labels=True,
        linewidth=2,
        color="black",
        alpha=0.5,
        linestyle="--",
    )

    # Plot climate divisions
    gdf_filtered.plot(
        ax=ax,
        transform=crs.PlateCarree(),
        column="category",
        cmap="tab10",
        alpha=0.3,
        legend=False,
    )

    # Normalize MAE for visual scaling
    mae = grouped_df["mae"].values
    min_mae, max_mae = mae.min(), mae.max()
    size_scaled = 300 + 1200 * (mae - min_mae) / (
        max_mae - min_mae
    )  # make size dynamic

    # Plot scatter points (size & color = MAE)
    sc = ax.scatter(
        grouped_df["lon"],
        grouped_df["lat"],
        s=size_scaled,
        c=mae,
        cmap="gist_stern",
        edgecolor="black",
        transform=crs.PlateCarree(),
        zorder=10,
        vmin=0,
        vmax=1,
    )

    # Annotate scatter points
    for i, row in grouped_df.iterrows():
        ax.annotate(
            row["station"],
            (row["lon"], row["lat"]),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
            fontsize=15,
            color="black",
            transform=crs.PlateCarree(),
            zorder=20,
        )

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation="vertical", shrink=0.6)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label(r"MAE (mm hr$^{-1}$)", fontsize=font_size)
    # (°C) (m s$^{-1}$)

    # Title and ticks
    plt.title(
        f"OKSM: Precipitation-Error Predictions\nLSTM MAE Averaged Across Forecast-Hours",
        fontsize=font_size,
    )
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)

    plt.tight_layout()
    plt.show()


nysm_df = pd.read_csv("/home/aevans/nwp_bias/src/landtype/data/oksm.csv")
keys = nysm_df["stid"].values
lats = nysm_df["nlat"].values
lons = nysm_df["elon"].values

station_coords = {k: (lat, lon) for k, lat, lon in zip(keys, lats, lons)}


grouped_df = pd.DataFrame({"station": grouped.index, "mae": grouped.values})

# Add lat/lon from your station_coords dictionary
grouped_df["lat_lon"] = grouped_df["station"].map(station_coords)
# grouped_df["elev"] = grouped_df["station"].map(elevations)


grouped_df[["lat", "lon"]] = pd.DataFrame(
    grouped_df["lat_lon"].tolist(), index=grouped_df.index
)

create_xCITE_gif(grouped_df)
