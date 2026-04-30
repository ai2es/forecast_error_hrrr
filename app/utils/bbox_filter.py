"""
app/utils/bbox_filter.py
========================
Spatial filtering utilities for station metadata.

All functions operate on plain pandas DataFrames — no geospatial libraries are
required.  Stations are kept when their ``(lat, lon)`` centroid falls strictly
inside the axis-aligned bounding box defined in ``config.yaml``.

Usage
-----
>>> from app.utils.config_loader import load_config
>>> from app.utils.bbox_filter import filter_stations
>>> import pandas as pd
>>> cfg = load_config()
>>> meta = pd.read_csv("stations.csv")   # must have 'lat' and 'lon' columns
>>> inside = filter_stations(meta, cfg.bbox)
"""

from __future__ import annotations

from typing import Union

import pandas as pd

from app.utils.config_loader import BboxConfig


def filter_stations(
    station_meta: pd.DataFrame,
    bbox: BboxConfig,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """
    Return only the rows of *station_meta* whose location falls inside *bbox*.

    The filter is inclusive on both edges (``>=`` / ``<=``), so stations
    sitting exactly on the bounding-box boundary are retained.

    Parameters
    ----------
    station_meta:
        DataFrame with at least two numeric columns for latitude and longitude.
    bbox:
        :class:`~app.utils.config_loader.BboxConfig` instance (the ``cfg.bbox``
        attribute produced by :func:`~app.utils.config_loader.load_config`).
    lat_col:
        Name of the latitude column.  Defaults to ``"lat"``.
    lon_col:
        Name of the longitude column.  Defaults to ``"lon"``.

    Returns
    -------
    pd.DataFrame
        Filtered copy of *station_meta* with the original index preserved.

    Raises
    ------
    KeyError
        If *lat_col* or *lon_col* is not found in *station_meta*.
    """
    for col in (lat_col, lon_col):
        if col not in station_meta.columns:
            raise KeyError(
                f"Column '{col}' not found in station_meta. "
                f"Available columns: {list(station_meta.columns)}"
            )

    mask = (
        (station_meta[lat_col] >= bbox.lat_min)
        & (station_meta[lat_col] <= bbox.lat_max)
        & (station_meta[lon_col] >= bbox.lon_min)
        & (station_meta[lon_col] <= bbox.lon_max)
    )
    return station_meta[mask].copy()


def bbox_to_string(bbox: BboxConfig) -> str:
    """
    Format a bbox as a compact human-readable string, e.g.
    ``"lat=[40.0, 43.5] lon=[-80.0, -71.5]"``.

    Useful for log messages and plot titles.
    """
    return (
        f"lat=[{bbox.lat_min}, {bbox.lat_max}] "
        f"lon=[{bbox.lon_min}, {bbox.lon_max}]"
    )


def bbox_corners(bbox: BboxConfig) -> list[tuple[float, float]]:
    """
    Return the four corners of the bounding box as ``(lat, lon)`` tuples,
    listed clockwise starting from the south-west corner.

    Useful when drawing the bbox polygon on a Folium map.
    """
    return [
        (bbox.lat_min, bbox.lon_min),  # SW
        (bbox.lat_max, bbox.lon_min),  # NW
        (bbox.lat_max, bbox.lon_max),  # NE
        (bbox.lat_min, bbox.lon_max),  # SE
        (bbox.lat_min, bbox.lon_min),  # back to SW to close polygon
    ]
