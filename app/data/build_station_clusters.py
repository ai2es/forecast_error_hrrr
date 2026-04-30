"""
app/data/build_station_clusters.py
===================================
Build the nearest-neighbour station-cluster lookup parquet that the training
and inference engines need to construct multi-station input windows.

The lookup parquet has one row per station and a ``closest_stations`` column
that stores a Python string repr of the four-element list
``[target_station, neighbour1, neighbour2, neighbour3]``.  This matches
exactly the format produced by
``src/model_data/get_closest_nysm_stations.main()`` and consumed by
``src/model_data/get_closest_nysm_stations.get_closest_stations_csv()``.

Output path
-----------
    {cfg.output.models_dir}/lookups/triangulate_stations.parquet

Usage
-----
From a notebook::

    from app.data.build_station_clusters import build_clusters
    from app.utils.config_loader import load_config
    cfg = load_config()
    build_clusters(cfg)

Or stand-alone::

    python -m app.data.build_station_clusters
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


EARTH_RADIUS_KM = 6378.137


def _compute_clusters(
    station_meta: pd.DataFrame,
    n_neighbors: int = 4,
    exclude_stations: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    For every station in *station_meta* find its *n_neighbors* nearest
    neighbours (inclusive of the station itself) using the Haversine
    great-circle metric.

    Parameters
    ----------
    station_meta:
        DataFrame with at least ``station``, ``lat``, ``lon`` columns.
    n_neighbors:
        Cluster size including the target station.  Must be ≤ the number of
        stations.  The pipeline expects 4.
    exclude_stations:
        Station ids to exclude from the neighbour search
        (e.g. stations with known data gaps).

    Returns
    -------
    pd.DataFrame
        Columns: ``station``, ``closest_stations``.
        ``closest_stations`` holds the Python string repr of a list of station
        ids, ordered from nearest to farthest.
    """
    if exclude_stations:
        station_meta = station_meta[
            ~station_meta["station"].isin(exclude_stations)
        ].copy()

    station_meta = station_meta.drop_duplicates("station").reset_index(drop=True)

    if len(station_meta) < n_neighbors:
        raise ValueError(
            f"Only {len(station_meta)} stations available but n_neighbors={n_neighbors}. "
            "Either lower n_neighbors or add more stations to the bbox."
        )

    lats = station_meta["lat"].values
    lons = station_meta["lon"].values
    station_ids = station_meta["station"].values

    coords_rad = np.deg2rad(np.column_stack([lats, lons]))
    tree = BallTree(coords_rad, metric="haversine")

    actual_k = min(n_neighbors, len(station_meta))
    _, indices = tree.query(coords_rad, k=actual_k)

    records = []
    for i, idx_row in enumerate(indices):
        cluster = [station_ids[j] for j in idx_row]
        records.append(
            {
                "station": station_ids[i],
                "closest_stations": str(cluster),
            }
        )

    return pd.DataFrame(records)


def build_clusters(
    cfg,
    station_meta: Optional[pd.DataFrame] = None,
    n_neighbors: int = 4,
    exclude_stations: Optional[list[str]] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Compute nearest-neighbour clusters for all stations in the bbox and write
    the lookup parquet.

    Parameters
    ----------
    cfg:
        :class:`~app.utils.config_loader.AppConfig` loaded from ``config.yaml``.
    station_meta:
        Optional pre-loaded DataFrame with ``station``, ``lat``, ``lon`` columns.
        When ``None``, the function reads the CSV written by
        :func:`app.data.fetch_asos.fetch_asos_for_bbox` at
        ``{cfg.output.models_dir}/lookups/station_meta.csv``.
    n_neighbors:
        Cluster size (target + neighbours).  Default is 4 to match the
        original NYSM-based pipeline.
    exclude_stations:
        Station ids to omit from the search (e.g. stations with insufficient
        data coverage).
    show_progress:
        Print status messages.

    Returns
    -------
    pd.DataFrame
        The lookup table written to disk.
    """
    lookups_dir = Path(cfg.output.models_dir) / "lookups"
    lookups_dir.mkdir(parents=True, exist_ok=True)

    if station_meta is None:
        meta_path = lookups_dir / "station_meta.csv"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"station_meta.csv not found at {meta_path}. "
                "Run fetch_asos_for_bbox first, or pass station_meta explicitly."
            )
        station_meta = pd.read_csv(meta_path)
        if show_progress:
            print(f"  Loaded {len(station_meta)} stations from {meta_path}")

    for col in ("station", "lat", "lon"):
        if col not in station_meta.columns:
            raise ValueError(
                f"station_meta must have a '{col}' column. "
                f"Found: {list(station_meta.columns)}"
            )

    if show_progress:
        n_actual = min(n_neighbors, len(station_meta))
        print(
            f"  Computing {n_actual}-nearest-neighbour clusters "
            f"for {len(station_meta)} stations …",
            end="",
            flush=True,
        )

    lookup_df = _compute_clusters(
        station_meta, n_neighbors=n_neighbors, exclude_stations=exclude_stations
    )

    out_path = lookups_dir / "triangulate_stations.parquet"
    lookup_df.to_parquet(out_path, index=False)

    if show_progress:
        print(f" done.  Written to {out_path}")

    return lookup_df


def patch_lookup_path(cfg) -> str:
    """
    Return the absolute path to the triangulate_stations parquet.

    Set this as ``LOOKUP_PARQUET`` in ``src/model_data/get_closest_nysm_stations``
    before calling any training or inference engine, e.g.::

        import src.model_data.get_closest_nysm_stations as gcn
        gcn.LOOKUP_PARQUET = patch_lookup_path(cfg)
    """
    return str(Path(cfg.output.models_dir) / "lookups" / "triangulate_stations.parquet")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from app.utils.config_loader import load_config

    cfg = load_config()
    build_clusters(cfg)
