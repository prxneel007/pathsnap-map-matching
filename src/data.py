"""Raw PathSnap data loading and basic geo helpers."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "Student folder"
PARTITIONS = ("train", "student_test")
EARTH_R = 6_371_008.8  # mean Earth radius, metres


def list_trips(partition):
    return sorted(p.name for p in (DATA_ROOT / partition).iterdir() if p.is_dir())


def _read(path, names, dtype):
    if not path.exists():
        return None
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


def load_trip(partition, trip_id, root=DATA_ROOT):
    """Return dict of DataFrames: track(lon,lat,t), nodes(lon,lat), arcs(u,v), route(arc_id) or None.

    Node ids and arc ids are the 0-based row indices of .nodes / .arcs.
    Reads <root>/<partition>/<trip_id>/<trip_id>.*; pass partition="" to read straight from <root>/<trip_id>/.
    """
    base = Path(root) / partition / trip_id / trip_id
    return {
        "trip_id": trip_id,
        "partition": partition,
        "track": _read(base.with_suffix(".track"), ["lon", "lat", "t"], float),
        "nodes": _read(base.with_suffix(".nodes"), ["lon", "lat"], float),
        "arcs": _read(base.with_suffix(".arcs"), ["u", "v"], np.int64),
        "route": _read(base.with_suffix(".route"), ["arc_id"], np.int64),
    }


def haversine(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (vectorised)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(a))


def to_local_xy(lon, lat, lon0, lat0):
    """Local equirectangular projection to metres around (lon0, lat0).

    Trips span at most a few tens of km; measured error vs. haversine is < 0.2% (see 01_eda).
    """
    x = np.radians(np.asarray(lon) - lon0) * EARTH_R * np.cos(np.radians(lat0))
    y = np.radians(np.asarray(lat) - lat0) * EARTH_R
    return x, y
