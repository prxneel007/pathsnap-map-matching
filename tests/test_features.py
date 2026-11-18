"""Synthetic check of candidates, features and route alignment. Run: .venv/bin/python tests/test_features.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import prepare, candidates, add_features, align_route  # noqa: E402

# main road: 4 nodes along the equator ~111 m apart (both directions); parallel road ~100 m north
nodes = pd.DataFrame({"lon": [0, .001, .002, .003, 0, .003], "lat": [0, 0, 0, 0, .0009, .0009]})
arcs = pd.DataFrame({"u": [0, 1, 2, 1, 2, 3, 4], "v": [1, 2, 3, 0, 1, 2, 5]})   # 0-2 east, 3-5 west, 6 parallel
lon = np.linspace(0.0001, 0.0029, 30)
track = pd.DataFrame({"lon": lon, "lat": np.full(30, 0.00002), "t": np.arange(1.0, 31.0)})  # driving east, 2 m off
trip = {"track": track, "nodes": nodes, "arcs": arcs, "route": pd.DataFrame({"arc_id": [0, 1, 2]})}

p = prepare(trip)
c = add_features(p, candidates(p, radius=50, k=10))
assert set(c.arc_id) <= {0, 1, 2, 3, 4, 5}, "parallel road 100 m away must not be a candidate"
east = c[c.arc_id.isin([0, 1, 2])].heading_diff
west = c[c.arc_id.isin([3, 4, 5])].heading_diff
assert east.max() < 5 and west.min() > 175, "heading_diff must separate the two directions"

true_arc, d = align_route(p, trip["route"].arc_id.values)
expected = np.digitize(lon, [0.001, 0.002])          # which eastbound arc each point lies on
assert (true_arc == expected).all(), (true_arc, expected)
assert d.max() < 3
print("ok")
