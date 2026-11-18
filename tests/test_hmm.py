"""Synthetic check of the HMM decoder. Run: .venv/bin/python tests/test_hmm.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import prepare, candidates, add_features  # noqa: E402
from hmm import decode  # noqa: E402

# two-way road of 3 arcs eastbound (0,1,2) and westbound (3,4,5); car drives east
nodes = pd.DataFrame({"lon": [0, .001, .002, .003], "lat": [0, 0, 0, 0]})
arcs = pd.DataFrame({"u": [0, 1, 2, 1, 2, 3], "v": [1, 2, 3, 0, 1, 2]})
lon = np.linspace(0.0001, 0.0029, 30)
trip = {"track": pd.DataFrame({"lon": lon, "lat": np.full(30, 0.00002), "t": np.arange(1.0, 31.0)}), "nodes": nodes, "arcs": arcs}
p = prepare(trip)
c = add_features(p, candidates(p, radius=50, k=10))
truth = np.digitize(lon, [0.001, 0.002])

# a noisy classifier: true arc 0.8, its westbound twin 0.5, others 0.1 - except points 12-16 where true and twin swap
t = truth[c.point_idx.values]
is_true, is_twin = c.arc_id.values == t, c.arc_id.values == t + 3
bad = c.point_idx.between(12, 16).values
prob = np.where(is_true, np.where(bad, 0.5, 0.8), np.where(is_twin, np.where(bad, 0.8, 0.5), 0.1))
top1 = c.assign(prob=prob).sort_values("prob", ascending=False).drop_duplicates("point_idx").sort_values("point_idx").arc_id.values
assert (top1 != truth).sum() == 5, "classifier-only should be wrong on the 5 bad points"

pred, breaks = decode(p, c, prob, beta=5.0)
assert (pred == truth).all(), (pred, truth)   # driving east then west then east is not a consistent path
assert breaks == 0
print("ok")
