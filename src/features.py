"""Step 2: candidate generation, per-point label alignment and candidate features.

Everything except `align_route` / `label_candidates` uses only GPS + road network, so the same
code runs at inference time in the API (the reference route is never needed for prediction).
"""
import numpy as np
import pandas as pd
import shapely

from data import load_trip, to_local_xy

GAP_SPLIT_S = 60.0      # a time gap above this starts a new independent segment
LABEL_MAX_DIST = 50.0   # a point aligned to a route arc farther than this gets no label (-1)
RADIUS = 50.0           # candidate search radius (m); chosen in 02_candidates_features.ipynb
MAX_CANDS = 8         # keep the K nearest candidates per point; chosen in 02_candidates_features.ipynb
SPEED_CAP = 70.0        # m/s, caps GPS speed outliers

FEATURES = ["dist", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands", "heading_diff", "speed",
            "arc_len", "proj_frac", "dist_to_arc_end", "two_way", "out_deg", "in_deg",
            "d_prev", "d_next", "d_prev2", "d_next2", "same_as_prev_nearest", "connects_prev_nearest"]


def prepare(trip):
    """Project a raw trip to local metres and build road geometry + GPS kinematics."""
    tr, nd, ar = trip["track"], trip["nodes"], trip["arcs"]
    lon0, lat0 = tr.lon.mean(), tr.lat.mean()
    gx, gy = to_local_xy(tr.lon.values, tr.lat.values, lon0, lat0)
    nx, ny = to_local_xy(nd.lon.values, nd.lat.values, lon0, lat0)
    u, v = ar.u.values, ar.v.values
    n = len(tr)

    # duplicate (u,v) arcs are geometrically identical -> map each arc to the first one (canonical id)
    canon = ar.groupby(["u", "v"]).cumcount().values == 0
    canon_id = pd.Series(np.arange(len(ar))).groupby([u, v]).transform("first").values
    arc_len = np.hypot(nx[v] - nx[u], ny[v] - ny[u])
    usable = canon & (arc_len > 0)          # zero-length arcs / self-loops carry no geometry

    lines = shapely.linestrings(np.stack([np.c_[nx[u], ny[u]], np.c_[nx[v], ny[v]]], axis=1))
    arc_idx = np.flatnonzero(usable)
    tree = shapely.STRtree(lines[arc_idx])

    # segments split at large gaps; neighbours never cross a segment boundary
    t = tr.t.values
    seg = np.r_[0, np.cumsum(np.diff(t) > GAP_SPLIT_S)]
    i = np.arange(n)
    prev = np.where((i > 0) & (seg == np.roll(seg, 1)), i - 1, i)
    nxt = np.where((i < n - 1) & (seg == np.roll(seg, -1)), i + 1, i)
    prev2, nxt2 = prev[prev], nxt[nxt]
    dt = np.maximum(t[nxt] - t[prev], 1e-9)
    dx, dy = gx[nxt] - gx[prev], gy[nxt] - gy[prev]
    speed = np.where(nxt != prev, np.minimum(np.hypot(dx, dy) / dt, SPEED_CAP), 0.0)
    heading = np.degrees(np.arctan2(dx, dy)) % 360    # compass bearing of travel

    keys = u.astype(np.int64) * len(nd) + v
    two_way = np.isin(v.astype(np.int64) * len(nd) + u, keys)
    out_deg = np.bincount(u, minlength=len(nd))
    in_deg = np.bincount(v, minlength=len(nd))
    return dict(n=n, gx=gx, gy=gy, t=t, seg=seg, prev=prev, nxt=nxt, prev2=prev2, nxt2=nxt2,
                speed=speed, heading=heading, points=shapely.points(gx, gy),
                u=u, v=v, lines=lines, arc_len=arc_len, arc_idx=arc_idx, tree=tree, canon_id=canon_id,
                arc_bearing=np.degrees(np.arctan2(nx[v] - nx[u], ny[v] - ny[u])) % 360,
                two_way=two_way, out_deg=out_deg, in_deg=in_deg)


def candidates(p, radius=RADIUS, k=MAX_CANDS):
    """Arcs within `radius` m of each GPS point, keeping the k nearest. Returns one row per (point, arc)."""
    pi, j = p["tree"].query(p["points"], predicate="dwithin", distance=radius)
    a = p["arc_idx"][j]
    d = shapely.distance(p["points"][pi], p["lines"][a])
    c = pd.DataFrame({"point_idx": pi, "arc_id": a, "dist": d}).sort_values(["point_idx", "dist"], kind="stable")
    c["dist_rank"] = c.groupby("point_idx").cumcount()
    return c[c.dist_rank < k].reset_index(drop=True)


def add_features(p, c):
    """Per-candidate features from GPS + network only (no route information)."""
    pi, a = c.point_idx.values, c.arc_id.values
    g = c.groupby("point_idx").dist
    dmin = g.transform("min").values
    c["dist_minus_min"] = c.dist.values - dmin
    c["dist_ratio"] = c.dist.values / (dmin + 1.0)
    c["n_cands"] = g.transform("size").values
    hd = np.abs((p["heading"][pi] - p["arc_bearing"][a] + 180) % 360 - 180)
    c["heading_diff"] = np.where(p["speed"][pi] > 1.0, hd, 90.0)   # heading is meaningless when ~stationary
    c["speed"] = p["speed"][pi]
    c["arc_len"] = p["arc_len"][a]
    frac = shapely.line_locate_point(p["lines"][a], p["points"][pi], normalized=True)
    c["proj_frac"] = frac
    c["dist_to_arc_end"] = np.minimum(frac, 1 - frac) * p["arc_len"][a]
    c["two_way"] = p["two_way"][a].astype(int)
    c["out_deg"] = p["out_deg"][p["v"][a]]
    c["in_deg"] = p["in_deg"][p["u"][a]]
    # neighbour support: do nearby GPS points also lie close to this arc?
    for name, nb in [("d_prev", "prev"), ("d_next", "nxt"), ("d_prev2", "prev2"), ("d_next2", "nxt2")]:
        c[name] = shapely.distance(p["points"][p[nb][pi]], p["lines"][a])
    nearest = np.full(p["n"], -1)
    r0 = c.dist_rank.values == 0
    nearest[pi[r0]] = a[r0]
    pn = nearest[p["prev"][pi]]
    has_prev = (p["prev"][pi] != pi) & (pn >= 0)
    c["same_as_prev_nearest"] = (has_prev & (pn == a)).astype(int)
    c["connects_prev_nearest"] = (has_prev & ((pn == a) | (p["v"][np.maximum(pn, 0)] == p["u"][a]))).astype(int)
    return c


def align_route(p, route_arc_ids):
    """Order-preserving alignment of GPS points to the reference route arcs (dynamic programming).

    Each point i gets route position j(i) with j non-decreasing in i, minimising the sum of
    point-to-arc distances. Returns (true_arc per point, alignment distance); true_arc = -1 when
    the aligned arc is farther than LABEL_MAX_DIST (unreliable reference / off-route GPS).
    """
    r = p["canon_id"][np.asarray(route_arc_ids)]
    r = r[p["arc_len"][r] > 0]
    r = r[np.r_[True, r[1:] != r[:-1]]]          # collapse consecutive repeats
    R = p["lines"][r]
    m, n = len(r), p["n"]
    back = np.empty((n, m), dtype=np.int16 if m < 32767 else np.int32)
    idx = np.arange(m)
    cost = shapely.distance(p["points"][0], R)
    for i in range(1, n):
        pm = np.minimum.accumulate(cost)
        back[i] = np.maximum.accumulate(np.where(cost == pm, idx, 0))   # argmin of cost[:j+1]
        cost = shapely.distance(p["points"][i], R) + pm
    j = int(np.argmin(cost))
    pos = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        pos[i] = j
        j = back[i][j]
    d = shapely.distance(p["points"], R[pos])
    return np.where(d <= LABEL_MAX_DIST, r[pos], -1), d


def label_candidates(c, true_arc):
    """label = 1 if candidate is the point's true arc, 0 otherwise, -1 if the point has no reliable label."""
    t = true_arc[c.point_idx.values]
    c["label"] = np.where(t < 0, -1, (c.arc_id.values == t).astype(int))
    return c


def build_trip(partition, trip_id, radius=RADIUS, k=MAX_CANDS, with_labels=True):
    """Candidate feature table for one trip, plus a per-point table."""
    trip = load_trip(partition, trip_id)
    p = prepare(trip)
    c = add_features(p, candidates(p, radius, k))
    pts = pd.DataFrame({"point_idx": np.arange(p["n"]), "seg_id": p["seg"], "t": p["t"],
                        "n_cands": np.bincount(c.point_idx, minlength=p["n"])})
    if with_labels:
        true_arc, d = align_route(p, trip["route"].arc_id.values)
        c = label_candidates(c, true_arc)
        pts["true_arc"], pts["align_dist"] = true_arc, d
        hit = c[c.label == 1].point_idx.unique()
        pts["true_in_cands"] = np.isin(pts.point_idx, hit)
    c.insert(0, "trip_id", trip_id); pts.insert(0, "trip_id", trip_id)
    c.insert(0, "partition", partition); pts.insert(0, "partition", partition)
    return c, pts
