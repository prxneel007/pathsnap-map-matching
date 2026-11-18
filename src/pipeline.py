"""End-to-end inference for one trip: candidates -> classifier -> HMM/Viterbi. Uses GPS + road network only."""
import json
from pathlib import Path

import joblib
import numpy as np

import hmm
from features import FEATURES, add_features, align_route, candidates, prepare

ART = Path(__file__).resolve().parents[1] / "artifacts"


def load_artifacts(art=ART):
    return {"model": joblib.load(art / "model.joblib"),
            "model_cfg": json.load(open(art / "model_config.json")),
            "hmm_cfg": json.load(open(art / "hmm_config.json"))}


def predict_trip(trip, a):
    """trip: dict from data.load_trip (route is ignored). Returns prepared trip, classifier-only and HMM arc per GPS point (-1 = none)."""
    cfg = a["model_cfg"]
    p = prepare(trip)
    c = add_features(p, candidates(p, cfg["radius_m"], cfg["max_cands"]))
    prob = a["model"].predict_proba(c[FEATURES])[:, 1] if len(c) else np.array([])
    clf = np.full(p["n"], -1)
    if len(c):
        top = c.assign(prob=prob).sort_values("prob", ascending=False, kind="stable").drop_duplicates("point_idx")
        clf[top.point_idx.values] = top.arc_id.values
    pred, breaks = hmm.decode(p, c, prob, beta=a["hmm_cfg"]["beta"], emit_w=a["hmm_cfg"]["emit_w"]) if len(c) else (clf.copy(), 0)
    return {"prep": p, "n_candidates": len(c), "clf_arc": clf, "hmm_arc": pred, "breaks": breaks}


def score(p, route_arc_ids, pred):
    """Server-side evaluation against the reference route (never fed back into prediction)."""
    true, _ = align_route(p, route_arc_ids)
    lab = true >= 0
    ref, s = set(true[lab]), set(pred[pred >= 0])
    x, t = np.maximum(pred[lab], 0), true[lab]
    road = (pred[lab] == t) | ((pred[lab] >= 0) & (p["u"][x] == p["v"][t]) & (p["v"][x] == p["u"][t]))   # same road, either direction
    return {"labelled_points": int(lab.sum()),
            "point_accuracy": float((pred[lab] == true[lab]).mean()) if lab.any() else None,
            "road_accuracy": float(road.mean()) if lab.any() else None,
            "route_arc_f1": 2 * len(s & ref) / max(len(s) + len(ref), 1)}
