"""PathSnap API (laptop 1).

Run from pathsnap/:
    PATHSNAP_DATA_DIR="/path/to/test_folder" .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000

PATHSNAP_DATA_DIR must contain one folder per trip (<id>/<id>.track, .nodes, .arcs and, for scoring only, .route).
The .route file is read only AFTER prediction, to compute accuracy server-side; it is never returned to the UI.
"""
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import load_trip  # noqa: E402
from metrics import route_continuity  # noqa: E402
from pipeline import load_artifacts, predict_trip, score  # noqa: E402

DATA_DIR = Path(os.environ.get("PATHSNAP_DATA_DIR", ROOT.parent / "Student folder" / "student_test")).resolve()
MAX_UI_POINTS = 3000

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pathsnap")
ART = load_artifacts()
app = FastAPI(title="PathSnap API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])
log.info("model=%s | data dir=%s", ART["model_cfg"]["model"], DATA_DIR)


def trip_ids():
    return sorted(d.name for d in DATA_DIR.iterdir() if d.is_dir() and (d / f"{d.name}.track").exists())


@app.get("/health")
def health():
    return {"status": "ok", "model": ART["model_cfg"]["model"], "trips": len(trip_ids())}


@app.get("/trips")
def trips():
    return {"data_dir": DATA_DIR.name, "trips": trip_ids()}


@app.get("/predict/{trip_id}")
def predict(trip_id: str):
    if trip_id not in trip_ids():          # also blocks path traversal: only listed folder names are accepted
        raise HTTPException(404, f"unknown trip {trip_id}")
    log.info("REQUEST predict trip=%s", trip_id)
    t0 = time.time()
    trip = load_trip("", trip_id, root=DATA_DIR)
    trip_for_model = {k: v for k, v in trip.items() if k != "route"}
    out = predict_trip(trip_for_model, ART)
    t_pred = time.time() - t0

    p, u, v = out["prep"], trip["arcs"].u.values, trip["arcs"].v.values
    nd, tr = trip["nodes"], trip["track"]
    result = {"trip_id": trip_id, "gps_points": p["n"], "candidates": out["n_candidates"], "chain_breaks": out["breaks"],
              "segments": int(p["seg"].max()) + 1, "prediction_seconds": round(t_pred, 3)}
    for name in ("clf_arc", "hmm_arc"):
        pred = out[name]
        idx = np.flatnonzero(pred >= 0)
        cont = route_continuity(_pred_df(trip_id, idx, pred[idx]), _points_df(trip_id, p), {trip_id: (u, v)})
        m = score(p, trip["route"].arc_id.values, pred) if trip["route"] is not None else {}
        result[name.replace("_arc", "")] = {**m, "route_continuity": cont}
    arcs_used = np.unique(out["hmm_arc"][out["hmm_arc"] >= 0])
    step = max(1, len(tr) // MAX_UI_POINTS)
    result["gps"] = tr[["lat", "lon"]].values[::step].round(6).tolist()
    result["predicted_route"] = [[[nd.lat[u[a]], nd.lon[u[a]]], [nd.lat[v[a]], nd.lon[v[a]]]] for a in arcs_used]
    log.info("RESPONSE trip=%s points=%d hmm_acc=%s hmm_road_acc=%s clf_acc=%s time=%.2fs", trip_id, p["n"],
             _fmt(result["hmm"].get("point_accuracy")), _fmt(result["hmm"].get("road_accuracy")),
             _fmt(result["clf"].get("point_accuracy")), t_pred)
    return result


def _fmt(x):
    return "n/a" if x is None else f"{x:.4f}"


def _pred_df(tid, idx, arcs):
    return pd.DataFrame({"trip_id": tid, "point_idx": idx, "arc_id": arcs})


def _points_df(tid, p):
    return pd.DataFrame({"trip_id": tid, "point_idx": np.arange(p["n"]), "seg_id": p["seg"]})
