"""Evaluation metrics: candidate-level classification + point-level Top-1 + route continuity."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score)

from data import DATA_ROOT, list_trips


def load_arcs(partition):
    """trip_id -> (u, v) arrays, used for route continuity."""
    out = {}
    for tid in list_trips(partition):
        a = pd.read_csv(DATA_ROOT / partition / tid / f"{tid}.arcs", sep=r"\s+", header=None).values
        out[tid] = (a[:, 0], a[:, 1])
    return out


def top1_per_point(c, prob):
    """Highest-scoring candidate for every point that has candidates."""
    d = c[["trip_id", "point_idx", "arc_id", "label"]].assign(prob=prob)
    return d.sort_values("prob", ascending=False, kind="stable").drop_duplicates(["trip_id", "point_idx"])


def route_continuity(pred, points, arcs):
    """Share of consecutive points (same segment, both predicted) whose arcs are equal or connected (prev.v == next.u)."""
    p = points[["trip_id", "point_idx", "seg_id"]].merge(pred[["trip_id", "point_idx", "arc_id"]], how="left")
    ok = tot = 0
    for tid, g in p.groupby("trip_id", sort=False):
        u, v = arcs[tid]
        a, s = g.arc_id.values, g.seg_id.values
        m = (s[1:] == s[:-1]) & ~np.isnan(a[1:]) & ~np.isnan(a[:-1])
        x, y = a[:-1][m].astype(int), a[1:][m].astype(int)
        ok += int(((x == y) | (v[x] == u[y])).sum()); tot += int(m.sum())
    return ok / max(tot, 1)


def evaluate(c, prob, points, arcs, thr=0.5):
    """c: candidate rows (any labels); points: per-point table. Labelled rows / points only are scored."""
    lab = c.label.values >= 0
    y, pr = c.label.values[lab], prob[lab]
    yhat = (pr >= thr).astype(int)
    top = top1_per_point(c, prob)
    n_lab = int((points.true_arc >= 0).sum())
    labelled_top = top.merge(points.loc[points.true_arc >= 0, ["trip_id", "point_idx"]])
    return {
        "accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "macro_f1": f1_score(y, yhat, average="macro"),
        "log_loss": log_loss(y, np.clip(pr, 1e-7, 1 - 1e-7)),
        "top1_acc": (labelled_top.label == 1).sum() / n_lab,   # points with no candidates count as wrong
        "route_continuity": route_continuity(top, points, arcs),
        "confusion_matrix": confusion_matrix(y, yhat).tolist(),
    }
