"""Step 4: HMM map-matching decoded with Viterbi.

Hidden state at GPS point i = one of its candidate arcs (from features.candidates).
Emission    log P(obs_i | arc)       = EMIT_W * log(classifier probability of that candidate)
Transition  log P(arc_b | arc_a)     = -|route_dist(a -> b) - straight_dist(p_i, p_i+1)| / BETA   (Newson & Krumm 2009)
            route_dist follows the directed road graph from the projection on a to the projection on b;
            unreachable within the search limit -> transition impossible (-inf).
If every transition between two points is impossible (disconnected candidates), the chain restarts at the
new point from the best previous state and the event is counted as a break.
"""
import heapq
import math

import numpy as np

BETA = 5.0             # metres; tolerance of route-vs-straight distance mismatch (tuned in 04_hmm.ipynb)
EMIT_W = 1.0           # weight of the classifier emission vs. transitions (tuned in 04_hmm.ipynb)
MIN_PROB = 1e-4        # floor so log(prob) stays finite
BACK_TOL = 15.0        # metres of apparent backwards movement on the same arc tolerated as GPS noise


class RoadGraph:
    """Directed graph with a cached, distance-bounded Dijkstra from any node."""

    def __init__(self, u, v, length):
        order = np.argsort(u, kind="stable")
        n = int(max(u.max(), v.max())) + 1
        self.indptr = np.searchsorted(u[order], np.arange(n + 1)).tolist()
        self.nbr, self.w = v[order].tolist(), length[order].tolist()
        self.cache = {}

    def dists_from(self, s, limit):
        hit = self.cache.get(s)
        if hit is not None and hit[0] >= limit:
            return hit[1]
        dist, heap, ip, nb, w = {s: 0.0}, [(0.0, s)], self.indptr, self.nbr, self.w
        while heap:
            d, x = heapq.heappop(heap)
            if d > dist[x]:
                continue
            for k in range(ip[x], ip[x + 1]):
                y, nd = nb[k], d + w[k]
                if nd <= limit and nd < dist.get(y, math.inf):
                    dist[y] = nd
                    heapq.heappush(heap, (nd, y))
        self.cache[s] = (limit, dist)
        return dist


def decode(p, c, prob, beta=BETA, emit_w=EMIT_W, graph=None):
    """Viterbi over each segment of a prepared trip `p` (features.prepare) with candidate table `c`.

    Returns (pred_arc per GPS point, -1 where the point has no candidates; number of chain breaks).
    """
    g = graph or RoadGraph(p["u"], p["v"], p["arc_len"])
    u, v, L = p["u"], p["v"], p["arc_len"]
    order = np.lexsort((-prob, c.point_idx.values))
    pi, arc, frac = c.point_idx.values[order], c.arc_id.values[order], c.proj_frac.values[order]
    emit = emit_w * np.log(np.maximum(prob[order], MIN_PROB))
    starts = np.flatnonzero(np.r_[True, pi[1:] != pi[:-1]])
    bounds = dict(zip(pi[starts], zip(starts, np.r_[starts[1:], len(pi)])))

    pred = np.full(p["n"], -1)
    breaks = 0
    for s in np.unique(p["seg"]):
        pts = [i for i in np.flatnonzero(p["seg"] == s) if i in bounds]
        if not pts:
            continue
        a0, a1 = bounds[pts[0]]
        delta, backs = emit[a0:a1].copy(), []
        prev_i, prev_rng = pts[0], (a0, a1)
        for i in pts[1:]:
            b0, b1 = bounds[i]
            d_gc = math.hypot(p["gx"][i] - p["gx"][prev_i], p["gy"][i] - p["gy"][prev_i])
            limit = 2 * d_gc + 150.0
            A, B = arc[prev_rng[0]:prev_rng[1]], arc[b0:b1]
            FA, FB = frac[prev_rng[0]:prev_rng[1]], frac[b0:b1]
            T = np.full((len(A), len(B)), -np.inf)
            for ia, a in enumerate(A):
                reach = g.dists_from(int(v[a]), limit)
                for ib, b in enumerate(B):
                    if a == b:
                        dr = (FB[ib] - FA[ia]) * L[a]
                        if dr < -BACK_TOL:
                            continue
                        # abs, not signed: a signed penalty scored worse on train OOF (91.4% vs 92.2%) because ~3% of
                        # reference labels point against the GPS travel direction (see 05_hmm.ipynb, section 5)
                        dr = abs(dr)
                    else:
                        sp = reach.get(int(u[b]))
                        if sp is None:
                            continue
                        dr = (1 - FA[ia]) * L[a] + sp + FB[ib] * L[b]
                    T[ia, ib] = -abs(dr - d_gc) / beta
            score = delta[:, None] + T
            best, arg = score.max(axis=0), score.argmax(axis=0)
            if not np.isfinite(best).any():          # disconnected: restart chain from the best previous state
                breaks += 1
                best, arg = np.full(len(B), delta.max()), np.full(len(B), int(delta.argmax()))
            delta = best + emit[b0:b1]
            backs.append(arg)
            prev_i, prev_rng = i, (b0, b1)
        j = int(np.argmax(delta))
        path = [j]
        for arg in reversed(backs):
            j = int(arg[j]); path.append(j)
        for i, j in zip(pts, reversed(path)):
            pred[i] = arc[bounds[i][0] + j]
    return pred, breaks
