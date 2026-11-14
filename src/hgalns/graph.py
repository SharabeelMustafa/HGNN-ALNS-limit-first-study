"""Heterogeneous graph construction and features (Final Methodology Section 6).

Global node indexing in the graph (per instance):
    0 .. n-1        P nodes (pickups)          == instance node ids
    n .. 2n-1       D nodes (deliveries)       == instance node ids
    2n .. 2n+C-1    T nodes (vehicle classes)
Relations (directed; message passing goes src -> dst):
    0 pair  P->D     1 pair_rev D->P
    2 succ  P->P     3 succ P->D     4 succ D->P     5 succ D->D
    6 compat T->P    7 compat_rev P->T
All features are normalised per instance (coordinates to [0,1]^2, times by the horizon H,
loads by the largest capacity per resource, distances by the mean arc length).
"""
from __future__ import annotations

import numpy as np

N_REL = 8
REL_NAMES = ["pair", "pair_rev", "succ_pp", "succ_pd", "succ_dp", "succ_dd", "compat", "compat_rev"]
F_STOP = 15      # P/D node feature size
F_CLASS = 12     # T node feature size
F_EDGE = 4
F_XI = 8         # route-state features
F_CHI = 4        # request-route pair features
M_PAD = 4        # resources padded to 4


def _scale(inst):
    n = inst.n
    H = float(inst.tw[2 * n:, 1].max())
    xy = inst.coords
    lo, hi = xy.min(0), xy.max(0)
    span = np.maximum(hi - lo, 1e-9)
    capmax = np.maximum(inst.cap.max(0), 1).astype(float)
    dbar = float(inst.dist[:2 * n, :2 * n].mean()) + 1e-9
    return H, lo, span, capmax, dbar


def _pad_m(a):
    out = np.zeros(a.shape[:-1] + (M_PAD,))
    m = min(a.shape[-1], M_PAD)
    out[..., :m] = a[..., :m]
    return out


def build_static_graph(inst, knn: int = 16):
    n = inst.n
    H, lo, span, capmax, dbar = _scale(inst)
    T, D = inst.time, inst.dist
    tw, s = inst.tw, inst.service
    classes = inst.vehicle_classes()
    C = int(classes.max()) + 1
    rep = np.array([int(np.flatnonzero(classes == c)[0]) for c in range(C)])
    depots = np.unique(np.concatenate([inst.start, inst.end]))
    # ---- stop nodes -----------------------------------------------------------------
    xs = np.zeros((2 * n, F_STOP), dtype=np.float32)
    nodes = np.arange(2 * n)
    xs[:, 0:2] = (inst.coords[:2 * n] - lo) / span
    xs[:, 2] = tw[:2 * n, 0] / H
    xs[:, 3] = np.minimum(tw[:2 * n, 1], H) / H
    xs[:, 4] = (np.minimum(tw[:2 * n, 1], H) - tw[:2 * n, 0]) / H
    xs[:, 5] = s[:2 * n] / H
    xs[:, 6:10] = _pad_m(np.abs(inst.demand[:2 * n]) / capmax)
    partner = np.concatenate([np.arange(n) + n, np.arange(n)])
    xs[:, 10] = T[nodes, partner] / H
    ride = np.minimum(np.concatenate([inst.max_ride, inst.max_ride]), H)
    xs[:, 11] = ride / H
    xs[:, 12] = T[np.ix_(depots, nodes)].min(0) / H
    xs[:, 13] = (nodes < n).astype(np.float32)
    xs[:, 14] = (np.minimum(tw[:2 * n, 1], H) - tw[:2 * n, 0] < 0.999 * H).astype(np.float32)
    # ---- vehicle-class nodes ------------------------------------------------------------
    xc = np.zeros((C, F_CLASS), dtype=np.float32)
    for c in range(C):
        k = rep[c]
        xc[c, 0:4] = _pad_m(inst.cap[k][None, :] / capmax)[0]
        xc[c, 4] = inst.fixed_cost[k] / (dbar * 10)
        xc[c, 5] = inst.var_cost[k]
        xc[c, 6] = tw[inst.start[k], 0] / H
        xc[c, 7] = min(tw[inst.end[k], 1], H) / H
        xc[c, 8] = min(inst.max_duration[k], H) / H
        xc[c, 9:11] = (inst.coords[inst.start[k]] - lo) / span
        xc[c, 11] = np.count_nonzero(classes == c) / inst.K
    # ---- edges --------------------------------------------------------------------------
    src, dst, rel, ef = [], [], [], []

    def add(a, b, r, f):
        src.append(a); dst.append(b); rel.append(r); ef.append(f)

    for r in range(n):
        f = [T[r, n + r] / H, D[r, n + r] / dbar, min(ride[r] / max(T[r, n + r], 1e-6), 10.0) / 10.0, 0.0]
        add(r, n + r, 0, f)
        add(n + r, r, 1, f)
    # reachable kNN successors
    e_i = tw[:2 * n, 0]
    l_j = np.minimum(tw[:2 * n, 1], H)
    reach = (e_i[:, None] + s[:2 * n, None] + T[:2 * n, :2 * n]) <= l_j[None, :] + 1e-9
    np.fill_diagonal(reach, False)
    reach[nodes[n:], nodes[n:] - n] = False          # delivery cannot precede its own pickup
    order = np.argsort(T[:2 * n, :2 * n], axis=1)
    for i in range(2 * n):
        cnt = 0
        for j in order[i]:
            if not reach[i, j]:
                continue
            slack = (l_j[j] - e_i[i] - s[i] - T[i, j]) / H
            wait = max(0.0, tw[j, 0] - l_j[i] - s[i] - T[i, j]) / H
            r_id = 2 + (0 if i < n else 2) + (0 if j < n else 1)
            add(i, j, r_id, [T[i, j] / H, D[i, j] / dbar, slack, wait])
            cnt += 1
            if cnt >= knn:
                break
    # compatibility class <-> pickup
    for c in range(C):
        k = rep[c]
        st, en = inst.start[k], inst.end[k]
        for r in range(n):
            if not inst.compat[r, k]:
                continue
            margin = np.min((inst.cap[k] - inst.demand[r]) / np.maximum(inst.cap[k], 1))
            tour = T[st, r] + s[r] + T[r, n + r] + s[n + r] + T[n + r, en]
            slack = (min(tw[en, 1], H) - tw[st, 0] - tour) / H
            f = [T[st, r] / H, T[n + r, en] / H, float(margin), float(slack)]
            add(2 * n + c, r, 6, f)
            add(r, 2 * n + c, 7, f)
    g = {
        "n": n, "C": C, "x_stop": xs, "x_class": xc,
        "edge_index": np.array([src, dst], dtype=np.int64),
        "edge_rel": np.array(rel, dtype=np.int64),
        "edge_attr": np.array(ef, dtype=np.float32),
        "classes": classes, "class_rep": rep, "H": H, "dbar": dbar,
    }
    return g


def route_state(inst, ev, route, k, H=None, capmax=None):
    """xi_k (Section 6.3): load utilisation per resource (4), duration fraction, #stops/2n,
    min forward slack proxy, route cost / (dbar * #stops)."""
    n = inst.n
    if H is None:
        H, _, _, capmax, _ = _scale(inst)
    xi = np.zeros(F_XI, dtype=np.float32)
    if not route:
        return xi
    load = np.cumsum(inst.demand[route], axis=0)
    xi[0:4] = _pad_m(load.max(0)[None, :] / np.maximum(inst.cap[k], 1))[0]
    ok, B = ev.schedule(route, k)
    xi[4] = (B[-1] - B[0]) / H
    xi[5] = len(route) / (2 * n)
    lat = np.minimum(inst.tw[route, 1], H)
    xi[6] = float(np.min(lat - np.array(B[1:-1]))) / H
    xi[7] = ev.route_cost(route, k) / (max(len(route), 1) * (inst.dist[:2 * n, :2 * n].mean() + 1e-9))
    return xi


def pair_features(inst, r, route, k, H, B=None):
    """chi_rk: min travel time pickup->route stops, delivery->route stops (incl. depot),
    time-window overlap of the request with the route time span, compatibility flag."""
    n = inst.n
    stops = [int(inst.start[k])] + list(route) + [int(inst.end[k])]
    T = inst.time
    f = np.zeros(F_CHI, dtype=np.float32)
    f[0] = T[stops, r].min() / H
    f[1] = T[n + r, stops].min() / H
    if B is not None and len(B) >= 2:
        a, b = B[0], B[-1]
        lo, hi = inst.tw[r, 0], min(inst.tw[n + r, 1], H)
        f[2] = max(0.0, min(b, hi) - max(a, lo)) / H
    else:
        f[2] = 1.0
    f[3] = 1.0 if inst.compat[r, k] else 0.0
    return f
