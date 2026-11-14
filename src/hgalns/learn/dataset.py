"""Destroyed-state imitation dataset (Final Methodology Section 8.1).

For an instance with elite solutions S(1..M) and best S*:
  * destroyed states are produced by applying the ALNS destroy operators (same q distribution)
    to S*;
  * for each removed request r the target is the partial route of S* containing r, or the
    empty-vehicle candidate of its class if that route was emptied (permutation-invariant);
  * co-routing soft labels y_rr' = fraction of elites in which r and r' share a route,
    defined on the graph's request-to-request edges.
"""
from __future__ import annotations

import numpy as np

from ..graph import build_static_graph, route_state, pair_features, _scale
from ..solution import Evaluator, Solution
from ..alns.operators import DESTROY
from ..alns.search import _q_range, ALNSConfig


def candidates(sol, classes):
    """Candidate vehicles: non-empty routes + one empty vehicle per class (first index)."""
    cand = [k for k, r in enumerate(sol.routes) if r]
    seen = set()
    for k, r in enumerate(sol.routes):
        if not r and int(classes[k]) not in seen:
            seen.add(int(classes[k]))
            cand.append(k)
    return cand


def state_tensors(inst, ev, sol, removed, classes, H, capmax, target_vehicle=None):
    """Features of one destroyed state. target_vehicle: dict r -> vehicle in S* (training)."""
    cand = candidates(sol, classes)
    sched = {k: ev.schedule(sol.routes[k], k)[1] for k in cand if sol.routes[k]}
    xi = np.stack([route_state(inst, ev, sol.routes[k], k, H, capmax) for k in cand]).astype(np.float32)
    stops = [list(sol.routes[k]) for k in cand]
    cls = np.array([int(classes[k]) for k in cand], dtype=np.int64)
    R = len(removed)
    chi = np.zeros((R, len(cand), 4), dtype=np.float32)
    mask = np.zeros((R, len(cand)), dtype=bool)
    tgt = np.full(R, -1, dtype=np.int64)
    pos = {k: i for i, k in enumerate(cand)}
    empty_rep = {int(classes[k]): i for i, k in enumerate(cand) if not sol.routes[k]}
    for a, r in enumerate(removed):
        for b, k in enumerate(cand):
            chi[a, b] = pair_features(inst, r, sol.routes[k], k, H, sched.get(k))
            mask[a, b] = bool(inst.compat[r, k])
        if target_vehicle is not None:
            kv = target_vehicle[r]
            if kv in pos and sol.routes[kv]:
                tgt[a] = pos[kv]
            else:
                tgt[a] = empty_rep.get(int(classes[kv]), -1)
            if tgt[a] >= 0:
                mask[a, tgt[a]] = True
    return {"cand": np.array(cand, dtype=np.int64), "cls": cls, "stops": stops, "xi": xi,
            "removed": np.array(removed, dtype=np.int64), "chi": chi, "mask": mask, "target": tgt}


def corouting_labels(inst, graph, elites):
    n = inst.n
    ei, rel = graph["edge_index"], graph["edge_rel"]
    sel = np.flatnonzero((rel >= 2) & (rel <= 5))
    ra, rb = ei[0, sel] % n, ei[1, sel] % n
    keep = ra != rb
    sel, ra, rb = sel[keep], ra[keep], rb[keep]
    y = np.zeros(len(sel), dtype=np.float32)
    for e in elites:
        veh = np.full(n, -1)
        for k, route in enumerate(e["routes"]):
            for v in route:
                if v < n:
                    veh[v] = k
        y += ((veh[ra] == veh[rb]) & (veh[ra] >= 0)).astype(np.float32)
    y /= max(1, len(elites))
    return sel.astype(np.int64), y


def build_samples(record, n_states=30, seed=0, knn=16):
    inst, elites = record["instance"], record["elites"]
    rng = np.random.default_rng(seed)
    ev = Evaluator(inst)
    graph = build_static_graph(inst, knn=knn)
    classes = graph["classes"]
    H, _, _, capmax, _ = _scale(inst)
    best = elites[0]
    target_vehicle = {}
    for k, route in enumerate(best["routes"]):
        for v in route:
            if v < inst.n:
                target_vehicle[v] = k
    cfg = ALNSConfig()
    lo, hi = _q_range(inst.n, cfg)
    ops = ["random", "shaw", "worst", "string", "route"]
    states = []
    for t in range(n_states):
        sol = Solution(ev, routes=[list(r) for r in best["routes"]])
        q = int(rng.integers(lo, hi + 1))
        op = ops[t % len(ops)]
        removed = DESTROY[op](sol, q, rng)
        removed = [r for r in removed if r in target_vehicle]
        if not removed:
            continue
        st = state_tensors(inst, ev, sol, removed, classes, H, capmax, target_vehicle)
        st["op"] = op
        states.append(st)
    co_edges, co_y = corouting_labels(inst, graph, elites)
    return {"name": inst.name, "family": inst.family, "graph": graph, "states": states,
            "co_edges": co_edges, "co_y": co_y}
