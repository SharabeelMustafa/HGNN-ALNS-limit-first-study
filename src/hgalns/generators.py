"""Training-instance generators (Final Methodology Section 5.4).

Generated instances are used ONLY for training/validation/tuning; official benchmark
instances are never used for learning.

gen_lilim   Li & Lim-style PDPTW following Solomon's published class parameters:
            classes C/R/RC x horizon types 1 (short) / 2 (long); capacity 200 (type 1),
            700 (C2) / 1000 (R2, RC2); service 90 (C) or 10 (R, RC); depot at the centre;
            a fraction of nodes without restrictive windows (as in Solomon/Li & Lim).
gen_hdarp   Multi-depot heterogeneous DARP following the Cordeau (2006) geometry and the
            Parragh (2011) group-I user/vehicle protocol used by Malheiros et al. (2021):
            coordinates in [-10,10]^2; 4 depots at (+-5,+-5), vehicle k at depot k mod 4;
            half the requests have a 15-min window at the pickup, half at the delivery;
            horizon H = 60 n / K; service 3; maximum ride time 30; route duration H;
            vehicle types T1 (1,6,0,1) and T2 (2,1,1,1); user types: 83% seated, 11% stretcher,
            6% wheelchair, 50% accompanied.
"""
from __future__ import annotations

import numpy as np

from .instance import Instance, euclid, INF
from .io.darp import nested4, MDHDARP_DEPOTS

LILIM_CLASSES = {
    #        horizon, service, capacity, tw half-width range, depot
    "C1": (1236.0, 90.0, 200, (30.0, 90.0), (40.0, 50.0)),
    "C2": (3390.0, 90.0, 700, (80.0, 320.0), (40.0, 50.0)),
    "R1": (230.0, 10.0, 200, (5.0, 30.0), (35.0, 35.0)),
    "R2": (1000.0, 10.0, 1000, (30.0, 250.0), (35.0, 35.0)),
    "RC1": (240.0, 10.0, 200, (5.0, 30.0), (40.0, 50.0)),
    "RC2": (960.0, 10.0, 1000, (30.0, 250.0), (40.0, 50.0)),
}


def _coords(kind, m, rng):
    if kind == "R":
        return rng.uniform(0, 100, size=(m, 2))
    nc = int(rng.integers(6, 11))
    centres = rng.uniform(10, 90, size=(nc, 2))
    pts = centres[rng.integers(nc, size=m)] + rng.normal(0, 5.0, size=(m, 2))
    pts = np.clip(pts, 0, 100)
    if kind == "RC":
        half = m // 2
        pts[:half] = rng.uniform(0, 100, size=(half, 2))
        rng.shuffle(pts)
    return pts


def gen_lilim(n: int, cls: str, rng, name: str = "") -> Instance:
    H, s, Q, (wlo, whi), dep = LILIM_CLASSES[cls]
    kind = cls.rstrip("12")
    N = 2 * n + 1
    coords = np.zeros((N, 2))
    coords[:2 * n] = _coords(kind, 2 * n, rng)
    coords[2 * n] = dep
    D = euclid(coords)
    tw = np.zeros((N, 2))
    tw[2 * n] = (0.0, H)
    service = np.full(N, s)
    service[2 * n] = 0.0
    demand = np.zeros((N, 1), dtype=np.int64)
    frac_tw = rng.choice([0.25, 0.5, 0.75, 1.0])
    dep_i = 2 * n
    for r in range(n):
        p, d = r, n + r
        q = int(rng.integers(10, 41))
        demand[p, 0], demand[d, 0] = q, -q
        t_p_min = D[dep_i, p]
        latest_p = H - s - D[p, d] - s - D[d, dep_i]
        if latest_p < t_p_min:       # move the pair closer to the depot if it does not fit
            coords[p] = coords[dep_i] + 0.4 * (coords[p] - coords[dep_i])
            coords[d] = coords[dep_i] + 0.4 * (coords[d] - coords[dep_i])
            D = euclid(coords)
            t_p_min = D[dep_i, p]
            latest_p = H - s - D[p, d] - s - D[d, dep_i]
        c_p = rng.uniform(t_p_min, max(t_p_min, latest_p))
        hw = rng.uniform(wlo, whi)
        if rng.random() < frac_tw:
            ep, lp = max(0.0, c_p - hw), min(H, c_p + hw)
        else:
            ep, lp = 0.0, max(t_p_min, latest_p)
        lp = max(lp, t_p_min)
        t_d_min = ep + s + D[p, d]
        latest_d = H - s - D[d, dep_i]
        c_d = rng.uniform(max(t_d_min, c_p + s + D[p, d]), max(t_d_min, latest_d) + 1e-9)
        hw = rng.uniform(wlo, whi)
        if rng.random() < frac_tw:
            ed, ld = max(0.0, c_d - hw), min(latest_d, c_d + hw)
        else:
            ed, ld = 0.0, latest_d
        ld = max(ld, max(t_d_min, ep + s + D[p, d]))
        ed = min(ed, ld)
        tw[p] = (ep, lp)
        tw[d] = (ed, ld)
    K = n   # ample fleet; vehicle count is minimised by the two-stage ALNS
    inst = Instance(
        name=name or f"gen_{cls}_{n}", family="gen-lilim", n=n, coords=coords, dist=D, time=D.copy(),
        tw=tw, service=service, demand=demand, max_ride=np.full(n, INF),
        start=np.full(K, dep_i, dtype=np.int64), end=np.full(K, dep_i, dtype=np.int64),
        cap=np.full((K, 1), Q, dtype=np.int64), max_duration=np.full(K, INF), fixed_cost=np.zeros(K),
        var_cost=np.ones(K), compat=np.ones((n, K), dtype=bool), objective="lexi",
        has_ride_time=False, raw_demand=demand.copy(), raw_cap=np.full((K, 1), Q, dtype=np.int64),
        resource_model="single", meta={"class": cls},
    )
    inst.validate()
    return inst


T1 = np.array([1, 6, 0, 1])
T2 = np.array([2, 1, 1, 1])


def _user(rng):
    u = rng.random()
    acc = 1 if rng.random() < 0.5 else 0
    if u < 0.83:
        return np.array([acc, 1, 0, 0])
    if u < 0.94:
        return np.array([acc, 0, 1, 0])
    return np.array([acc, 0, 0, 1])


def gen_hdarp(n: int, rng, req_per_vehicle: int | None = None, name: str = "") -> Instance:
    rpv = req_per_vehicle or int(rng.choice([8, 10, 12]))
    K = max(2, int(round(n / rpv)))
    H = 60.0 * n / K
    s, L, W = 3.0, 30.0, 15.0
    N = 2 * n + 4
    coords = np.zeros((N, 2))
    coords[:2 * n] = rng.uniform(-10, 10, size=(2 * n, 2))
    coords[2 * n:] = MDHDARP_DEPOTS
    D = euclid(coords)
    tw = np.zeros((N, 2))
    tw[2 * n:] = (0.0, H)
    service = np.full(N, s)
    service[2 * n:] = 0.0
    raw = np.zeros((N, 4), dtype=np.int64)
    far = D[:2 * n, 2 * n:].max(1)          # worst-case depot distance (any depot)
    for r in range(n):
        p, d = r, n + r
        u = _user(rng)
        raw[p], raw[d] = u, -u
        direct = D[p, d]
        tw[p] = (0.0, H)
        tw[d] = (0.0, H)
        if r < n // 2:     # outbound: window at the destination
            lo = far[p] + s + direct
            hi = H - s - far[d] - W
            e = rng.uniform(lo, max(lo, hi))
            tw[d] = (e, e + W)
        else:              # inbound: window at the origin
            lo = far[p]
            hi = H - W - s - direct - s - far[d]
            e = rng.uniform(lo, max(lo, hi))
            tw[p] = (e, e + W)
    types = np.where(rng.random(K) < 0.5, 0, 1)
    raw_cap = np.stack([T1 if t == 0 else T2 for t in types]).astype(np.int64)
    depot_of = (np.arange(K) % 4) + 2 * n
    inst = Instance(
        name=name or f"gen_hdarp_{n}", family="gen-hdarp", n=n, coords=coords, dist=D, time=D.copy(),
        tw=tw, service=service, demand=nested4(raw), max_ride=np.full(n, L),
        start=depot_of.astype(np.int64), end=depot_of.astype(np.int64), cap=nested4(raw_cap),
        max_duration=np.full(K, H), fixed_cost=np.zeros(K), var_cost=np.ones(K),
        compat=np.ones((n, K), dtype=bool), objective="cost", has_ride_time=True, raw_demand=raw,
        raw_cap=raw_cap, resource_model="nested4", meta={"req_per_vehicle": rpv},
    )
    inst.compat = np.all(inst.demand[:n, None, :] <= inst.cap[None, :, :], axis=2)
    inst.validate()
    return inst


def all_requests_servable(inst) -> bool:
    """Every request must be feasible alone for at least one compatible vehicle."""
    from .solution import Evaluator
    ev = Evaluator(inst)
    reps = {}
    for k, c in enumerate(ev.classes):
        reps.setdefault(int(c), k)
    for r in range(inst.n):
        if not any(ev.insertion([], k, r)[1] >= 0 for k in reps.values()):
            return False
    return True


def generate(family: str, n: int, rng, name: str = "", **kw) -> Instance:
    """Draw instances until every request is individually servable."""
    for _ in range(50):
        if family == "lilim":
            cls = kw.get("cls") or str(rng.choice(list(LILIM_CLASSES)))
            inst = gen_lilim(n, cls, rng, name=name)
        elif family == "hdarp":
            inst = gen_hdarp(n, rng, req_per_vehicle=kw.get("req_per_vehicle"), name=name)
        else:
            raise ValueError(family)
        if all_requests_servable(inst):
            return inst
    raise RuntimeError("could not generate a feasible instance")
