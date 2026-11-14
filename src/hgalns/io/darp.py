"""Readers for dial-a-ride benchmarks.

Malheiros et al. (2021) MD-H-DARP  (github.com/igormalheiros/Instances-MDHDARP)
    line 1:            n_vehicles n_requests
    per vehicle:       route_duration C0 C1 C2 C3
    vertices 0..2n+1:  id x y service max_ride q0 q1 q2 q3 e l
    Vertex 0 / 2n+1 are dummy depots at (0,0) (single-depot HDARP). For the multi-depot
    variant (MDHDARP, used for the published BKS), Malheiros et al. (2021, Sec. 5) place
    4 depots at (-5,-5), (5,5), (-5,5), (5,-5); vehicle k belongs to depot k mod 4.
    Resources (Parragh 2011): 0 staff seat, 1 patient seat, 2 stretcher, 3 wheelchair.
    Upgrade chain: accompanying person may use resource 0, 1 or 2; seated patient 1 or 2;
    stretcher only 2; wheelchair only 3.  Because the admissible sets are nested
    ({2} c {1,2} c {0,1,2}) plus the separate {3}, a load vector is feasible iff
        a2 <= C2,  a1+a2 <= C1+C2,  a0+a1+a2 <= C0+C1+C2,  a3 <= C3.
    We therefore store demand/capacity in this cumulative 'nested4' form, which turns the
    check into an ordinary component-wise capacity test.

Cordeau (2006) DARP a/b instances
    line 1:            K n T Q L        (vehicles, requests, max route duration, capacity, max ride)
    vertices 0..2n+1:  id x y service load e l
"""
from __future__ import annotations

import os
import numpy as np

from ..instance import Instance, euclid, INF

MDHDARP_DEPOTS = np.array([[-5.0, -5.0], [5.0, 5.0], [-5.0, 5.0], [5.0, -5.0]])


def nested4(q: np.ndarray) -> np.ndarray:
    """raw (.., 4) [staff, patient, stretcher, wheelchair] -> cumulative capacity-check form."""
    out = np.zeros_like(q)
    out[..., 0] = q[..., 3]                              # wheelchair
    out[..., 1] = q[..., 2]                              # stretcher
    out[..., 2] = q[..., 1] + q[..., 2]                  # patient seat or stretcher
    out[..., 3] = q[..., 0] + q[..., 1] + q[..., 2]      # any seat/stretcher
    return out


def read_malheiros(path: str, multi_depot: bool = True) -> Instance:
    with open(path) as f:
        rows = [ln.split() for ln in f if ln.strip()]
    K, n = int(rows[0][0]), int(rows[0][1])
    veh = np.array([[float(x) for x in rows[1 + k]] for k in range(K)])
    verts = np.array([[float(x) for x in r] for r in rows[1 + K:1 + K + 2 * n + 2]])
    assert verts.shape[0] == 2 * n + 2, path
    D = 4 if multi_depot else 1
    N = 2 * n + D
    coords = np.zeros((N, 2))
    coords[:2 * n] = verts[1:2 * n + 1, 1:3]
    coords[2 * n:] = MDHDARP_DEPOTS if multi_depot else verts[0, 1:3]
    service = np.zeros(N)
    service[:2 * n] = verts[1:2 * n + 1, 3]
    tw = np.zeros((N, 2))
    tw[:2 * n] = verts[1:2 * n + 1, 9:11]
    tw[2 * n:] = verts[0, 9:11]                          # depot horizon [0, H]
    raw = np.zeros((N, 4), dtype=np.int64)
    raw[:2 * n] = verts[1:2 * n + 1, 5:9].astype(np.int64)
    max_ride = verts[1:n + 1, 4].copy()
    raw_cap = veh[:, 1:5].astype(np.int64)
    depot_of = (np.arange(K) % D) + 2 * n
    dist = euclid(coords)
    inst = Instance(
        name=os.path.splitext(os.path.basename(path))[0], family="malheiros", n=n,
        coords=coords, dist=dist, time=dist.copy(), tw=tw, service=service,
        demand=nested4(raw), max_ride=max_ride, start=depot_of.astype(np.int64),
        end=depot_of.astype(np.int64), cap=nested4(raw_cap), max_duration=veh[:, 0].copy(),
        fixed_cost=np.zeros(K), var_cost=np.ones(K), compat=np.ones((n, K), dtype=bool),
        objective="cost", has_ride_time=True, raw_demand=raw, raw_cap=raw_cap,
        resource_model="nested4", meta={"multi_depot": multi_depot},
    )
    # compatibility: a request is compatible with k iff its load alone fits the vehicle
    inst.compat = np.all(inst.demand[:n, None, :] <= inst.cap[None, :, :], axis=2)
    inst.validate()
    return inst


def read_cordeau(path: str) -> Instance:
    with open(path) as f:
        rows = [ln.split() for ln in f if ln.strip()]
    K, n, T, Q, L = int(rows[0][0]), int(rows[0][1]), float(rows[0][2]), int(rows[0][3]), float(rows[0][4])
    verts = np.array([[float(x) for x in r] for r in rows[1:2 * n + 3]])
    N = 2 * n + 1
    coords = np.zeros((N, 2))
    coords[:2 * n] = verts[1:2 * n + 1, 1:3]
    coords[2 * n] = verts[0, 1:3]
    service = np.zeros(N)
    service[:2 * n] = verts[1:2 * n + 1, 3]
    demand = np.zeros((N, 1), dtype=np.int64)
    demand[:2 * n, 0] = verts[1:2 * n + 1, 4].astype(np.int64)
    tw = np.zeros((N, 2))
    tw[:2 * n] = verts[1:2 * n + 1, 5:7]
    tw[2 * n] = verts[0, 5:7]
    dist = euclid(coords)
    inst = Instance(
        name=os.path.basename(path), family="cordeau", n=n, coords=coords, dist=dist,
        time=dist.copy(), tw=tw, service=service, demand=demand, max_ride=np.full(n, L),
        start=np.full(K, 2 * n, dtype=np.int64), end=np.full(K, 2 * n, dtype=np.int64),
        cap=np.full((K, 1), Q, dtype=np.int64), max_duration=np.full(K, T),
        fixed_cost=np.zeros(K), var_cost=np.ones(K), compat=np.ones((n, K), dtype=bool),
        objective="cost", has_ride_time=True, raw_demand=demand.copy(),
        raw_cap=np.full((K, 1), Q, dtype=np.int64), resource_model="single",
    )
    inst.validate()
    return inst
