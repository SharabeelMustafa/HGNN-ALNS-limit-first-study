"""Reader for the Li & Lim PDPTW benchmark (SINTEF format).

File format:
    line 1:  K  Q  speed
    lines :  id  x  y  demand  e  l  service  pickup_idx  delivery_idx
Node 0 is the depot. A pickup has demand > 0, pickup_idx = 0, delivery_idx = its delivery.
A delivery has demand < 0, delivery_idx = 0, pickup_idx = its pickup. Pairs are interleaved.
Distances are Euclidean (double precision); travel time = distance / speed (speed = 1).
"""
from __future__ import annotations

import os
import numpy as np

from ..instance import Instance, euclid, INF


def read_lilim(path: str) -> Instance:
    with open(path) as f:
        rows = [ln.split() for ln in f if ln.strip()]
    K, Q, speed = int(rows[0][0]), int(rows[0][1]), float(rows[0][2])
    if speed <= 0:  # ten SINTEF files carry speed 0; the benchmark convention is time = distance
        speed = 1.0
    nodes = {int(r[0]): r for r in rows[1:]}
    pickups = sorted(i for i, r in nodes.items() if i != 0 and int(r[3]) > 0)
    n = len(pickups)
    N = 2 * n + 1
    coords = np.zeros((N, 2))
    tw = np.zeros((N, 2))
    service = np.zeros(N)
    demand = np.zeros((N, 1), dtype=np.int64)
    file_id = np.zeros(N, dtype=np.int64)
    for r, p in enumerate(pickups):
        d = int(nodes[p][8])
        assert int(nodes[d][7]) == p, f"pair mismatch {p}->{d} in {path}"
        for local, fid in ((r, p), (n + r, d)):
            row = nodes[fid]
            coords[local] = (float(row[1]), float(row[2]))
            demand[local, 0] = int(row[3])
            tw[local] = (float(row[4]), float(row[5]))
            service[local] = float(row[6])
            file_id[local] = fid
    dep = nodes[0]
    coords[2 * n] = (float(dep[1]), float(dep[2]))
    tw[2 * n] = (float(dep[4]), float(dep[5]))
    service[2 * n] = float(dep[6])
    dist = euclid(coords)
    time = dist / speed
    inst = Instance(
        name=os.path.splitext(os.path.basename(path))[0], family="lilim", n=n,
        coords=coords, dist=dist, time=time, tw=tw, service=service, demand=demand,
        max_ride=np.full(n, INF), start=np.full(K, 2 * n, dtype=np.int64),
        end=np.full(K, 2 * n, dtype=np.int64), cap=np.full((K, 1), Q, dtype=np.int64),
        max_duration=np.full(K, INF), fixed_cost=np.zeros(K), var_cost=np.ones(K),
        compat=np.ones((n, K), dtype=bool), objective="lexi", has_ride_time=False,
        raw_demand=demand.copy(), raw_cap=np.full((K, 1), Q, dtype=np.int64),
        resource_model="single", meta={"file_id": file_id, "capacity": Q, "max_vehicles": K},
    )
    inst.validate()
    return inst
