"""Set partitioning over a pool of routes (matheuristic post-optimisation).

During the search every route of an accepted solution is stored. At the end one integer program picks the
cheapest set of stored routes that serves every request exactly once and respects the number of vehicles of
each class. This is the route-pool step of Subramanian et al. (2013), Parragh & Schmid (2013),
Gschwind & Drexl (2019), Sartori & Buriol (2020) and Malheiros et al. (2021).

The incumbent's routes are in the pool, so the optimum of the program is never worse than the incumbent.
Solver: HiGHS through scipy.optimize.milp, with a time limit.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, milp


class RoutePool:
    def __init__(self, ev, max_size: int = 40000):
        self.ev = ev
        inst = ev.inst
        keys = {}
        self.vc = []                         # interchangeable vehicles: same class and same compatible requests
        for k in range(ev.K):
            key = (int(ev.classes[k]), inst.compat[:, k].tobytes())
            self.vc.append(keys.setdefault(key, len(keys)))
        self.n_vc = len(keys)
        self.pool = {}                       # (vehicle class, route) -> cost
        self.max_size = max_size
        self.stats = {}

    def add(self, sol):
        pool, vc = self.pool, self.vc
        if len(pool) >= self.max_size:
            return
        for k, r in enumerate(sol.routes):
            if r:
                key = (vc[k], tuple(r))
                if key not in pool:
                    pool[key] = sol.costs[k]

    def solve(self, incumbent, time_limit: float, lex: bool = False):
        """Returns a Solution that is at least as good as `incumbent` (the incumbent itself if nothing better)."""
        from ..solution import Solution
        t0 = time.perf_counter()
        ev, n = self.ev, self.ev.n
        self.stats = {"pool": len(self.pool), "improved": False, "time": 0.0, "status": "skipped",
                      "before_distance": float(incumbent.distance), "before_vehicles": int(incumbent.n_vehicles)}
        if incumbent.bank or time_limit <= 0.05:
            return incumbent
        self.add(incumbent)
        keys = list(self.pool)
        cost = np.array([self.pool[k] for k in keys])
        m = len(keys)
        rows, cols = [], []
        for j, (_, route) in enumerate(keys):
            for v in route:
                if v < n:
                    rows.append(v); cols.append(j)
        A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, m))
        cls = np.array([k[0] for k in keys])
        C = sparse.csr_matrix((np.ones(m), (cls, np.arange(m))), shape=(self.n_vc, m))
        cap = np.bincount(np.asarray(self.vc), minlength=self.n_vc).astype(float)
        c = cost.copy()
        if lex:                              # vehicles first, then distance
            c = c + 10.0 * float(incumbent.distance) + 1.0
        try:
            res = milp(c, constraints=[LinearConstraint(A, 1.0, 1.0), LinearConstraint(C, -np.inf, cap)],
                       integrality=np.ones(m), bounds=Bounds(0.0, 1.0),
                       options={"time_limit": float(time_limit), "disp": False, "mip_rel_gap": 1e-6})
        except Exception as e:               # never let the post-step break a run
            self.stats.update(status="error: " + repr(e)[:80], time=time.perf_counter() - t0)
            return incumbent
        self.stats.update(status=str(res.status), time=time.perf_counter() - t0)
        if res.x is None:
            return incumbent
        sel = np.flatnonzero(res.x > 0.5)
        free = {}
        for k in range(ev.K):
            free.setdefault(self.vc[k], []).append(k)
        routes = [[] for _ in range(ev.K)]
        for j in sel:
            c_, route = keys[j]
            if not free[c_]:
                return incumbent
            routes[free[c_].pop(0)] = list(route)
        new = Solution(ev, routes=routes)
        served = sorted(v for r in routes for v in r if v < n)
        if served != list(range(n)) or any(r and not ev.feasible(r, k) for k, r in enumerate(routes)):
            self.stats["status"] += " (rejected: not a valid solution)"
            return incumbent
        better = ((new.n_vehicles, new.distance) < (incumbent.n_vehicles, incumbent.distance - 1e-9)) if lex \
            else (new.distance < incumbent.distance - 1e-9)
        if better:
            self.stats["improved"] = True
            self.stats["gain_pct"] = 100.0 * (incumbent.distance - new.distance) / max(incumbent.distance, 1e-9)
            self.stats["vehicles_saved"] = incumbent.n_vehicles - new.n_vehicles
            return new
        return incumbent
