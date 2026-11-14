"""Solution state for ALNS: routes per vehicle + request bank (Final Methodology Section 9)."""
from __future__ import annotations

import numpy as np

from .feasibility import Arrays, best_insertion, route_feasible, route_dist, insert_into, BIG


class Evaluator:
    """Instance-level helper holding jit arrays, scratch buffers and vehicle classes."""

    def __init__(self, inst, max_checks: int = 10_000):
        self.inst = inst
        self.A = Arrays(inst)
        self.n = inst.n
        self.K = inst.K
        self.classes = inst.vehicle_classes()
        self.max_checks = max_checks
        self._B = np.empty(4 * inst.n + 8)
        self._pos = np.zeros(inst.N, dtype=np.int64)
        self.n_insert_evals = 0
        # rejection penalty rho (3.2): larger than any possible routing cost
        self.rho = float(inst.dist.max() * (2 * inst.n + 2) * max(1.0, inst.var_cost.max())
                          + inst.fixed_cost.sum() + 1.0)

    # ---- kernels -------------------------------------------------------------------
    def insertion(self, route, k, r):
        A = self.A
        seq = np.asarray(route, dtype=np.int64)
        self.n_insert_evals += 1
        d, i, j = best_insertion(seq, len(route), k, r, A.n, A.tw, A.service, A.time, A.dist,
                                 A.demand, A.cap, A.start, A.end, A.max_duration, A.max_ride,
                                 A.darp, self._B, self._pos, self.max_checks)
        if i < 0:
            return BIG, -1, -1
        return d * A.var_cost[k] + (A.fixed_cost[k] if len(route) == 0 else 0.0), i, j

    def feasible(self, route, k):
        A = self.A
        seq = np.asarray(route, dtype=np.int64)
        return route_feasible(seq, len(route), k, A.n, A.tw, A.service, A.time, A.demand,
                              A.cap, A.start, A.end, A.max_duration, A.max_ride, A.darp,
                              self._B, self._pos)

    def schedule(self, route, k):
        A = self.A
        seq = np.asarray(route, dtype=np.int64)
        B = np.empty(len(route) + 2)
        ok = route_feasible(seq, len(route), k, A.n, A.tw, A.service, A.time, A.demand,
                            A.cap, A.start, A.end, A.max_duration, A.max_ride, A.darp,
                            B, self._pos)
        return ok, B.tolist()

    def route_cost(self, route, k):
        if not route:
            return 0.0
        A = self.A
        seq = np.asarray(route, dtype=np.int64)
        return route_dist(seq, len(route), k, A.start, A.end, A.dist) * A.var_cost[k] + A.fixed_cost[k]


class Solution:
    __slots__ = ("ev", "routes", "costs", "bank", "max_routes")

    def __init__(self, ev: Evaluator, routes=None, bank=None, max_routes=None):
        self.ev = ev
        self.routes = routes if routes is not None else [[] for _ in range(ev.K)]
        self.costs = [ev.route_cost(r, k) for k, r in enumerate(self.routes)]
        self.bank = set(bank) if bank is not None else set()
        self.max_routes = max_routes

    def copy(self):
        s = Solution.__new__(Solution)
        s.ev = self.ev
        s.routes = [list(r) for r in self.routes]
        s.costs = list(self.costs)
        s.bank = set(self.bank)
        s.max_routes = self.max_routes
        return s

    # ---- objective -----------------------------------------------------------------
    @property
    def n_vehicles(self) -> int:
        return sum(1 for r in self.routes if r)

    @property
    def distance(self) -> float:
        return float(sum(self.costs))

    def objective(self) -> float:
        """Search objective: routing cost + rho * |bank| (3.1)-(3.2).
        For lexicographic benchmarks the vehicle count is handled by the two-stage scheme."""
        return self.distance + self.ev.rho * len(self.bank)

    def key(self):
        return tuple(tuple(r) for r in self.routes)

    # ---- modification --------------------------------------------------------------
    def request_vehicle(self):
        """Map request -> vehicle for served requests."""
        n = self.ev.n
        out = {}
        for k, r in enumerate(self.routes):
            for v in r:
                if v < n:
                    out[v] = k
        return out

    def remove(self, reqs):
        n = self.ev.n
        reqs = set(reqs)
        drop = reqs | {n + r for r in reqs}
        touched = []
        for k, route in enumerate(self.routes):
            if any(v in drop for v in route):
                self.routes[k] = [v for v in route if v not in drop]
                touched.append(k)
        for k in touched:
            if self.routes[k] and not self.ev.feasible(self.routes[k], k):
                # extremely rare with Euclidean times: fall back to unloading the route
                for v in self.routes[k]:
                    if v < n:
                        reqs.add(v)
                self.routes[k] = []
            self.costs[k] = self.ev.route_cost(self.routes[k], k)
        self.bank |= reqs
        return touched

    def insert(self, r, k, i, j):
        self.routes[k] = insert_into(self.routes[k], r, self.ev.n, i, j)
        self.costs[k] = self.ev.route_cost(self.routes[k], k)
        self.bank.discard(r)

    def candidate_vehicles(self):
        """Non-empty routes + one empty vehicle per class (respecting max_routes)."""
        used = [k for k, r in enumerate(self.routes) if r]
        cand = list(used)
        if self.max_routes is None or len(used) < self.max_routes:
            seen = set()
            for k, r in enumerate(self.routes):
                if not r:
                    c = int(self.ev.classes[k])
                    if c not in seen:
                        seen.add(c)
                        cand.append(k)
        return cand

    def schedules(self):
        out = []
        for k, r in enumerate(self.routes):
            ok, B = self.ev.schedule(r, k)
            out.append(B)
        return out
