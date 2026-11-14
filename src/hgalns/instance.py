"""Unified instance schema (Final Methodology, Section 5.3).

Node indexing convention used everywhere in the code:
    pickups    0 .. n-1        (request r has pickup node r)
    deliveries n .. 2n-1       (request r has delivery node n + r)
    depots     2n .. 2n+D-1
Vehicles k = 0..K-1 start at node start[k] and end at node end[k].
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

INF = 1e18


@dataclass
class Instance:
    name: str
    family: str                 # 'lilim' | 'malheiros' | 'cordeau' | 'generated-*'
    n: int                      # number of requests
    coords: np.ndarray          # (N, 2) float64
    dist: np.ndarray            # (N, N) float64  cost/distance matrix
    time: np.ndarray            # (N, N) float64  travel-time matrix
    tw: np.ndarray              # (N, 2) float64  [earliest, latest] start of service
    service: np.ndarray         # (N,)  float64
    demand: np.ndarray          # (N, m) int64   capacity-check form (pickup >0, delivery <0)
    max_ride: np.ndarray        # (n,)  float64  INF if no ride-time limit
    start: np.ndarray           # (K,)  int64   start node of vehicle k
    end: np.ndarray             # (K,)  int64   end node of vehicle k
    cap: np.ndarray             # (K, m) int64  capacity in capacity-check form
    max_duration: np.ndarray    # (K,)  float64  INF if none
    fixed_cost: np.ndarray      # (K,)  float64
    var_cost: np.ndarray        # (K,)  float64  multiplier on dist
    compat: np.ndarray          # (n, K) bool
    objective: str              # 'lexi' (vehicles, then distance) | 'cost'
    has_ride_time: bool
    # original (raw) resource demand/capacity, kept for the independent checker
    raw_demand: np.ndarray | None = None
    raw_cap: np.ndarray | None = None
    resource_model: str = "single"   # 'single' | 'nested4' (Parragh/Malheiros upgrade chain)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ derived
    @property
    def N(self) -> int:
        return self.coords.shape[0]

    @property
    def K(self) -> int:
        return self.start.shape[0]

    @property
    def m(self) -> int:
        return self.demand.shape[1]

    def pickup(self, r: int) -> int:
        return r

    def delivery(self, r: int) -> int:
        return self.n + r

    def vehicle_classes(self) -> np.ndarray:
        """Class id per vehicle: identical (start, end, cap, duration, costs) => same class."""
        keys = {}
        cls = np.zeros(self.K, dtype=np.int64)
        for k in range(self.K):
            key = (int(self.start[k]), int(self.end[k]), tuple(int(c) for c in self.cap[k]),
                   float(self.max_duration[k]), float(self.fixed_cost[k]), float(self.var_cost[k]))
            if key not in keys:
                keys[key] = len(keys)
            cls[k] = keys[key]
        return cls

    def validate(self) -> None:
        n, N, K, m = self.n, self.N, self.K, self.m
        assert self.dist.shape == (N, N) and self.time.shape == (N, N)
        assert self.tw.shape == (N, 2) and self.service.shape == (N,)
        assert self.demand.shape == (N, m) and self.cap.shape == (K, m)
        assert self.max_ride.shape == (n,) and self.compat.shape == (n, K)
        assert np.all(self.demand[:n] >= 0) and np.all(self.demand[n:2 * n] <= 0)
        assert np.array_equal(self.demand[:n], -self.demand[n:2 * n]), "pickup/delivery loads must mirror"
        assert np.all(self.tw[:, 0] <= self.tw[:, 1] + 1e-9)
        assert np.all((self.start >= 2 * n) & (self.end >= 2 * n))


def euclid(coords: np.ndarray) -> np.ndarray:
    d = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((d ** 2).sum(-1))
