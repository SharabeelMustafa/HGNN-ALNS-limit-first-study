"""Destroy and repair operators (Final Methodology Sections 9.2-9.4).

Destroy operators remove whole requests (pickup + delivery) and put them in the bank.
Repair operators insert banked requests at their cheapest feasible position.
A `guide` object (HGNN guidance, Section 9.4) can restrict/re-rank candidate routes:
    guide.prepare(sol, requests) -> None
    guide.candidates(r, cand_vehicles) -> list of vehicles to evaluate first (top-K)
    guide.bias(r, k) -> additive cost term  (-lambda * dbar * log pi(k|r))
With guide=None the operators are exactly the classical ones.
"""
from __future__ import annotations

import numpy as np

from ..feasibility import BIG


# ======================================================================================
# helpers
def _served(sol):
    n = sol.ev.n
    out = []
    for r in sol.routes:
        out.extend(v for v in r if v < n)
    return out


def _times(sol):
    """Service start time of every visited customer node (current schedule)."""
    T = {}
    for k, route in enumerate(sol.routes):
        if route:
            _, B = sol.ev.schedule(route, k)
            for p, v in enumerate(route):
                T[v] = B[p + 1]
    return T


def _pick(sorted_list, rng, p):
    """Ropke & Pisinger randomised selection: element at floor(y^p * |L|)."""
    y = rng.random()
    return sorted_list[int((y ** p) * len(sorted_list))]


# ======================================================================================
# destroy operators
def random_removal(sol, q, rng, **kw):
    served = _served(sol)
    if not served:
        return []
    q = min(q, len(served))
    rem = list(rng.choice(served, size=q, replace=False))
    sol.remove(rem)
    return rem


def shaw_removal(sol, q, rng, p=6, weights=(9.0, 3.0, 2.0, 5.0), **kw):
    """Ropke & Pisinger relatedness (9.1), vectorised per reference request."""
    inst = sol.ev.inst
    n = inst.n
    served = _served(sol)
    if not served:
        return []
    q = min(q, len(served))
    T = _times(sol)
    D = inst.dist
    phi, chi, psi, omega = weights
    dmax = D.max() + 1e-9
    tmax = inst.tw[:, 1].max() + 1e-9
    dem = inst.demand[:n].astype(float)
    qnorm = np.abs(dem).sum(1).max() + 1e-9
    comp = inst.compat.astype(np.int64)
    ncomp = np.maximum(comp.sum(1), 1)
    srv = np.asarray(served)
    tp = np.array([T[r] for r in served])
    td = np.array([T[n + r] for r in served])
    pos = {r: i for i, r in enumerate(served)}
    alive = np.ones(len(served), dtype=bool)
    first = int(rng.integers(len(served)))
    removed = [served[first]]
    alive[first] = False
    while len(removed) < q and alive.any():
        ref = removed[int(rng.integers(len(removed)))]
        ir = pos[ref]
        idx = np.flatnonzero(alive)
        js = srv[idx]
        inter = comp[js] @ comp[ref]
        mn = np.minimum(ncomp[js], ncomp[ref])
        rel = (phi * (D[ref, js] + D[n + ref, n + js]) / (2 * dmax)
               + chi * (np.abs(tp[ir] - tp[idx]) + np.abs(td[ir] - td[idx])) / (2 * tmax)
               + psi * np.abs(dem[js] - dem[ref]).sum(1) / qnorm
               + omega * (1.0 - inter / mn))
        order = np.argsort(rel, kind="stable")
        pick = idx[order[int((rng.random() ** p) * len(idx))]]
        removed.append(served[pick])
        alive[pick] = False
    sol.remove(removed)
    return removed


def _removal_gain(route, i_p, i_d, start, end, D):
    """Exact distance saved by removing the pickup at index i_p and delivery at i_d (i_p < i_d)."""
    full = [start] + route + [end]
    a, b = i_p + 1, i_d + 1               # positions in full
    if b == a + 1:
        u, x, y, w = full[a - 1], full[a], full[b], full[b + 1]
        return D[u, x] + D[x, y] + D[y, w] - D[u, w]
    g = D[full[a - 1], full[a]] + D[full[a], full[a + 1]] - D[full[a - 1], full[a + 1]]
    g += D[full[b - 1], full[b]] + D[full[b], full[b + 1]] - D[full[b - 1], full[b + 1]]
    return g


def worst_removal(sol, q, rng, p=3, **kw):
    ev = sol.ev
    inst = ev.inst
    n = ev.n
    D = inst.dist
    removed = []
    for _ in range(q):
        gains = []
        for k, route in enumerate(sol.routes):
            if not route:
                continue
            where = {v: i for i, v in enumerate(route)}
            vc = inst.var_cost[k]
            for v in route:
                if v < n:
                    g = _removal_gain(route, where[v], where[n + v], int(inst.start[k]), int(inst.end[k]), D) * vc
                    if len(route) == 2:
                        g += inst.fixed_cost[k]
                    gains.append((g, v))
        if not gains:
            break
        gains.sort(reverse=True)
        v = _pick(gains, rng, p)[1]
        sol.remove([v])
        removed.append(v)
    return removed


def route_removal(sol, q, rng, **kw):
    used = [k for k, r in enumerate(sol.routes) if r]
    if not used:
        return []
    sizes = np.array([len(sol.routes[k]) for k in used], dtype=float)
    w = 1.0 / sizes
    k = used[rng.choice(len(used), p=w / w.sum())]
    rem = [v for v in sol.routes[k] if v < sol.ev.n]
    sol.remove(rem)
    return rem


def string_removal(sol, q, rng, max_len=10, **kw):
    """Pair-string removal adapted from SISR (Christiaens & Vanden Berghe, 2020)."""
    inst = sol.ev.inst
    n = inst.n
    served = _served(sol)
    if not served:
        return []
    seed = served[rng.integers(len(served))]
    loc = sol.request_vehicle()
    # routes ordered by proximity of their stops to the seed pickup
    order = np.argsort(inst.dist[seed, :2 * n])
    removed = set()
    done_routes = set()
    for v in order:
        if len(removed) >= q:
            break
        r = v if v < n else v - n
        if r not in loc or loc[r] in done_routes:
            continue
        k = loc[r]
        route = sol.routes[k]
        L = len(route)
        ls = int(rng.integers(1, min(max_len, L) + 1))
        pos = route.index(v)
        start = max(0, min(pos - int(rng.integers(0, ls)), L - ls))
        for u in route[start:start + ls]:
            removed.add(u if u < n else u - n)
        done_routes.add(k)
    rem = list(removed)
    sol.remove(rem)
    return rem


# ======================================================================================
# repair operators
class _Cache:
    """best insertion of request r into vehicle k: (cost, i, j)."""

    def __init__(self, sol, guide):
        self.sol = sol
        self.guide = guide
        self.c = {}

    def get(self, r, k):
        key = (r, k)
        v = self.c.get(key)
        if v is None:
            v = self.sol.ev.insertion(self.sol.routes[k], k, r)
            self.c[key] = v
        return v

    def invalidate_vehicle(self, k):
        for key in [key for key in self.c if key[1] == k]:
            del self.c[key]


def _options(sol, cache, r, cand, guide, noise_amp, rng):
    """Feasible (adjusted_cost, k, i, j) options for request r, cheapest first.
    With a guide: evaluate top-K candidates; fall back to the rest only if none feasible."""
    if guide is not None:
        first = guide.candidates(r, cand)
        groups = [first, [k for k in cand if k not in set(first)]]
    else:
        groups = [cand]
    out = []
    for g in groups:
        for k in g:
            c, i, j = cache.get(r, k)
            if i >= 0:
                adj = c
                if guide is not None:
                    adj += guide.bias(r, k)
                if noise_amp > 0.0:
                    adj = max(0.0, adj + noise_amp * (2.0 * rng.random() - 1.0))
                out.append((adj, k, i, j))
        if out:
            break
    out.sort()
    return out


def _repair(sol, rng, regret_k, noise, guide):
    reqs = list(sol.bank)
    if not reqs:
        return
    if guide is not None:
        guide.prepare(sol, reqs)
    noise_amp = noise * float(sol.ev.inst.dist.max()) if noise else 0.0
    cache = _Cache(sol, guide)
    pending = set(reqs)
    failed = set()
    while pending:
        cand = sol.candidate_vehicles()
        best = None
        for r in pending:
            opts = _options(sol, cache, r, cand, guide, noise_amp, rng)
            if not opts:
                continue
            if regret_k <= 1:
                score = -opts[0][0]
            else:
                score = 0.0
                for t in range(1, regret_k):
                    score += (opts[t][0] if t < len(opts) else BIG * 1e-3) - opts[0][0]
            key = (score, -opts[0][0])
            if best is None or key > best[0]:
                best = (key, r, opts[0])
        if best is None:
            failed |= pending
            break
        _, r, (_, k, i, j) = best
        was_empty = not sol.routes[k]
        sol.insert(r, k, i, j)
        pending.discard(r)
        cache.invalidate_vehicle(k)
        if guide is not None:
            guide.route_changed(sol, k)
        if was_empty:
            pass  # a new empty-vehicle representative may appear; computed lazily
    # failed requests stay in the bank


def greedy_insertion(sol, rng, noise=0.0, guide=None, **kw):
    _repair(sol, rng, 1, noise, guide)


def regret2_insertion(sol, rng, noise=0.0, guide=None, **kw):
    _repair(sol, rng, 2, noise, guide)


def regret3_insertion(sol, rng, noise=0.0, guide=None, **kw):
    _repair(sol, rng, 3, noise, guide)


DESTROY = {
    "random": random_removal,
    "shaw": shaw_removal,
    "worst": worst_removal,
    "route": route_removal,
    "string": string_removal,
}
REPAIR = {
    "greedy": greedy_insertion,
    "regret2": regret2_insertion,
    "regret3": regret3_insertion,
}
