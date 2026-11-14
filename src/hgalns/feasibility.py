"""Route feasibility and insertion evaluation (Final Methodology Section 9.7).

All hot loops are Numba-compiled. A route is stored as an int64 array of *customer* nodes;
the full sequence is  start[k], seq[0..L-1], end[k].

PDPTW mode (no ride-time limits): earliest-start forward simulation (time windows,
capacity, depot horizon, route duration).
DARP mode: the Cordeau & Laporte (2003, Sec. 4.7) eight-step scheduling procedure, which
delays the depot departure and the service start at origins by the forward time slack
(their eq. 5) to minimise route duration and ride times without creating violations.
"""
from __future__ import annotations

import numpy as np
from numba import njit

BIG = 1e18
EPS = 1e-6


class Arrays:
    """Bundle of read-only instance arrays passed to the jitted kernels."""

    def __init__(self, inst):
        n, N = inst.n, inst.N
        self.n = n
        self.tw = np.ascontiguousarray(inst.tw, dtype=np.float64)
        self.service = np.ascontiguousarray(inst.service, dtype=np.float64)
        self.time = np.ascontiguousarray(inst.time, dtype=np.float64)
        self.dist = np.ascontiguousarray(inst.dist, dtype=np.float64)
        self.demand = np.ascontiguousarray(inst.demand, dtype=np.int64)
        self.cap = np.ascontiguousarray(inst.cap, dtype=np.int64)
        self.start = np.ascontiguousarray(inst.start, dtype=np.int64)
        self.end = np.ascontiguousarray(inst.end, dtype=np.int64)
        self.max_duration = np.ascontiguousarray(np.minimum(inst.max_duration, BIG), dtype=np.float64)
        self.var_cost = np.ascontiguousarray(inst.var_cost, dtype=np.float64)
        self.fixed_cost = np.ascontiguousarray(inst.fixed_cost, dtype=np.float64)
        ride = np.full(N, BIG)
        ride[:n] = np.minimum(inst.max_ride, BIG)
        self.max_ride = ride                       # indexed by pickup node
        self.darp = bool(inst.has_ride_time)

    def args(self):
        return (self.n, self.tw, self.service, self.time, self.demand, self.cap, self.start,
                self.end, self.max_duration, self.max_ride, self.darp)


# --------------------------------------------------------------------------------------
@njit(cache=True)
def _fill_full(seq, L, s, e, full):
    full[0] = s
    for p in range(L):
        full[p + 1] = seq[p]
    full[L + 1] = e


@njit(cache=True)
def _forward(full, q, D0, tw, service, time, B):
    """Earliest-start forward pass from departure D0 at position 0. Returns waiting array via B."""
    B[0] = D0
    for p in range(1, q + 1):
        a = B[p - 1] + service[full[p - 1]] + time[full[p - 1], full[p]]
        B[p] = a if a > tw[full[p], 0] else tw[full[p], 0]


@njit(cache=True)
def _schedule(full, q, k, n, tw, service, time, demand, cap, start, end, max_duration,
              max_ride, darp, B, pos):
    """Evaluate the full sequence full[0..q] (q = index of end depot).
    Fills B (service start times). Returns True iff feasible.
    pos is scratch: pos[node] = position in route (only customer nodes are set)."""
    m = demand.shape[1]
    # capacity (independent of times)
    for r in range(m):
        load = 0
        for p in range(1, q):
            load += demand[full[p], r]
            if load > cap[k, r] or load < 0:
                return False
    # step 1-2: earliest schedule
    _forward(full, q, tw[full[0], 0], tw, service, time, B)
    for p in range(1, q + 1):
        if B[p] > tw[full[p], 1] + EPS:
            return False
    if not darp:
        if B[q] - B[0] > max_duration[k] + EPS:
            # delaying the departure can shorten the duration; use forward slack at 0
            pass
        else:
            return True
    for p in range(1, q):
        pos[full[p]] = p
    # step 3-5: delay departure from the depot by min(F0, total waiting)
    W = np.empty(q + 1)
    W[0] = 0.0
    for p in range(1, q + 1):
        a = B[p - 1] + service[full[p - 1]] + time[full[p - 1], full[p]]
        W[p] = B[p] - a
    F0 = BIG
    cumw = 0.0
    for j in range(0, q + 1):
        if j > 0:
            cumw += W[j]
        sl = tw[full[j], 1] - B[j]
        if sl < 0.0:
            sl = 0.0
        v = cumw + sl
        if v < F0:
            F0 = v
    totw = 0.0
    for p in range(1, q):
        totw += W[p]
    delay = F0 if F0 < totw else totw
    _forward(full, q, tw[full[0], 0] + delay, tw, service, time, B)
    # step 6-7: delay service start at each origin to reduce ride times
    if darp:
        for i in range(1, q):
            v = full[i]
            if v >= n:
                continue          # only origins (pickups)
            # waiting times after i
            for p in range(1, q + 1):
                a = B[p - 1] + service[full[p - 1]] + time[full[p - 1], full[p]]
                W[p] = B[p] - a
            Fi = BIG
            cumw = 0.0
            for j in range(i, q + 1):
                if j > i:
                    cumw += W[j]
                sl = tw[full[j], 1] - B[j]
                u = full[j]
                if u >= n and u < 2 * n:
                    pu = u - n
                    pp = pos[pu]
                    if pp < i:     # rider already on board when delaying at i
                        ride = B[j] - (B[pp] + service[pu])
                        rs = max_ride[pu] - ride
                        if rs < sl:
                            sl = rs
                if sl < 0.0:
                    sl = 0.0
                v2 = cumw + sl
                if v2 < Fi:
                    Fi = v2
            wa = 0.0
            for p in range(i + 1, q):
                wa += W[p]
            d = Fi if Fi < wa else wa
            if d > 0.0:
                B[i] += d
                for p in range(i + 1, q + 1):
                    a = B[p - 1] + service[full[p - 1]] + time[full[p - 1], full[p]]
                    B[p] = a if a > tw[full[p], 0] else tw[full[p], 0]
    # step 8: check all constraints
    for p in range(1, q + 1):
        if B[p] > tw[full[p], 1] + EPS:
            return False
    if B[q] - B[0] > max_duration[k] + EPS:
        return False
    if darp:
        for p in range(1, q):
            u = full[p]
            if u < n:
                dpos = pos[u + n]
                ride = B[dpos] - (B[p] + service[u])
                if ride > max_ride[u] + EPS:
                    return False
    return True


@njit(cache=True)
def route_feasible(seq, L, k, n, tw, service, time, demand, cap, start, end, max_duration,
                   max_ride, darp, B, pos):
    full = np.empty(L + 2, dtype=np.int64)
    _fill_full(seq, L, start[k], end[k], full)
    return _schedule(full, L + 1, k, n, tw, service, time, demand, cap, start, end,
                     max_duration, max_ride, darp, B, pos)


@njit(cache=True)
def route_dist(seq, L, k, start, end, dist):
    if L == 0:
        return 0.0
    c = dist[start[k], seq[0]]
    for p in range(L - 1):
        c += dist[seq[p], seq[p + 1]]
    c += dist[seq[L - 1], end[k]]
    return c


@njit(cache=True)
def best_insertion(seq, L, k, r, n, tw, service, time, dist, demand, cap, start, end,
                   max_duration, max_ride, darp, B, pos, max_checks):
    """Cheapest feasible insertion of request r (pickup r, delivery n+r) into route k.
    Returns (delta_distance, i, j) where the pickup goes after position i and the delivery
    after position j of the *new* sequence semantics described below; (BIG,-1,-1) if none.
    Positions refer to the full sequence 0..L+1 (0 = start depot): pickup inserted between
    full[i] and full[i+1]; delivery inserted between full[j] and full[j+1] (j >= i; j == i
    means the delivery immediately follows the pickup)."""
    p_node = r
    d_node = n + r
    m = demand.shape[1]
    q = L + 1
    full = np.empty(L + 2, dtype=np.int64)
    _fill_full(seq, L, start[k], end[k], full)
    # earliest schedule of the current route (for pruning)
    Bc = np.empty(L + 2)
    _forward(full, q, tw[full[0], 0], tw, service, time, Bc)
    # cumulative loads after each position
    load = np.zeros((L + 2, m), dtype=np.int64)
    for p in range(1, q):
        for c in range(m):
            load[p, c] = load[p - 1, c] + demand[full[p], c]
    # quick check: request alone must fit
    for c in range(m):
        if demand[p_node, c] > cap[k, c]:
            return BIG, -1, -1
    ncand = 0
    cd = np.empty((L + 1) * (L + 2) // 2 + 1)
    ci = np.empty((L + 1) * (L + 2) // 2 + 1, dtype=np.int64)
    cj = np.empty((L + 1) * (L + 2) // 2 + 1, dtype=np.int64)
    for i in range(0, q):
        a = full[i]
        b = full[i + 1]
        # pickup reachability (earliest start at pickup)
        arr_p = Bc[i] + service[a] + time[a, p_node]
        if arr_p > tw[p_node, 1] + EPS:
            continue
        ok = True
        for c in range(m):
            if load[i, c] + demand[p_node, c] > cap[k, c]:
                ok = False
        if not ok:
            continue
        dp = dist[a, p_node] + dist[p_node, b] - dist[a, b]
        for j in range(i, q):
            if j > i:
                # delivery after position j; load between i+1..j must accommodate the request
                over = False
                for c in range(m):
                    if load[j, c] + demand[p_node, c] > cap[k, c]:
                        over = True
                if over:
                    break
                c1 = full[j]
                c2 = full[j + 1]
                delta = dp + dist[c1, d_node] + dist[d_node, c2] - dist[c1, c2]
            else:
                delta = dist[a, p_node] + dist[p_node, d_node] + dist[d_node, b] - dist[a, b]
            cd[ncand] = delta
            ci[ncand] = i
            cj[ncand] = j
            ncand += 1
    if ncand == 0:
        return BIG, -1, -1
    order = np.argsort(cd[:ncand])
    newfull = np.empty(L + 4, dtype=np.int64)
    checks = 0
    for t in range(ncand):
        idx = order[t]
        i = ci[idx]
        j = cj[idx]
        # build new full sequence
        w = 0
        for p in range(0, i + 1):
            newfull[w] = full[p]
            w += 1
        newfull[w] = p_node
        w += 1
        for p in range(i + 1, j + 1):
            newfull[w] = full[p]
            w += 1
        newfull[w] = d_node
        w += 1
        for p in range(j + 1, q + 1):
            newfull[w] = full[p]
            w += 1
        if _schedule(newfull, q + 2, k, n, tw, service, time, demand, cap, start, end,
                     max_duration, max_ride, darp, B, pos):
            return cd[idx], i, j
        checks += 1
        if checks >= max_checks:
            break
    return BIG, -1, -1


def insert_into(seq_list, r, n, i, j):
    """Apply an insertion returned by best_insertion to a Python list of customer nodes."""
    # positions are in the full sequence (0 = depot); list index = position
    new = list(seq_list)
    new.insert(i, r)            # after full[i] => list index i
    new.insert(j + 1, n + r)    # after full[j] in the original => shifted by the pickup
    return new
