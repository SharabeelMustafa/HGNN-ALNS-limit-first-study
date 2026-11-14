"""Independent solution checker (Final Methodology Section 9.7 / gate G3).

Deliberately shares NO code with feasibility.py. It verifies a solution given as
    routes:   list (per vehicle) of customer-node lists
    schedule: list (per vehicle) of service-start times for [start, *route, end]
against the model (3.3)-(3.10), using the ORIGINAL resource semantics (raw demand/capacity
and the Parragh/Malheiros upgrade rules), not the transformed capacity form.
"""
from __future__ import annotations

import math

TOL = 1e-4


def _fits(raw_load, raw_cap, model):
    """Can the current on-board passengers be seated in the vehicle?"""
    if model == "single":
        return all(raw_load[i] <= raw_cap[i] for i in range(len(raw_cap)))
    # nested4: resources [0 staff, 1 patient seat, 2 stretcher, 3 wheelchair]
    acc, seat, stretch, wheel = raw_load
    c0, c1, c2, c3 = raw_cap
    if wheel > c3 or stretch > c2:
        return False
    free2 = c2 - stretch                 # stretchers left for seated patients / companions
    seat_left = seat - c1                # seated patients that do not fit on patient seats
    if seat_left > 0:
        if seat_left > free2:
            return False
        free2 -= seat_left
        free1 = 0
    else:
        free1 = -seat_left               # unused patient seats
    acc_left = acc - c0
    return acc_left <= free1 + free2


def check_solution(inst, routes, schedule, bank=()):
    """Return (ok, list_of_violation_strings, metrics_dict)."""
    n = inst.n
    errors = []
    seen = {}
    total_dist = 0.0
    used = 0
    for k, route in enumerate(routes):
        if not route:
            continue
        used += 1
        seq = [int(inst.start[k])] + [int(v) for v in route] + [int(inst.end[k])]
        times = schedule[k]
        if len(times) != len(seq):
            errors.append(f"veh {k}: schedule length {len(times)} != route length {len(seq)}")
            continue
        for v in route:
            if not (0 <= v < 2 * n):
                errors.append(f"veh {k}: invalid node {v}")
            if v in seen:
                errors.append(f"node {v} visited twice (vehicles {seen[v]} and {k})")
            seen[v] = k
        position = {v: p for p, v in enumerate(seq)}
        # pairing, precedence, compatibility
        for v in route:
            if v < n:
                if (n + v) not in position:
                    errors.append(f"veh {k}: pickup {v} without its delivery")
                elif position[n + v] < position[v]:
                    errors.append(f"veh {k}: delivery of {v} before pickup")
                if not inst.compat[v, k]:
                    errors.append(f"veh {k}: request {v} not compatible")
            elif (v - n) not in position:
                errors.append(f"veh {k}: delivery {v} without its pickup")
        # times
        for p in range(len(seq)):
            v = seq[p]
            e, l = inst.tw[v]
            if times[p] < e - TOL or times[p] > l + TOL:
                errors.append(f"veh {k}: node {v} served at {times[p]:.3f} outside [{e},{l}]")
            if p > 0:
                u = seq[p - 1]
                if times[p] < times[p - 1] + inst.service[u] + inst.time[u, v] - TOL:
                    errors.append(f"veh {k}: travel time violated {u}->{v}")
        if times[-1] - times[0] > inst.max_duration[k] + TOL:
            errors.append(f"veh {k}: duration {times[-1] - times[0]:.3f} > {inst.max_duration[k]}")
        # ride times
        for v in route:
            if v < n and (n + v) in position:
                ride = times[position[n + v]] - (times[position[v]] + inst.service[v])
                if ride > inst.max_ride[v] + TOL:
                    errors.append(f"veh {k}: ride time of {v} = {ride:.3f} > {inst.max_ride[v]}")
        # capacity with original semantics
        raw_cap = [int(c) for c in inst.raw_cap[k]]
        load = [0] * len(raw_cap)
        for v in route:
            for i in range(len(load)):
                load[i] += int(inst.raw_demand[v, i])
            if not _fits(load, raw_cap, inst.resource_model):
                errors.append(f"veh {k}: capacity exceeded after node {v}: load {load} cap {raw_cap}")
                break
        total_dist += sum(inst.dist[seq[p], seq[p + 1]] for p in range(len(seq) - 1)) * inst.var_cost[k]
        total_dist += inst.fixed_cost[k]
    served = {v for v in seen if v < n}
    missing = [r for r in range(n) if r not in served and r not in set(bank)]
    if missing:
        errors.append(f"{len(missing)} requests neither served nor banked: {missing[:10]}")
    metrics = {"vehicles": used, "distance": total_dist, "unserved": len(set(bank)),
               "served": len(served)}
    return (len(errors) == 0), errors, metrics


def is_close(a, b, tol=1e-2):
    return math.isclose(a, b, abs_tol=tol)
