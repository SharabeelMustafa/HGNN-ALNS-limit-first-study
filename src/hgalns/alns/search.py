"""ALNS main loop (Final Methodology Section 9; pseudocode Section 4.4).

run_alns(inst, cfg) -> result dict with the best solution, objective values and the
best-solution trajectory (for the primal integral), all under a wall-clock budget.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..solution import Evaluator, Solution
from .operators import DESTROY, REPAIR
from .control import make_selector, make_acceptance, SIGMA


@dataclass
class ALNSConfig:
    budget: float = 30.0                    # wall-clock seconds (includes construction)
    seed: int = 0
    destroy: tuple = ("random", "shaw", "worst", "route", "string")
    repair: tuple = ("greedy", "regret2", "regret3")
    selection: str = "roulette"             # roulette | uniform | swucb
    acceptance: str = "sa_time"             # sa_time | sa_iter | rrt
    q_min: int = 4
    q_max: int = 100
    xi: float = 0.4
    noise: float = 0.025                    # eta in (9.4)
    noise_prob: float = 0.5                 # choose noisy/clean repair (R&P choose adaptively; we randomise)
    vehicle_stage_frac: float = 0.4         # share of budget for vehicle minimisation (lexi objective)
    max_iters: int = 10 ** 9
    extra_destroy: dict = field(default_factory=dict)   # name -> callable (e.g. HGNN removal)
    guide: object = None                    # HGNN guidance (None = classical)
    guide_factory: object = None            # callable(inst, ev) -> guide; built INSIDE the timed budget
    gnn_destroy: bool = False               # add the HGNN misplacement-removal operator (9.2)
    log_every: float = 0.0
    iter_clock: bool = False                # follow-up experiments: the clock is the iteration counter
                                            # (elapsed = budget * iters / max_iters), so runs have equal iterations
    diag: bool = False                      # count how often repair puts a request back into its origin route
    sp: bool = False                        # set partitioning over the routes of accepted solutions, at the end
    sp_frac: float = 0.1                    # share of the time budget kept for it (time clock); with the iteration clock
                                            # it runs after the last iteration with a limit of sp_frac * budget seconds
    best_of: int = 1                        # pilot: try m destroy+repair moves from the current solution and keep the
                                            # best one (a perfect "which neighbourhood" selector); 1 = classical ALNS


def _q_range(n, cfg):
    hi = max(1, min(cfg.q_max, int(cfg.xi * n)))
    lo = min(cfg.q_min, hi)
    return lo, hi


def construct(ev, rng, guide=None):
    sol = Solution(ev, bank=range(ev.n))
    REPAIR["regret2"](sol, rng, noise=0.0, guide=guide)
    return sol


class _Run:
    def __init__(self, inst, cfg: ALNSConfig):
        self.inst = inst
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t0 = time.perf_counter()
        self.ev = Evaluator(inst)
        self.traj = []          # (elapsed, objective, vehicles, banked)
        self.iters = 0
        self.guide = cfg.guide
        if cfg.guide_factory is not None:
            self.guide = cfg.guide_factory(inst, self.ev)
        self.t_guide = time.perf_counter() - self.t0
        extra = dict(cfg.extra_destroy)
        if cfg.gnn_destroy and self.guide is not None:
            extra["gnn"] = self.guide.misplacement_removal
        dnames = list(cfg.destroy) + list(extra)
        self.dfun = [DESTROY.get(d) or extra[d] for d in dnames]
        self.dnames = dnames
        self.rnames = list(cfg.repair)
        self.sel_d = make_selector(cfg.selection, self.dnames, self.rng)
        self.sel_r = make_selector(cfg.selection, self.rnames, self.rng)
        self.visited = set()
        self.pool = None
        if cfg.sp:
            from .setpart import RoutePool
            self.pool = RoutePool(self.ev)
        self.diag = {"iters": 0, "removed": 0, "defined": 0, "undo": 0, "top1_origin": 0, "top1_defined": 0,
                     "same": 0, "revisit": 0, "accepted": 0, "improved": 0, "new_best": 0}

    def elapsed(self):
        if self.cfg.iter_clock:
            return self.cfg.budget * min(1.0, self.iters / max(1, self.cfg.max_iters))
        return time.perf_counter() - self.t0

    def _diag(self, cur, new, removed):
        """Origin route of a removed request = the route it was taken from, if that route is still
        non-empty after the destroy step (repair never moves the requests that stayed)."""
        d = self.diag
        loc0 = cur.request_vehicle()
        loc1 = new.request_vehicle()
        lp_all = getattr(self.guide, "_lp", None) if self.guide is not None else None
        stay = {}
        rem = set(int(r) for r in removed)
        for r, k in loc0.items():
            if r not in rem:
                stay[k] = stay.get(k, 0) + 1
        for r in rem:
            d["removed"] += 1
            k0 = loc0.get(r)
            if k0 is None or k0 not in stay:
                continue
            d["defined"] += 1
            d["undo"] += loc1.get(r) == k0
            if lp_all is not None and r in lp_all and lp_all[r][1]:
                lp = lp_all[r][1]
                if k0 in lp:
                    d["top1_defined"] += 1
                    d["top1_origin"] += max(lp, key=lp.get) == k0

    def log(self, sol):
        self.traj.append((self.elapsed(), sol.distance, sol.n_vehicles, len(sol.bank)))

    def iterate(self, cur, best, acc, deadline, best_fn):
        """One ALNS iteration. best_fn(s) -> True if s is a new global best."""
        cfg, rng = self.cfg, self.rng
        lo, hi = _q_range(self.inst.n, cfg)
        best_trial = None
        for _ in range(max(1, cfg.best_of)):
            q = int(rng.integers(lo, hi + 1))
            di = self.sel_d.choose()
            ri = self.sel_r.choose()
            new = cur.copy()
            removed = self.dfun[di](new, q, rng)
            noise = cfg.noise if rng.random() < cfg.noise_prob else 0.0
            REPAIR[self.rnames[ri]](new, rng, noise=noise, guide=self.guide)
            if best_trial is None or new.objective() < best_trial[0].objective() - 1e-12:
                best_trial = (new, removed, di, ri)
        new, removed, di, ri = best_trial
        self.iters += 1
        f_new, f_cur = new.objective(), cur.objective()
        sigma = 0.0
        key = hash(new.key())
        unvisited = key not in self.visited
        if cfg.diag:
            dg = self.diag
            dg["iters"] += 1
            self._diag(cur, new, removed or [])
            dg["same"] += new.key() == cur.key()
            dg["revisit"] += not unvisited
            dg["improved"] += f_new < f_cur - 1e-9
        if len(self.visited) < 200_000:
            self.visited.add(key)
        if best_fn(new):
            sigma = SIGMA[0]
            cur = new
            best = new.copy()
            self.log(best)
            self.diag["new_best"] += 1
            if hasattr(self.guide, "on_new_best"):
                self.guide.on_new_best(best)
            self.diag["accepted"] += 1
        elif acc.accept(f_new, f_cur, best.objective(), self.elapsed()):
            if unvisited:
                sigma = SIGMA[1] if f_new < f_cur else SIGMA[2]
            cur = new
            self.diag["accepted"] += 1
        if self.pool is not None and cur is new and not new.bank:
            self.pool.add(new)
        self.sel_d.update(di, sigma)
        self.sel_r.update(ri, sigma)
        return cur, best


def run_alns(inst, cfg: ALNSConfig):
    R = _Run(inst, cfg)
    rng = R.rng
    sol = construct(R.ev, rng, guide=R.guide)
    if hasattr(R.guide, "on_new_best"):
        R.guide.on_new_best(sol)
    R.log(sol)
    budget = cfg.budget * (1.0 - cfg.sp_frac) if (cfg.sp and not cfg.iter_clock) else cfg.budget
    f0 = max(sol.distance, 1e-6)

    if inst.objective == "lexi":
        best_full = sol.copy() if not sol.bank else None
        # -------- stage 1: vehicle minimisation (Ropke & Pisinger 2006, Section 9.1) ----
        stage_end = cfg.vehicle_stage_frac * budget
        cur = sol.copy()
        if not cur.bank:
            cur = _drop_route(cur, rng)
        acc = make_acceptance(cfg.acceptance, f0, stage_end, rng)
        best_stage = cur.copy()

        def better_stage(s):
            return (len(s.bank), s.distance) < (len(best_stage.bank), best_stage.distance)

        while R.elapsed() < stage_end and R.iters < cfg.max_iters and cur.max_routes is not None:
            def bf(s):
                nonlocal best_stage
                if better_stage(s):
                    best_stage = s.copy()
                    return True
                return False
            cur, _ = R.iterate(cur, best_stage, acc, stage_end, bf)
            if not best_stage.bank:                  # success: all served with fewer routes
                best_full = best_stage.copy()
                R.log(best_full)
                cur = _drop_route(best_full.copy(), rng)
                best_stage = cur.copy()
        if best_full is None:
            best_full = best_stage if not best_stage.bank else sol
        # -------- stage 2: distance minimisation with fixed fleet size ---------------
        cur = best_full.copy()
        cur.max_routes = best_full.n_vehicles
        best = cur.copy()
        acc = make_acceptance(cfg.acceptance, max(best.distance, 1e-6), budget - R.elapsed(), rng)
        t_stage2 = R.elapsed()
        acc.budget = max(1e-6, budget - t_stage2)

        class _Shift:   # acceptance clock relative to stage 2 start
            def __init__(self, a, t):
                self.a, self.t = a, t

            def accept(self, fn, fc, fb, el):
                return self.a.accept(fn, fc, fb, el - self.t)
        acc2 = _Shift(acc, t_stage2)
        while R.elapsed() < budget and R.iters < cfg.max_iters:
            def bf(s):
                nonlocal best
                if s.objective() < best.objective() - 1e-9:
                    best = s.copy()
                    return True
                return False
            cur, _ = R.iterate(cur, best, acc2, budget, bf)
        best.max_routes = None
    else:
        cur = sol.copy()
        best = sol.copy()
        acc = make_acceptance(cfg.acceptance, f0, budget, rng)
        while R.elapsed() < budget and R.iters < cfg.max_iters:
            def bf(s):
                nonlocal best
                if s.objective() < best.objective() - 1e-9:
                    best = s.copy()
                    return True
                return False
            cur, _ = R.iterate(cur, best, acc, budget, bf)

    sp_stats = None
    if R.pool is not None:
        tl = cfg.sp_frac * cfg.budget if cfg.iter_clock else cfg.budget - (time.perf_counter() - R.t0)
        best = R.pool.solve(best, tl, lex=inst.objective == "lexi")
        sp_stats = dict(R.pool.stats)
    R.log(best)
    return {
        "sp": sp_stats,
        "solution": best,
        "distance": best.distance,
        "vehicles": best.n_vehicles,
        "unserved": len(best.bank),
        "objective": best.objective(),
        "iterations": R.iters,
        "elapsed": R.elapsed(),
        "real_elapsed": time.perf_counter() - R.t0,
        "diag": dict(R.diag) if cfg.diag else None,
        "trajectory": R.traj,
        "insert_evals": R.ev.n_insert_evals,
        "t_guide_setup": R.t_guide,
        "guide_calls": getattr(R.guide, "n_calls", 0),
        "destroy_weights": dict(zip(R.dnames, getattr(R.sel_d, "w", np.ones(len(R.dnames))).tolist())),
        "repair_weights": dict(zip(R.rnames, getattr(R.sel_r, "w", np.ones(len(R.rnames))).tolist())),
    }


def _drop_route(sol, rng):
    """Remove the route with the fewest requests; forbid opening new vehicles."""
    used = [k for k, r in enumerate(sol.routes) if r]
    if len(used) <= 1:
        sol.max_routes = None
        return sol
    k = min(used, key=lambda k: (len(sol.routes[k]), rng.random()))
    sol.remove([v for v in sol.routes[k] if v < sol.ev.n])
    sol.max_routes = len(used) - 1
    return sol
