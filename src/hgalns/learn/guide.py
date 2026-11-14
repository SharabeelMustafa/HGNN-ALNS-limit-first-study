"""HGNN guidance inside ALNS (Final Methodology Sections 7.4, 9.2, 9.4).

The static encoder runs ONCE per instance (inside the timed budget, with torch).
Everything that depends only on the static embeddings is then precomputed, so that during
search a (request, route) score costs a few NumPy vector operations:

  s(r,k) = w2 . act( A_r + G_k + chi_rk W1d + b1 ) + b2            (7.7)
     A_r = h^P_r W1a + h^D_r W1b           (static, per request)
     G_k = g_k W1c                          (per route, cached by route content)
     g_k = h_T(k) + sum_v softmax_v(b_v) (W_g h_v + b_g) + xi_k W_s + b_s   (7.6)
     b_v = w_beta . h_v                     (static, per stop)

API used by the repair operators (Section 9.4):
  prepare(sol, reqs) / route_changed(sol, k) / candidates(r, cand) / bias(r, k)
and the misplacement destroy operator (9.2).
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.special import erf

from ..graph import build_static_graph, route_state, _scale, F_CHI
from .model import HGNN


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = HGNN(**ck["hparams"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    m.hparams = ck["hparams"]
    return m


def _gelu(x):
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def _relu(x):
    return np.maximum(x, 0.0)


class HGNNGuide:
    def __init__(self, model, inst, ev, K=5, lam=0.5, T_pi=1.0, prune=True, prior=True, knn=16):
        self.inst, self.ev = inst, ev
        self.K = K if prune else 10 ** 9
        self.lam = lam if prior else 0.0
        self.T_pi = T_pi
        n = inst.n
        self.n = n
        g = build_static_graph(inst, knn=knn)
        self.H, _, _, self.capmax, self.dbar = _scale(inst)
        self.classes = g["classes"]
        N = 2 * n + g["C"]
        ntype = np.concatenate([np.zeros(n), np.ones(n), np.full(g["C"], 2)]).astype(np.int64)
        hp = getattr(model, "hparams", {})
        self.act = _gelu if hp.get("act", "gelu") == "gelu" else _relu
        self.use_xi = hp.get("route_state", True)
        with torch.no_grad():
            h, e = model.encode(torch.as_tensor(g["x_stop"]), torch.as_tensor(g["x_class"]),
                                torch.arange(2 * n), torch.arange(2 * n, N), torch.as_tensor(ntype),
                                torch.as_tensor(g["edge_index"]), torch.as_tensor(g["edge_rel"]),
                                torch.as_tensor(g["edge_attr"]))
            # co-routing probabilities on request-request graph edges (static)
            ei, rel = g["edge_index"], g["edge_rel"]
            sel = np.flatnonzero((rel >= 2) & (rel <= 5))
            ra, rb = ei[0, sel] % n, ei[1, sel] % n
            keep = ra != rb
            sel, ra, rb = sel[keep], ra[keep], rb[keep]
            co = np.zeros((n, n), dtype=np.float32)
            if len(sel):
                u = model.co_logits(h, e, torch.as_tensor(ei[0, sel]), torch.as_tensor(ei[1, sel]),
                                    torch.as_tensor(sel))
                pr = torch.sigmoid(u).numpy()
                np.maximum.at(co, (ra, rb), pr)
                np.maximum.at(co, (rb, ra), pr)
            self.co = co
            H_ = h.numpy().astype(np.float64)
            d = H_.shape[1]
            Wg = model.w_g.weight.numpy().astype(np.float64); bg = model.w_g.bias.numpy()
            wb = model.w_beta.weight.numpy()[0].astype(np.float64)
            Ws = model.w_s.weight.numpy().astype(np.float64); bs = model.w_s.bias.numpy()
            W1 = model.score[0].weight.numpy().astype(np.float64); b1 = model.score[0].bias.numpy()
            w2 = model.score[2].weight.numpy()[0].astype(np.float64); b2 = float(model.score[2].bias.numpy()[0])
        hs = H_[:2 * n]
        self.blog = hs @ wb                              # (2n,)
        self.WGh = hs @ Wg.T + bg                        # (2n, d)
        self.hT = H_[2 * n:]                             # (C, d)
        self.Ws, self.bs = Ws.T, bs                      # (F_XI, d)
        self.A = H_[:n] @ W1[:, :d].T + H_[n:2 * n] @ W1[:, d:2 * d].T + b1   # (n, d)
        self.W1c = W1[:, 2 * d:3 * d].T                  # (d, d)
        self.W1d = W1[:, 3 * d:].T                       # (F_CHI, d)
        self.w2, self.b2 = w2, b2
        self._G = {}
        self.S = {}
        self._lp = {}
        self.version = 0
        self.n_calls = 0

    # ------------------------------------------------------------------------- scoring
    def _Gk(self, sol, k):
        route = sol.routes[k]
        key = (k, tuple(route))
        G = self._G.get(key)
        if G is None:
            g = self.hT[int(self.classes[k])].copy()
            if route:
                idx = np.asarray(route)
                b = self.blog[idx]
                w = np.exp(b - b.max())
                w /= w.sum()
                g += w @ self.WGh[idx]
            if self.use_xi:
                g += route_state(self.inst, self.ev, route, k, self.H, self.capmax) @ self.Ws + self.bs
            G = g @ self.W1c
            if len(self._G) > 50000:
                self._G.clear()
            self._G[key] = G
        return G

    def _chi(self, sol, reqs, k):
        inst, n, H, T = self.inst, self.n, self.H, self.inst.time
        route = sol.routes[k]
        stops = [int(inst.start[k])] + list(route) + [int(inst.end[k])]
        ra = np.asarray(reqs)
        f = np.zeros((len(reqs), F_CHI))
        f[:, 0] = T[np.ix_(stops, ra)].min(0) / H
        f[:, 1] = T[np.ix_(ra + n, stops)].min(1) / H
        if route:
            ok, B = self.ev.schedule(route, k)
            lo = inst.tw[ra, 0]
            hi = np.minimum(inst.tw[ra + n, 1], H)
            f[:, 2] = np.maximum(0.0, np.minimum(B[-1], hi) - np.maximum(B[0], lo)) / H
        else:
            f[:, 2] = 1.0
        f[:, 3] = inst.compat[ra, k]
        return f

    def _score(self, sol, reqs, ks):
        if not reqs or not ks:
            return
        ra = np.asarray(reqs)
        A = self.A[ra]
        for k in ks:
            empty = not sol.routes[k]
            if empty:
                ck = ("empty", int(self.classes[k]))
                if all((r, ck) in self.S for r in reqs):
                    for r in reqs:
                        self.S[(r, k)] = self.S[(r, ck)]
                    continue
            z = A + self._Gk(sol, k) + self._chi(sol, reqs, k) @ self.W1d
            s = self.act(z) @ self.w2 + self.b2
            for r, v in zip(reqs, s):
                self.S[(r, k)] = float(v)
                if empty:
                    self.S[(r, ck)] = float(v)
        self.version += 1
        self.n_calls += 1

    # ------------------------------------------------------------------------- API
    def prepare(self, sol, reqs):
        """Score all (request, candidate) pairs ONCE on the destroyed state - the same kind of
        state the HGNN was trained on (Section 8.1)."""
        self.sol = sol
        self.S = {k: v for k, v in self.S.items() if isinstance(k[1], tuple)}   # keep per-class empty scores
        self._lp = {}
        self.version = 0
        self.pending = list(reqs)
        self._score(sol, self.pending, sol.candidate_vehicles())

    def route_changed(self, sol, k):
        """Scores are kept from the destroyed state; only new candidates (e.g. a fresh empty
        vehicle representative) are scored lazily in _logpi."""
        return

    def _logpi(self, r, cand):
        cached = self._lp.get(r)
        if cached is not None and cached[0] == cand:
            return cached[1]
        S = self.S
        miss = [k for k in cand if (r, k) not in S]
        if miss:
            self._score(self.sol, [r], miss)
        comp = self.inst.compat[r]
        ks = [k for k in cand if comp[k]]
        if not ks:
            lp = {}
        else:
            inv = 1.0 / self.T_pi
            vals = [S[(r, k)] * inv for k in ks]
            mx = max(vals)
            lse = mx + math.log(sum(math.exp(v - mx) for v in vals))
            lp = {k: v - lse for k, v in zip(ks, vals)}
        self._lp[r] = (list(cand), lp)
        return lp

    def candidates(self, r, cand):
        """Top-K routes by pi(k|r) (9.6). K >= 1: absolute count; 0 < K < 1: share of the
        compatible candidates (at least 3), so pruning scales with the number of routes."""
        lp = self._logpi(r, cand)
        self._cand = cand
        K = self.K
        if K < 1:
            K = max(3, int(math.ceil(K * len(lp))))
        if len(lp) <= K:
            return list(lp)
        return sorted(lp, key=lambda k: -lp[k])[:K]

    def bias(self, r, k):
        if self.lam == 0.0:
            return 0.0
        lp = self._lp[r][1].get(k)
        return 0.0 if lp is None else -self.lam * self.dbar * lp

    # ------------------------------------------------------------------------- destroy
    def misplacement_removal(self, sol, q, rng, p=6, **kw):
        n = self.n
        mu = []
        for k, route in enumerate(sol.routes):
            reqs = [v for v in route if v < n]
            if not reqs:
                continue
            if len(reqs) == 1:
                mu.append((1.0, reqs[0]))
                continue
            sub = self.co[np.ix_(reqs, reqs)]
            mean = (sub.sum(1) - np.diag(sub)) / (len(reqs) - 1)
            mu.extend((1.0 - float(v), r) for r, v in zip(reqs, mean))
        if not mu:
            return []
        mu.sort(reverse=True)
        removed = []
        while len(removed) < q and mu:
            removed.append(mu.pop(int((rng.random() ** p) * len(mu)))[1])
        sol.remove(removed)
        return removed


class OracleGuide(HGNNGuide):
    """Upper-bound diagnostic (follow-up experiment): the 'prediction' is read from a reference
    solution instead of the network. A request scores a route by the share of its reference
    route-mates that are in that route now. Same API, same top-K rule and same log-prior bias."""

    def __init__(self, ref_routes, inst, ev, K=5, lam=0.5, T_pi=1.0, prune=True, prior=True, beta=6.0):
        self.inst, self.ev = inst, ev
        self.K = K if prune else 10 ** 9
        self.lam = lam if prior else 0.0
        self.T_pi = T_pi
        n = inst.n
        self.n = n
        self.H, _, _, self.capmax, self.dbar = _scale(inst)
        self.classes = np.asarray(ev.classes)
        lab = np.full(n, -1, dtype=np.int64)
        for k, route in enumerate(ref_routes):
            for v in route:
                if v < n:
                    lab[v] = k
        self.lab = lab
        self.nlab = len(ref_routes) + 1
        co = (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0)
        np.fill_diagonal(co, False)
        self.co = co.astype(np.float32)
        self.nmates = np.maximum(1, co.sum(1))
        self.ref_class = np.array([int(self.classes[k]) if k >= 0 else -1 for k in lab])
        self.beta = beta
        self.S, self._lp = {}, {}
        self.version = 0
        self.n_calls = 0

    def _score(self, sol, reqs, ks):
        if not reqs or not ks:
            return
        n, lab = self.n, self.lab
        cnt = {}
        tot = np.zeros(self.nlab)
        for k, route in enumerate(sol.routes):
            if route:
                c = np.bincount(lab[[v for v in route if v < n]] + 1, minlength=self.nlab)
                cnt[k] = c
                tot += c
        for k in ks:
            c = cnt.get(k)
            for r in reqs:
                if c is not None:
                    s = self.beta * c[lab[r] + 1] / self.nmates[r] if lab[r] >= 0 else 0.0
                elif lab[r] >= 0 and tot[lab[r] + 1] == 0:
                    s = self.beta * (1.0 if int(self.classes[k]) == self.ref_class[r] else 0.5)
                else:
                    s = 0.0
                self.S[(r, k)] = float(s)
        self.version += 1
        self.n_calls += 1

    def prepare(self, sol, reqs):
        self.sol = sol
        self.S, self._lp = {}, {}
        self.version = 0
        self.pending = list(reqs)
        self._score(sol, self.pending, sol.candidate_vehicles())


class CoGuide(OracleGuide):
    """Pilot for the revised methodology: the guide reads a SOFT, instance-level co-routing matrix
    (n x n, values in [0, 1]; e.g. the co-routing head of the network) instead of a reference solution.
    A request scores a route by the share of its co-routing mass (over all placed requests) that sits
    in that route now. Same top-K rule and log-prior bias as the other guides."""

    def __init__(self, co, inst, ev, K=5, lam=0.5, T_pi=1.0, prune=True, prior=True, beta=6.0):
        self.inst, self.ev = inst, ev
        self.K = K if prune else 10 ** 9
        self.lam = lam if prior else 0.0
        self.T_pi = T_pi
        self.n = inst.n
        self.H, _, _, self.capmax, self.dbar = _scale(inst)
        self.classes = np.asarray(ev.classes)
        co = np.array(co, dtype=np.float64)
        np.fill_diagonal(co, 0.0)
        self.co = co
        self.beta = beta
        self.S, self._lp = {}, {}
        self.version = 0
        self.n_calls = 0

    def _score(self, sol, reqs, ks):
        if not reqs or not ks:
            return
        n, co = self.n, self.co
        ra = np.asarray(reqs)
        placed = [v for route in sol.routes for v in route if v < n]
        tot = co[np.ix_(ra, placed)].sum(1) if placed else np.zeros(len(ra))
        for k in ks:
            route = [v for v in sol.routes[k] if v < n]
            m = co[np.ix_(ra, route)].sum(1) if route else None
            for i, r in enumerate(reqs):
                if m is not None:
                    s = self.beta * m[i] / tot[i] if tot[i] > 1e-9 else 0.0
                else:
                    s = self.beta * 0.5 if tot[i] <= 1e-9 else 0.0
                self.S[(r, k)] = float(s)
        self.version += 1
        self.n_calls += 1


class SelfGuide(OracleGuide):
    """Pilot: 'elite consensus' without a network. The reference solution is the best solution of the
    run itself, refreshed whenever a new best is found (search.py calls on_new_best)."""

    def __init__(self, inst, ev, K=5, lam=0.5, T_pi=1.0, prune=True, prior=True, beta=6.0):
        super().__init__([], inst, ev, K=K, lam=lam, T_pi=T_pi, prune=prune, prior=prior, beta=beta)

    def on_new_best(self, sol):
        n = self.n
        lab = np.full(n, -1, dtype=np.int64)
        for k, route in enumerate(sol.routes):
            for v in route:
                if v < n:
                    lab[v] = k
        self.lab = lab
        self.nlab = len(sol.routes) + 1
        co = (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0)
        np.fill_diagonal(co, False)
        self.co = co.astype(np.float32)
        self.nmates = np.maximum(1, co.sum(1))
        self.ref_class = np.array([int(self.classes[k]) if k >= 0 else -1 for k in lab])

    def candidates(self, r, cand):
        if self.lab[r] < 0:                      # no information about this request yet: no pruning
            return list(self._logpi(r, cand))
        return super().candidates(r, cand)


def consensus_co(ref_solutions, n):
    """Soft co-routing matrix: share of the reference solutions in which two requests share a route."""
    co = np.zeros((n, n))
    for routes in ref_solutions:
        lab = np.full(n, -1, dtype=np.int64)
        for k, route in enumerate(routes):
            for v in route:
                if v < n:
                    lab[v] = k
        co += (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0)
    co /= max(1, len(ref_solutions))
    np.fill_diagonal(co, 0.0)
    return co


def static_relatedness(inst, weights=(9.0, 3.0, 5.0)):
    """Request-request relatedness from instance data only (Shaw / Ropke-Pisinger form with time-window
    centres instead of scheduled times): distance of the pickups and of the deliveries, time-window
    distance, and overlap of the compatible vehicles. Smaller = more related."""
    n = inst.n
    D = inst.dist
    phi, chi, omega = weights
    dmax = D.max() + 1e-9
    c = inst.tw.mean(1)
    tmax = inst.tw[:, 1].max() + 1e-9
    comp = inst.compat.astype(np.float64)
    inter = comp @ comp.T
    mn = np.minimum.outer(comp.sum(1), comp.sum(1)).clip(min=1.0)
    R = (phi * (D[:n, :n] + D[n:2 * n, n:2 * n]) / (2 * dmax)
         + chi * (np.abs(c[:n, None] - c[None, :n]) + np.abs(c[n:2 * n, None] - c[None, n:2 * n])) / (2 * tmax)
         + omega * (1.0 - inter / mn))
    return R


def static_relatedness_co(inst, knn=16):
    """Soft co-routing proxy without learning: exp(-relatedness / tau) on the k most related requests."""
    R = static_relatedness(inst)
    n = inst.n
    np.fill_diagonal(R, np.inf)
    k = min(knn, n - 1)
    part = np.sort(R, axis=1)
    tau = float(np.median(part[:, min(4, n - 2)])) + 1e-9
    co = np.exp(-R / tau)
    thr = part[:, k - 1][:, None]
    co = np.where(R <= thr, co, 0.0)
    co = np.maximum(co, co.T)
    np.fill_diagonal(co, 0.0)
    return co
