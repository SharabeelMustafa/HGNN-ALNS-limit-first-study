"""Operator selection (Section 9.5) and acceptance criteria (Section 9.6)."""
from __future__ import annotations

import math
from collections import deque

import numpy as np

SIGMA = (33.0, 9.0, 13.0)   # new global best, better than current (unvisited), accepted worse (unvisited)


class Roulette:
    """Ropke & Pisinger (2006): P(i) = w_i / sum w; segment-based weight update (9.8)-(9.9)."""

    def __init__(self, names, rng, segment=100, reaction=0.1):
        self.names = list(names)
        self.rng = rng
        self.w = np.ones(len(self.names))
        self.score = np.zeros(len(self.names))
        self.used = np.zeros(len(self.names))
        self.segment = segment
        self.reaction = reaction
        self.it = 0

    def choose(self):
        p = self.w / self.w.sum()
        return int(self.rng.choice(len(self.names), p=p))

    def update(self, i, sigma):
        self.score[i] += sigma
        self.used[i] += 1
        self.it += 1
        if self.it % self.segment == 0:
            for j in range(len(self.names)):
                if self.used[j] > 0:
                    self.w[j] = (1 - self.reaction) * self.w[j] + self.reaction * self.score[j] / self.used[j]
                self.w[j] = max(self.w[j], 1e-3)
            self.score[:] = 0
            self.used[:] = 0


class Uniform(Roulette):
    def choose(self):
        return int(self.rng.integers(len(self.names)))

    def update(self, i, sigma):
        pass


class SlidingWindowUCB:
    """(9.10): argmax mean reward in window W + C sqrt(2 ln N / N_i)."""

    def __init__(self, names, rng, window=100, c=0.5):
        self.names = list(names)
        self.rng = rng
        self.hist = deque(maxlen=window)
        self.c = c

    def choose(self):
        K = len(self.names)
        cnt = np.zeros(K)
        rew = np.zeros(K)
        for i, r in self.hist:
            cnt[i] += 1
            rew[i] += r
        if np.any(cnt == 0):
            return int(self.rng.choice(np.flatnonzero(cnt == 0)))
        tot = cnt.sum()
        ucb = rew / cnt + self.c * np.sqrt(2 * math.log(tot) / cnt)
        return int(np.argmax(ucb))

    def update(self, i, sigma):
        self.hist.append((i, sigma / SIGMA[0]))


def make_selector(kind, names, rng):
    if kind == "roulette":
        return Roulette(names, rng)
    if kind == "uniform":
        return Uniform(names, rng)
    if kind == "swucb":
        return SlidingWindowUCB(names, rng)
    raise ValueError(kind)


# --------------------------------------------------------------------------------------
class TimeSA:
    """Simulated annealing with time-based cooling (9.11)-(9.12)."""

    def __init__(self, f0, budget, w=0.05, wf=0.0005, rng=None):
        f0 = max(f0, 1e-9)
        self.T0 = -w * f0 / math.log(0.5)
        self.Tf = -wf * f0 / math.log(0.5)
        self.budget = budget
        self.rng = rng

    def temperature(self, elapsed):
        x = min(1.0, max(0.0, elapsed / self.budget))
        return self.T0 * (self.Tf / self.T0) ** x

    def accept(self, f_new, f_cur, f_best, elapsed):
        if f_new < f_cur:
            return True
        T = self.temperature(elapsed)
        return self.rng.random() < math.exp(-(f_new - f_cur) / T)


class IterSA(TimeSA):
    """Iteration-based geometric cooling (ablation A-ALNS-4): T <- c T per iteration."""

    def __init__(self, f0, budget, w=0.05, c=0.99975, rng=None):
        super().__init__(f0, budget, w=w, rng=rng)
        self.T = self.T0
        self.c = c

    def accept(self, f_new, f_cur, f_best, elapsed):
        self.T *= self.c
        if f_new < f_cur:
            return True
        return self.rng.random() < math.exp(-(f_new - f_cur) / max(self.T, 1e-12))


class RRT:
    """Record-to-record travel (ablation A-ALNS-3): accept if f_new <= (1+delta(t)) f_best."""

    def __init__(self, f0, budget, d0=0.05, rng=None):
        self.d0 = d0
        self.budget = budget

    def accept(self, f_new, f_cur, f_best, elapsed):
        x = min(1.0, elapsed / self.budget)
        return f_new <= (1.0 + self.d0 * (1.0 - x)) * f_best


def make_acceptance(kind, f0, budget, rng):
    if kind == "sa_time":
        return TimeSA(f0, budget, rng=rng)
    if kind == "sa_iter":
        return IterSA(f0, budget, rng=rng)
    if kind == "rrt":
        return RRT(f0, budget, rng=rng)
    raise ValueError(kind)
