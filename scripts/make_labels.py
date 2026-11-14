"""Generate training/validation instances and elite solutions (Final Methodology Section 8.1).

For each generated instance we run the classical ALNS with M seeds and store the instance
and the M final solutions (elite set). Destroyed states are sampled later, on the fly,
during HGNN training.

Usage:
  python scripts/make_labels.py --family hdarp --split train --count 600 --nmin 25 --nmax 100 \
         --seeds 3 --budget-per-req 0.3 --workers 12
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

SPLIT_SEED = {"train": 1_000_000, "valml": 2_000_000, "valopt": 3_000_000}


def job(args):
    family, split, idx, nmin, nmax, seeds, bpr, bmin, out = args
    from hgalns.generators import generate
    from hgalns.alns import ALNSConfig, run_alns
    path = os.path.join(out, f"{split}_{idx:05d}.pkl")
    if os.path.exists(path):
        return path, 0.0, "skip"
    rng = np.random.default_rng(SPLIT_SEED[split] + idx)
    n = int(rng.integers(nmin, nmax + 1))
    inst = generate(family, n, rng, name=f"{family}_{split}_{idx:05d}")
    budget = max(bmin, bpr * n)
    elites = []
    t = time.time()
    for s in range(seeds):
        res = run_alns(inst, ALNSConfig(budget=budget, seed=1000 * idx + s))
        sol = res["solution"]
        elites.append({"routes": [list(map(int, r)) for r in sol.routes], "distance": res["distance"],
                       "vehicles": res["vehicles"], "unserved": res["unserved"], "seed": s})
    elites.sort(key=lambda e: (e["unserved"], e["vehicles"] if inst.objective == "lexi" else 0, e["distance"]))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"instance": inst, "elites": elites, "budget": budget}, f)
    os.replace(tmp, path)
    return path, time.time() - t, f"n={n} best={elites[0]['distance']:.1f} veh={elites[0]['vehicles']} unserved={elites[0]['unserved']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["lilim", "hdarp"])
    ap.add_argument("--split", default="train", choices=list(SPLIT_SEED))
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--nmin", type=int, default=25)
    ap.add_argument("--nmax", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget-per-req", type=float, default=0.3)
    ap.add_argument("--budget-min", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(__file__), "..", "data", "labels", a.family)
    os.makedirs(out, exist_ok=True)
    jobs = [(a.family, a.split, i, a.nmin, a.nmax, a.seeds, a.budget_per_req, a.budget_min, out)
            for i in range(a.start, a.start + a.count)]
    # warm up numba cache in the parent process
    from hgalns.generators import generate
    from hgalns.alns import ALNSConfig, run_alns
    run_alns(generate(a.family, 12, np.random.default_rng(0)), ALNSConfig(budget=0.5, seed=0))
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(job, j) for j in jobs]
        for f in as_completed(futs):
            path, dt, msg = f.result()
            done += 1
            print(f"[{done}/{len(jobs)} {time.time() - t0:7.0f}s] {os.path.basename(path)} {dt:5.1f}s {msg}", flush=True)


if __name__ == "__main__":
    main()
