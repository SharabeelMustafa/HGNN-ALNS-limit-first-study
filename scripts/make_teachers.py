"""Teacher solutions for the revised method: long classical-ALNS runs on generated instances.

For every instance of a split, run classical ALNS `--seeds` times with `--mult` times the iteration budget
that classical ALNS gets at 0.25 s per request (the tuning budget). The iteration clock is used, so the runs
are reproducible and do not depend on machine load or pauses. Resumable: finished runs are skipped.

Output: data/teachers/<family>/m<mult>/<split>_XXXXX.s<seed>.json  (routes, distance, vehicles, unserved, iterations)

Usage: python scripts/make_teachers.py --family hdarp --split train --limit 300 --seeds 2 --mult 8 --workers 8
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
TUNED = {"lilim": {"xi": 0.4, "noise": 0.025}, "hdarp": {"xi": 0.2, "noise": 0.0}}


def iter_model(family):
    """Iterations of classical ALNS at 0.25 s per request as a function of n: log N = a + b log n (tuning instances)."""
    import numpy as np
    it = json.load(open(os.path.join(REPO, "results", f"iter_budget_valopt_{family}.json")))
    xs, ys = [], []
    for name, v in it.items():
        p = os.path.join(REPO, "data", "labels", family, name + ".pkl")
        if os.path.exists(p):
            xs.append(np.log(pickle.load(open(p, "rb"))["instance"].n)); ys.append(np.log(v))
    b, a = np.polyfit(xs, ys, 1)
    return float(a), float(b)


def job(args):
    src, dst, seed, mult, a, b, family = args
    import math
    from hgalns.alns import ALNSConfig, run_alns
    inst = pickle.load(open(src, "rb"))["instance"]
    iters = int(round(mult * math.exp(a + b * math.log(inst.n))))
    cfg = ALNSConfig(budget=0.25 * inst.n, seed=1000 + seed, max_iters=iters, iter_clock=True, **TUNED[family])
    res = run_alns(inst, cfg)
    sol = res["solution"]
    out = {"routes": [list(map(int, r)) for r in sol.routes], "vehicles": res["vehicles"], "distance": res["distance"],
           "unserved": res["unserved"], "iterations": res["iterations"], "max_iters": iters, "seed": 1000 + seed, "n": inst.n,
           "real_elapsed": res["real_elapsed"]}
    with open(dst + f".tmp{os.getpid()}", "w") as f:
        json.dump(out, f)
    os.replace(dst + f".tmp{os.getpid()}", dst)
    return os.path.basename(dst), inst.n, res["distance"], res["real_elapsed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["lilim", "hdarp"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--mult", type=float, default=8.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-n", type=int, default=0, help="skip instances with more requests than this (0 = no limit)")
    a = ap.parse_args()
    srcs = sorted(glob.glob(os.path.join(REPO, "data", "labels", a.family, f"{a.split}_*.pkl")))
    if a.limit:
        import numpy as np                                             # evenly spread over the file list: the large
        idx = sorted(set(np.linspace(0, len(srcs) - 1, a.limit).round().astype(int).tolist()))   # instances are at its end
        srcs = [srcs[i] for i in idx]
    out = os.path.join(REPO, "data", "teachers", a.family, f"m{a.mult:g}")
    os.makedirs(out, exist_ok=True)
    ca, cb = iter_model(a.family)
    jobs = []
    for s in srcs:
        for sd in range(a.seeds):
            dst = os.path.join(out, os.path.basename(s)[:-4] + f".s{sd}.json")
            if not os.path.exists(dst):
                jobs.append((s, dst, sd, a.mult, ca, cb, a.family))
    random.Random(7).shuffle(jobs)
    print(f"{a.family}/{a.split}: {len(srcs)} instances x {a.seeds} seeds, {len(jobs)} runs to do (iterations = {a.mult:g} x exp({ca:.2f} + {cb:.2f} ln n))", flush=True)
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(job, j) for j in jobs]
        for f in as_completed(futs):
            try:
                name, n, d, t = f.result()
                done += 1
                if done % 20 == 0 or done == len(jobs):
                    print(f"[{done}/{len(jobs)}] {name} n={n} dist={d:.1f} {t:.0f}s", flush=True)
            except Exception as e:
                print("FAILED", repr(e), flush=True)
    left = sum(not os.path.exists(j[1]) for j in jobs)
    print("runs left:", left, flush=True)
    return 0 if left == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
