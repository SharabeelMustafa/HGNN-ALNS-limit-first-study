"""Convert label files (instance + elites) into HGNN training samples (graph + destroyed states).

Usage: python scripts/build_dataset.py --family hdarp --split train --states 30 --workers 12
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")


def job(args):
    src, dst, states, seed = args
    if os.path.exists(dst):
        return dst
    from hgalns.learn.dataset import build_samples
    with open(src, "rb") as f:
        rec = pickle.load(f)
    sample = build_samples(rec, n_states=states, seed=seed)
    with open(dst + ".tmp", "wb") as f:
        pickle.dump(sample, f)
    os.replace(dst + ".tmp", dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--states", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    srcs = sorted(glob.glob(os.path.join(ROOT, "labels", a.family, f"{a.split}_*.pkl")))
    out = os.path.join(ROOT, "datasets", a.family, a.split)
    os.makedirs(out, exist_ok=True)
    jobs = [(s, os.path.join(out, os.path.basename(s)), a.states, i) for i, s in enumerate(srcs)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, d in enumerate(ex.map(job, jobs)):
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"{i + 1}/{len(jobs)} {os.path.basename(d)}", flush=True)


if __name__ == "__main__":
    main()
