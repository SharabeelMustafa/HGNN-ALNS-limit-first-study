"""How well does the exact insertion cost predict the training label of the network?

On the held-out Validation-ML instances we rebuild the same destroyed states as the training
pipeline (learn/dataset.py, same seeds) and rank the candidate routes of every removed request by
  * the exact cheapest feasible insertion cost in the destroyed state (what ALNS computes anyway);
  * the HGNN route prior (through the same HGNNGuide code that is used inside the search);
  * the homogeneous GNN route prior.
Metric: top-1 accuracy and recall@5 against the label (the origin route in the elite solution).

Usage: PYTHONPATH=src python scripts/eval_insertion_baseline.py --out results/insertion_baseline.json
"""
import argparse
import glob
import math
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hgalns.solution import Evaluator, Solution            # noqa: E402
from hgalns.alns.operators import DESTROY                   # noqa: E402
from hgalns.alns.search import _q_range, ALNSConfig         # noqa: E402
from hgalns.learn.guide import load_model, HGNNGuide        # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "results", "insertion_baseline.json"))
    ap.add_argument("--states", type=int, default=30)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    res = {}
    for fam in ("lilim", "hdarp"):
        models = {m: load_model(os.path.join(REPO, "models", f"{fam}_{m}.pt")) for m in ("hgnn", "homgnn")}
        acc = {}
        srcs = sorted(glob.glob(os.path.join(REPO, "data", "labels", fam, "valml_*.pkl")))
        for idx, src in enumerate(srcs):
            rec = pickle.load(open(src, "rb"))
            inst, best = rec["instance"], rec["elites"][0]
            sub = fam if fam == "lilim" else ("hdarp n<=100" if inst.n <= 100 else "hdarp n>100")
            A = acc.setdefault(sub, {k: {"top1": 0, "rec5": 0, "rech": 0} for k in ("insertion", "hgnn", "homgnn", "random")} | {"cnt": 0, "inst": 0, "cand": 0})
            A["inst"] += 1
            ev = Evaluator(inst)
            guides = {m: HGNNGuide(mod, inst, ev, K=999, lam=0.0) for m, mod in models.items()}
            classes = ev.classes
            tv = {v: k for k, route in enumerate(best["routes"]) for v in route if v < inst.n}
            rng = np.random.default_rng(idx)                      # same seed as scripts/build_dataset.py
            lo, hi = _q_range(inst.n, ALNSConfig())
            ops = ["random", "shaw", "worst", "string", "route"]
            for t in range(a.states):
                sol = Solution(ev, routes=[list(r) for r in best["routes"]])
                q = int(rng.integers(lo, hi + 1))
                removed = [int(r) for r in DESTROY[ops[t % len(ops)]](sol, q, rng) if r in tv]
                if not removed:
                    continue
                cand = sol.candidate_vehicles()
                empty_rep = {int(classes[k]): k for k in cand if not sol.routes[k]}
                for g in guides.values():
                    g.prepare(sol, removed)
                for r in removed:
                    kv = tv[r]
                    tgt = kv if sol.routes[kv] else empty_rep.get(int(classes[kv]))
                    ks = [k for k in cand if inst.compat[r, k]]
                    if tgt is None or tgt not in ks:
                        continue
                    A["cnt"] += 1; A["cand"] += len(ks)
                    rank = {}
                    cost = [ev.insertion(sol.routes[k], k, r)[0] for k in ks]
                    rank["insertion"] = 1 + sum(c < cost[ks.index(tgt)] for c in cost)
                    for m, g in guides.items():
                        s = [g.S[(r, k)] for k in ks]
                        rank[m] = 1 + sum(x > s[ks.index(tgt)] for x in s)
                    kh = max(3, int(math.ceil(0.5 * len(ks))))        # the tuned Li & Lim pruning rule: top 50%, at least 3
                    for m, rk in rank.items():
                        A[m]["top1"] += rk == 1; A[m]["rec5"] += rk <= 5; A[m]["rech"] += rk <= kh
                    A["random"]["top1"] += 1.0 / len(ks); A["random"]["rec5"] += min(5, len(ks)) / len(ks)
                    A["random"]["rech"] += min(kh, len(ks)) / len(ks)
            if (idx + 1) % 10 == 0:
                print(fam, idx + 1, "/", len(srcs), flush=True)
        for sub, A in acc.items():
            c = max(1, A["cnt"])
            res[sub] = {"instances": A["inst"], "requests": A["cnt"], "mean_candidates": A["cand"] / c,
                        **{m: {"top1": A[m]["top1"] / c, "recall@5": A[m]["rec5"] / c, "recall@half": A[m]["rech"] / c} for m in ("random", "hgnn", "homgnn", "insertion")}}
            print(sub, json.dumps(res[sub]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
