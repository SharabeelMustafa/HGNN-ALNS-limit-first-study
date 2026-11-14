"""Initial evaluation of the learned route prior on the held-out Validation-ML split (gate G1).

Compares, on exactly the same destroyed states and candidate sets:
  * HGNN (heterogeneous, main model) and homogeneous GNN (baseline B5);
  * nearest-route heuristic: rank routes by min travel time pickup->route + delivery->route
    (features chi_0 + chi_1, no learning);
  * random ranking (expected value, analytic).
Metrics: top-1 accuracy, recall@5 (= share of requests whose true route survives top-5 pruning), NDCG@3.

Usage: PYTHONPATH=src python scripts/eval_ranking.py --out results/ranking_eval.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from hgalns.learn.train import load_dir, evaluate          # noqa: E402
from hgalns.learn.guide import load_model                  # noqa: E402


def heuristic_metrics(data, K=5):
    top1 = rec = ndcg = cnt = 0
    r_top1 = r_rec = r_ndcg = 0.0
    m_sum = 0
    for sm in data:
        for st in sm["states"]:
            for a in range(len(st["removed"])):
                t = int(st["target"][a])
                if t < 0:
                    continue
                ok = np.flatnonzero(st["mask"][a])
                d = st["chi"][a, ok, 0] + st["chi"][a, ok, 1]
                order = ok[np.argsort(d, kind="stable")]
                rank = int(np.flatnonzero(order == t)[0]) + 1
                cnt += 1
                top1 += rank == 1
                rec += rank <= K
                ndcg += (1.0 / math.log2(rank + 1)) if rank <= 3 else 0.0
                m = len(ok)
                m_sum += m
                r_top1 += 1.0 / m
                r_rec += min(K, m) / m
                r_ndcg += sum(1.0 / math.log2(i + 1) for i in range(1, min(3, m) + 1)) / m
    c = max(cnt, 1)
    return ({"top1": top1 / c, f"recall@{K}": rec / c, "ndcg@3": ndcg / c},
            {"top1": r_top1 / c, f"recall@{K}": r_rec / c, "ndcg@3": r_ndcg / c},
            {"requests": cnt, "mean_candidates": m_sum / c})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/datasets")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="results/ranking_eval.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {}
    for fam in ("lilim", "hdarp"):
        data = load_dir(os.path.join(a.data, fam, "valml"))
        subsets = {"all": data}
        if fam == "hdarp":
            nn = lambda sm: sm["graph"]["x_stop"].shape[0] // 2   # noqa: E731
            subsets["n<=100"] = [sm for sm in data if nn(sm) <= 100]
            subsets["n>100"] = [sm for sm in data if nn(sm) > 100]
        for sub, dd in subsets.items():
            key = f"{fam}/{sub}"
            heur, rnd, info = heuristic_metrics(dd)
            row = {"info": {**info, "instances": len(dd)}, "nearest_route": heur, "random": rnd}
            for mname in ("hgnn", "homgnn"):
                p = os.path.join(a.models, f"{fam}_{mname}.pt")
                m = load_model(p, dev).to(dev)
                r = evaluate(m, dd, dev, 0.5)
                row[mname] = {k: float(v) for k, v in r.items()}
            res[key] = row
            print(key, json.dumps({k: (v if k == "info" else {kk: round(vv, 3) for kk, vv in v.items()})
                                   for k, v in row.items()}), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
