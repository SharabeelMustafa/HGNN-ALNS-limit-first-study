"""Offline gate for the revised method: how well can the co-routing of LONG runs be predicted?

Target: do two requests share a route in a 4x-iteration reference run (held-out seed)?  Instances: the 50 tuning
instances (references in results/refs/, made by the pilot experiments).  Predictors compared, per instance (AUC and
average precision over all request pairs, then averaged over instances):

  ceiling   other long runs      consensus of the OTHER 4x reference runs of the same instance
  search    1x run / 0.6x run    co-routing in one shorter run (costs search time, no learning)
            short-run elites     consensus of the 3 short runs that made the old training labels
  free      static relatedness   Shaw-type relatedness from instance data (no learning, no search)
  learned   old HGNN head        co-routing head of the current model (trained on short-run elites)
            new models           any --model name=path given (trained on long-run teachers)

Usage: python scripts/eval_co.py [--model hgnn_v2=models/v2/lilim_co.pt:lilim ...] --out results/co_eval/base.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
from scipy.stats import rankdata

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))


def co_of(routes, n):
    lab = np.full(n, -1, dtype=np.int64)
    for k, r in enumerate(routes):
        for v in r:
            if v < n:
                lab[v] = k
    c = (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0)
    return c.astype(np.float64)


def auc_ap(score, y):
    y = y.astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return np.nan, np.nan
    r = rankdata(score)
    auc = (r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
    order = np.argsort(-score, kind="stable")
    ys = y[order]
    prec = np.cumsum(ys) / (np.arange(len(ys)) + 1)
    return float(auc), float((prec * ys).sum() / npos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="name=path:family (family = lilim or hdarp)")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "co_eval", "base.json"))
    a = ap.parse_args()
    import torch
    torch.set_num_threads(1)
    from hgalns.solution import Evaluator
    from hgalns.learn.guide import HGNNGuide, load_model, static_relatedness, static_relatedness_co
    extra = {}
    for spec in a.model:
        name, rest = spec.split("=", 1)
        path, fam = rest.rsplit(":", 1)
        extra.setdefault(fam, {})[name] = load_model(os.path.join(REPO, path))
    res = {}
    for fam in ("lilim", "hdarp"):
        old = load_model(os.path.join(REPO, "models", f"{fam}_hgnn.pt"))
        acc = {}
        ninst = 0
        for lab in sorted(glob.glob(os.path.join(REPO, "data", "labels", fam, "valopt_*.pkl"))):
            name = os.path.basename(lab)[:-4]
            refs = sorted(glob.glob(os.path.join(REPO, "results", "refs", f"gen-{fam}_{name}.s*.json")))
            if len(refs) < 2:
                continue
            rec = pickle.load(open(lab, "rb"))
            inst = rec["instance"]
            n = inst.n
            long_ = [json.load(open(f)) for f in refs]
            if any(x["unserved"] for x in long_):
                continue
            ninst += 1
            L = [co_of(x["routes"], n) for x in long_]
            iu = np.triu_indices(n, 1)
            pred = {}
            for tag, d in (("1x run (other seed)", "refs_x1"), ("0.6x run (other seed)", "refs_x06")):
                p = os.path.join(REPO, "results", d, f"gen-{fam}_{name}.json")
                if os.path.exists(p):
                    pred[tag] = co_of(json.load(open(p))["routes"], n)
            pred["short-run elites (old labels)"] = np.mean([co_of(e["routes"], n) for e in rec["elites"]], axis=0)
            R = static_relatedness(inst)
            pred["static relatedness (free)"] = -R
            pred["static relatedness, 16 nearest"] = static_relatedness_co(inst)
            ev = Evaluator(inst)
            pred["old HGNN co-routing head"] = HGNNGuide(old, inst, ev, K=999, lam=0.0).co
            for nm, mod in extra.get(fam, {}).items():
                pred[nm] = HGNNGuide(mod, inst, ev, K=999, lam=0.0).co
            for j in range(len(L)):                                   # held-out long run j
                y = L[j][iu]
                others = [L[t] for t in range(len(L)) if t != j]
                allp = dict(pred)
                allp["other long runs (ceiling)"] = np.mean(others, axis=0)
                for tag, P in allp.items():
                    au, apv = auc_ap(np.asarray(P, dtype=np.float64)[iu], y)
                    if not np.isnan(au):
                        acc.setdefault(tag, []).append((au, apv))
        res[fam] = {"instances": ninst, "base_rate": None,
                    "predictors": {t: {"auc": float(np.mean([x[0] for x in v])), "ap": float(np.mean([x[1] for x in v])), "n": len(v)}
                                   for t, v in acc.items()}}
        print(f"\n{fam}: {ninst} instances")
        print("| Predictor | AUC | Average precision |\n|---|---|---|")
        for t, v in sorted(res[fam]["predictors"].items(), key=lambda kv: -kv[1]["ap"]):
            print(f"| {t} | {v['auc']:.3f} | {v['ap']:.3f} |")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
