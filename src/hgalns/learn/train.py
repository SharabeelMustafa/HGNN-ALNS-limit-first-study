"""HGNN training (Final Methodology Section 8.2-8.3).

loss = listwise route CE (8.1) + mu * focal co-routing loss (8.2) (+ AdamW weight decay)
Early stopping on validation NDCG@3 / pruning recall@K.

Usage: python -m hgalns.learn.train --family hdarp --epochs 30 --out models/hdarp_hgnn.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import random
import time

import numpy as np
import torch
from torch_geometric.utils import softmax as seg_softmax

from .model import HGNN
from .batch import collate


def load_dir(d):
    out = []
    for p in sorted(glob.glob(os.path.join(d, "*.pkl"))):
        with open(p, "rb") as f:
            out.append(pickle.load(f))
    return out


def forward_losses(model, b, mu=0.5, gamma=2.0):
    h, e = model.encode(b["x_stop"], b["x_class"], b["stop_idx"], b["class_idx"], b["ntype"],
                        b["ei"], b["rel"], b["eattr"])
    g = model.route_embed(h, b["cand_T"], b["cand_xi"], b["stop_node"], b["stop_cand"], b["n_cand"])
    s = model.pair_scores(h, g, b["pair_P"], b["pair_D"], b["pair_c"], b["chi"])
    grp = b["pair_g"]
    # log-softmax per group: s - logsumexp_group(s)
    p = seg_softmax(s, grp, num_nodes=b["n_groups"])
    has_t = torch.zeros(b["n_groups"], device=s.device).index_add_(0, grp, b["pair_y"]) > 0
    tgt_logp = torch.zeros(b["n_groups"], device=s.device).index_add_(0, grp, b["pair_y"] * torch.log(p + 1e-12))
    l_route = -(tgt_logp[has_t]).mean() if has_t.any() else s.sum() * 0
    l_co = s.sum() * 0
    if mu > 0 and b["co_eidx"].numel():
        u = model.co_logits(h, e, b["co_src"], b["co_dst"], b["co_eidx"])
        pr = torch.sigmoid(u).clamp(1e-6, 1 - 1e-6)
        y = b["co_y"]
        l_co = -(y * (1 - pr) ** gamma * torch.log(pr) + (1 - y) * pr ** gamma * torch.log(1 - pr)).mean()
    return l_route + mu * l_co, l_route, l_co, (s, grp, b["pair_y"], has_t, b["n_groups"])


@torch.no_grad()
def rank_metrics(s, grp, y, has_t, G, K=5):
    s, grp, y = s.cpu().numpy(), grp.cpu().numpy(), y.cpu().numpy()
    order = np.lexsort((-s, grp))
    top1 = rec = ndcg = cnt = 0
    start = 0
    gs = grp[order]
    ys = y[order]
    bounds = np.flatnonzero(np.diff(gs)) + 1
    for seg in np.split(np.arange(len(order)), bounds):
        yy = ys[seg]
        if yy.sum() == 0:
            continue
        rank = int(np.flatnonzero(yy > 0)[0]) + 1
        cnt += 1
        top1 += rank == 1
        rec += rank <= K
        ndcg += (1.0 / math.log2(rank + 1)) if rank <= 3 else 0.0
    return {"top1": top1 / max(cnt, 1), f"recall@{K}": rec / max(cnt, 1), "ndcg@3": ndcg / max(cnt, 1), "groups": cnt}


def batches(data, bs, states_per, rng):
    idx = list(range(len(data)))
    rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        items = []
        for j in idx[i:i + bs]:
            sm = data[j]
            if not sm["states"]:
                continue
            k = min(states_per, len(sm["states"]))
            sel = rng.sample(range(len(sm["states"])), k)
            items.append((sm, [sm["states"][t] for t in sel]))
        if items:
            yield items


def evaluate(model, data, device, mu, K=5):
    model.eval()
    agg = {"loss": 0.0, "n": 0}
    allm = []
    with torch.no_grad():
        for i in range(0, len(data), 8):
            items = [(sm, sm["states"]) for sm in data[i:i + 8] if sm["states"]]
            if not items:
                continue
            b = collate(items, device)
            loss, lr, lc, (s, grp, y, ht, G) = forward_losses(model, b, mu)
            agg["loss"] += float(loss) * len(items)
            agg["n"] += len(items)
            allm.append(rank_metrics(s, grp, y, ht, G, K))
    tot = sum(m["groups"] for m in allm) or 1
    res = {k: sum(m[k] * m["groups"] for m in allm) / tot for k in ("top1", f"recall@{K}", "ndcg@3")}
    res["loss"] = agg["loss"] / max(agg["n"], 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--data", default="data/datasets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--states-per", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--hetero", type=int, default=1)
    ap.add_argument("--act", default="gelu")
    ap.add_argument("--norm", default="layer")
    ap.add_argument("--edge-feats", type=int, default=1)
    ap.add_argument("--route-state", type=int, default=1)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    tr = load_dir(os.path.join(a.data, a.family, "train"))
    va = load_dir(os.path.join(a.data, a.family, "valml"))
    print(f"train {len(tr)} instances / {sum(len(s['states']) for s in tr)} states; val {len(va)}", flush=True)
    hp = dict(d=a.d, heads=a.heads, layers=a.layers, hetero=bool(a.hetero), act=a.act, norm=a.norm,
              edge_feats=bool(a.edge_feats), route_state=bool(a.route_state))
    model = HGNN(**hp).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    best, bad, hist = -1.0, 0, []
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        tl = tn = 0.0
        for items in batches(tr, a.bs, a.states_per, rng):
            b = collate(items, a.device)
            loss, lr_, lc_, _ = forward_losses(model, b, a.mu)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tl += float(loss) * len(items)
            tn += len(items)
        vm = evaluate(model, va, a.device, a.mu)
        crit = vm["ndcg@3"] + vm["recall@5"]
        hist.append({"epoch": ep, "train_loss": tl / max(tn, 1), **vm, "time": time.time() - t0})
        print(json.dumps(hist[-1]), flush=True)
        if crit > best:
            best, bad = crit, 0
            torch.save({"state_dict": model.state_dict(), "hparams": hp, "val": vm, "epoch": ep}, a.out)
        else:
            bad += 1
            if bad >= a.patience:
                break
    with open(a.out + ".history.json", "w") as f:
        json.dump(hist, f, indent=1)


if __name__ == "__main__":
    main()
