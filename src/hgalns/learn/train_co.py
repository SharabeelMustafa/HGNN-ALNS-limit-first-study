"""Revised method, training: predict the co-routing structure of LONG classical-ALNS runs (strong teacher).

Target for a request pair (r, r'): share of the teacher runs of the instance in which r and r' share a route
(soft label in [0, 1]). The prediction is made once per instance from the static graph (no search state), on the
request-request edges of the k-nearest graph. Loss: binary cross-entropy with soft targets.

Differences to the first method (learn/train.py): the teacher is several times stronger than the search that will
use the advice; the target does not depend on a search state; there is no route head.

Checkpoints: <out>.ckpt after every epoch (resume with the same command); best model (lowest validation loss) in <out>.
An existing <out> of a finished run is never overwritten unless --overwrite is given.

Usage: python -m hgalns.learn.train_co --family hdarp --teachers m8 --out models/v2/hdarp_co_hgnn.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .model import HGNN
from .batch import collate

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def teacher_co(files, n):
    co = np.zeros((n, n), dtype=np.float32)
    k = 0
    for f in files:
        t = json.load(open(f))
        if t["unserved"]:
            continue
        lab = np.full(n, -1, dtype=np.int64)
        for v, route in enumerate(t["routes"]):
            for x in route:
                if x < n:
                    lab[x] = v
        co += (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0)
        k += 1
    return (co / k) if k else None


def load_split(family, split, teachers, limit=0):
    """Samples = static graph + co edges of the old datasets, with new targets from the teacher runs."""
    out = []
    tdir = os.path.join(REPO, "data", "teachers", family, teachers)
    for p in sorted(glob.glob(os.path.join(REPO, "data", "datasets", family, split, "*.pkl"))):
        name = os.path.basename(p)[:-4]
        tf = sorted(glob.glob(os.path.join(tdir, name + ".s*.json")))
        if not tf:
            continue
        sm = pickle.load(open(p, "rb"))
        g = sm["graph"]
        n = g["n"]
        co = teacher_co(tf, n)
        ce = sm.get("co_edges")
        if co is None or ce is None or len(ce) == 0:
            continue
        ei = g["edge_index"]
        y = co[ei[0, ce] % n, ei[1, ce] % n].astype(np.float32)
        out.append({"graph": g, "co_edges": ce, "co_y": y, "states": [], "name": name})
        if limit and len(out) >= limit:
            break
    return out


def run_epoch(model, data, device, bs, opt=None, sched=None, rng=None):
    idx = list(range(len(data)))
    if opt is not None:
        rng.shuffle(idx)
    tot = cnt = 0.0
    P, Y = [], []
    model.train(opt is not None)
    for i in range(0, len(idx), bs):
        b = collate([(data[j], []) for j in idx[i:i + bs]], device)
        with torch.set_grad_enabled(opt is not None):
            h, e = model.encode(b["x_stop"], b["x_class"], b["stop_idx"], b["class_idx"], b["ntype"], b["ei"], b["rel"], b["eattr"])
            u = model.co_logits(h, e, b["co_src"], b["co_dst"], b["co_eidx"])
            loss = F.binary_cross_entropy_with_logits(u, b["co_y"])
        if opt is not None:
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        else:
            P.append(torch.sigmoid(u).cpu().numpy()); Y.append(b["co_y"].cpu().numpy())
        tot += float(loss) * u.numel(); cnt += u.numel()
    res = {"loss": tot / max(cnt, 1)}
    if P:
        p, y = np.concatenate(P), np.concatenate(Y)
        yb = y >= 0.5
        order = np.argsort(-p)
        prec = np.cumsum(yb[order]) / (np.arange(len(p)) + 1)
        res["ap"] = float((prec * yb[order]).sum() / max(1, yb.sum()))
        res["brier"] = float(((p - y) ** 2).mean())
        res["base_brier"] = float(((y.mean() - y) ** 2).mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--teachers", default="m8", help="sub-directory of data/teachers/<family>/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--hetero", type=int, default=1)
    ap.add_argument("--edge-feats", type=int, default=1)
    ap.add_argument("--init", default="", help="checkpoint of the first method to start the encoder from (optional)")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = a.out if os.path.isabs(a.out) else os.path.join(REPO, a.out)
    ck_path = out + ".ckpt"
    if os.path.exists(out) and not os.path.exists(ck_path) and not a.overwrite:
        print("finished model exists, not overwriting:", out); return 0
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    tr = load_split(a.family, "train", a.teachers, a.limit)
    va = load_split(a.family, "valml", a.teachers)
    print(f"train {len(tr)} instances, {sum(len(s['co_y']) for s in tr)} pairs; val {len(va)} instances; "
          f"mean target {np.mean(np.concatenate([s['co_y'] for s in tr])):.3f}", flush=True)
    if not tr or not va:
        print("no data"); return 1
    hp = dict(d=a.d, heads=a.heads, layers=a.layers, hetero=bool(a.hetero), act="gelu", norm="layer", edge_feats=bool(a.edge_feats),
              route_state=True)
    model = HGNN(**hp).to(a.device)
    if a.init:
        sd = torch.load(os.path.join(REPO, a.init), map_location=a.device, weights_only=False)["state_dict"]
        model.load_state_dict(sd, strict=False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * math.ceil(len(tr) / a.bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    ep0, best, bad, hist = 0, float("inf"), 0, []
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(ck_path):                       # resume after a pause, crash or reboot
        ck = torch.load(ck_path, map_location=a.device, weights_only=False)
        model.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        ep0, best, bad, hist = ck["epoch"] + 1, ck["best"], ck["bad"], ck["hist"]
        print(f"resumed at epoch {ep0}", flush=True)
    for ep in range(ep0, a.epochs):
        t0 = time.time()
        trm = run_epoch(model, tr, a.device, a.bs, opt, sched, rng)
        vm = run_epoch(model, va, a.device, a.bs)
        hist.append({"epoch": ep, "train_loss": trm["loss"], **{"val_" + k: v for k, v in vm.items()}, "time": time.time() - t0})
        print(json.dumps(hist[-1]), flush=True)
        if vm["loss"] < best - 1e-5:
            best, bad = vm["loss"], 0
            torch.save({"state_dict": model.state_dict(), "hparams": hp, "val": vm, "epoch": ep, "args": vars(a)}, out + ".tmp")
            os.replace(out + ".tmp", out)
        else:
            bad += 1
        torch.save({"state_dict": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "best": best, "bad": bad, "hist": hist}, ck_path + ".tmp")
        os.replace(ck_path + ".tmp", ck_path)
        if bad >= a.patience:
            break
    json.dump(hist, open(out + ".history.json", "w"), indent=1)
    if os.path.exists(ck_path):
        os.remove(ck_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
