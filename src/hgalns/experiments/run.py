"""Experiment runner (Final Methodology Sections 10-11).

* one JSON line per (instance, algorithm, seed, budget) in the results file;
* resumable: completed runs are skipped on restart;
* every timed run is single-threaded (thread env vars + torch.set_num_threads(1));
* the HGNN guide is built inside the timed budget;
* every returned solution is verified by the independent checker.

Usage:
  python -m hgalns.experiments.run --config configs/core.yaml --workers 15 --out results/core.jsonl
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import json
import platform
import random
import socket
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# --------------------------------------------------------------------------- variants
# guide = None (classical) or dict(model=<key>, K=, lam=, prune=, prior=, gnn_destroy=)
VARIANTS = {
    "B0_regret2": {"construct_only": True},
    "B1_alns": {},
    "B2_alns_uniform": {"selection": "uniform"},
    "P_hgnn": {"guide": {"model": "hgnn", "K": 5, "lam": 0.5, "prune": True, "prior": True}, "gnn_destroy": True},
    "A_INT1_prune": {"guide": {"model": "hgnn", "K": 5, "lam": 0.0, "prune": True, "prior": False}},
    "A_INT2_prior": {"guide": {"model": "hgnn", "K": 5, "lam": 0.5, "prune": False, "prior": True}},
    "A_INT3_noGNNdestroy": {"guide": {"model": "hgnn", "K": 5, "lam": 0.5, "prune": True, "prior": True}},
    "B5_homgnn": {"guide": {"model": "homgnn", "K": 5, "lam": 0.5, "prune": True, "prior": True}, "gnn_destroy": True},
    "A_ALNS3_rrt": {"acceptance": "rrt"},
    # follow-up diagnostics (iteration-clocked blocks only)
    "REF_long": {"iter_mult": 4, "save_ref": True},          # classical ALNS, 4x iterations -> reference solution
    "O_oracle": {"guide": {"model": "oracle", "K": 5, "lam": 0.5, "prune": True, "prior": True}, "gnn_destroy": True},
    "A_ALNS4_itersa": {"acceptance": "sa_iter"},
    # pilot for the revised methodology: perfect neighbourhood selection among m random moves.
    # S_* = same number of iterations as classical ALNS (m times the work); W_* = same work (iterations / m).
    "S_best2": {"best_of": 2}, "S_best4": {"best_of": 4}, "S_best8": {"best_of": 8},
    "W_best2": {"best_of": 2, "iter_mult": 0.5}, "W_best4": {"best_of": 4, "iter_mult": 0.25},
    "W_best8": {"best_of": 8, "iter_mult": 0.125},
    # classical upgrade of the base: set partitioning over the route pool at the end of the run
    "B1_sp": {"sp": True},
    # pilot: who supplies the co-routing information? the network's co-routing head / the run's own best solution
    "C_copred": {"guide": {"model": "copred", "K": 5, "lam": 0.5, "prune": True, "prior": True}, "gnn_destroy": True},
    "C_self": {"guide": {"model": "self", "K": 5, "lam": 0.5, "prune": True, "prior": True}, "gnn_destroy": True},
    # revised method: co-routing model trained on long-run teachers, used once per instance, prior only (no pruning)
    "C2_co": {"guide": {"model": "copred", "key": "co_hgnn", "K": 999, "lam": 1.0, "prune": False, "prior": True}},
    "C2_sp": {"guide": {"model": "copred", "key": "co_hgnn", "K": 999, "lam": 0.2, "prune": False, "prior": True}, "sp": True},
    # pilot: soft consensus of several long runs as guide; static relatedness as guide (no learning)
    "O_soft": {"guide": {"model": "osoft", "K": 5, "lam": 0.5, "prune": True, "prior": True}, "gnn_destroy": True},
    "C_shaw": {"guide": {"model": "shaw", "K": 5, "lam": 0.5, "prune": True, "prior": True}, "gnn_destroy": True},
    # pilot: how good must the teacher be? oracle guides built from weaker references (other seed: +100)
    "REF_x1": {"iter_mult": 1, "save_ref": True, "ref_tag": "x1", "seed_add": 100},
    "REF_x06": {"iter_mult": 0.6, "save_ref": True, "ref_tag": "x06", "seed_add": 100},
    "O_ref1": {"guide": {"model": "oracle", "K": 5, "lam": 0.5, "prune": True, "prior": True, "ref_tag": "x1"}, "gnn_destroy": True},
    "O_ref06": {"guide": {"model": "oracle", "K": 5, "lam": 0.5, "prune": True, "prior": True, "ref_tag": "x06"}, "gnn_destroy": True},
}

_MODELS = {}


def _model(path):
    if path not in _MODELS:
        import torch
        torch.set_num_threads(1)
        from hgalns.learn.guide import load_model
        _MODELS[path] = load_model(path)
    return _MODELS[path]


def _bks_tables():
    out = {}
    p = os.path.join(REPO, "data", "bks", "lilim_bks.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            out[r["instance"].lower()] = (int(r["vehicles"]), float(r["distance"]))
    p = os.path.join(REPO, "data", "bks", "malheiros2021_mdhdarp_bks.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            out[r["instance"].lower()] = (None, float(r["bks"]))
    return out


def run_one(job):
    import sys
    sys.path.insert(0, os.path.join(REPO, "src"))
    from hgalns.io import load_instance
    from hgalns.alns import ALNSConfig, run_alns
    from hgalns.checker import check_solution
    path, algo, seed, budget, models, tune = job[:6]
    opts = job[6] if len(job) > 6 else {}
    name = os.path.splitext(os.path.basename(path))[0]
    inst = load_instance(path)
    base_algo = algo.split("@")[0]               # "P_hgnn@K0.5_lam1" -> variant P_hgnn + overrides
    v = dict(VARIANTS[base_algo])
    over = dict(tune.get(algo, {}))
    if "by_family" in over:                      # tuned values may differ per benchmark family
        famkey = "lilim" if "lilim" in inst.family else "hdarp"
        over = dict(over["by_family"].get(famkey, {}))
    if "guide" in over and v.get("guide"):
        g = dict(v["guide"]); g.update(over.pop("guide")); v["guide"] = g
    v.update(over)
    cfg = ALNSConfig(budget=budget, seed=seed + int(v.get("seed_add", 0)))
    if v.get("construct_only"):
        cfg.max_iters = 0
    for key in ("selection", "acceptance", "noise", "q_max", "xi", "vehicle_stage_frac"):
        if key in v:
            setattr(cfg, key, v[key])
    cfg.gnn_destroy = bool(v.get("gnn_destroy", False))
    cfg.best_of = int(v.get("best_of", 1))
    cfg.sp = bool(v.get("sp", False))
    if "sp_frac" in v:
        cfg.sp_frac = float(v["sp_frac"])
    if opts.get("max_iters"):                    # iteration-clocked run (equal iterations for all algorithms)
        cfg.max_iters = int(round(opts["max_iters"] * v.get("iter_mult", 1)))
        cfg.iter_clock = True
    cfg.diag = bool(opts.get("diag", False))
    ref_tag = v.get("ref_tag") or (v.get("guide") or {}).get("ref_tag") or ""
    ref_path = os.path.join(REPO, "results", "refs" + ("_" + ref_tag if ref_tag else ""), (inst.family + "_" if inst.family.startswith("gen-") else "") + name + ".json")
    gspec = v.get("guide")
    if gspec and gspec["model"] == "oracle":
        from hgalns.learn.guide import OracleGuide
        ref_routes = json.load(open(ref_path))["routes"]

        def factory(inst_, ev_, g=gspec):
            return OracleGuide(ref_routes, inst_, ev_, K=g.get("K", 5), lam=g.get("lam", 0.5),
                               T_pi=g.get("T_pi", 1.0), prune=g.get("prune", True), prior=g.get("prior", True))
        cfg.guide_factory = factory
    elif gspec and gspec["model"] in ("osoft", "shaw"):
        from hgalns.learn.guide import CoGuide, consensus_co, static_relatedness_co
        import glob as _glob

        def factory(inst_, ev_, g=gspec):
            if g["model"] == "osoft":      # soft consensus of all stored reference runs of this instance
                refs = [json.load(open(f))["routes"] for f in sorted(_glob.glob(ref_path[:-5] + ".s*.json"))]
                co = consensus_co(refs, inst_.n)
            else:                          # no learning, no search: static relatedness of the requests
                co = static_relatedness_co(inst_)
            return CoGuide(co, inst_, ev_, K=g.get("K", 5), lam=g.get("lam", 0.5), T_pi=g.get("T_pi", 1.0),
                           prune=g.get("prune", True), prior=g.get("prior", True))
        cfg.guide_factory = factory
    elif gspec and gspec["model"] == "self":
        from hgalns.learn.guide import SelfGuide

        def factory(inst_, ev_, g=gspec):
            return SelfGuide(inst_, ev_, K=g.get("K", 5), lam=g.get("lam", 0.5), T_pi=g.get("T_pi", 1.0),
                             prune=g.get("prune", True), prior=g.get("prior", True))
        cfg.guide_factory = factory
    elif gspec and gspec["model"] == "copred":
        fam = "lilim" if "lilim" in inst.family else "hdarp"
        model = _model(models[gspec.get("key", "hgnn")][fam])     # which co-routing model: old head or a teacher-trained one
        from hgalns.learn.guide import HGNNGuide, CoGuide

        def factory(inst_, ev_, model=model, g=gspec):
            co = HGNNGuide(model, inst_, ev_, K=999, lam=0.0).co      # the network's static co-routing head (set-up is timed)
            return CoGuide(co, inst_, ev_, K=g.get("K", 5), lam=g.get("lam", 0.5), T_pi=g.get("T_pi", 1.0),
                           prune=g.get("prune", True), prior=g.get("prior", True))
        cfg.guide_factory = factory
    elif gspec:
        fam = "lilim" if "lilim" in inst.family else "hdarp"
        mpath = models[gspec["model"]][fam]
        model = _model(mpath)
        from hgalns.learn.guide import HGNNGuide

        def factory(inst_, ev_, model=model, g=gspec):
            return HGNNGuide(model, inst_, ev_, K=g.get("K", 5), lam=g.get("lam", 0.5),
                             T_pi=g.get("T_pi", 1.0), prune=g.get("prune", True), prior=g.get("prior", True))
        cfg.guide_factory = factory
    t0 = time.time()
    res = run_alns(inst, cfg)
    sol = res["solution"]
    ok, errs, met = check_solution(inst, sol.routes, sol.schedules(), sol.bank)
    if v.get("save_ref"):
        os.makedirs(os.path.dirname(ref_path), exist_ok=True)
        new = {"routes": [list(map(int, r)) for r in sol.routes], "vehicles": res["vehicles"],
               "distance": res["distance"], "unserved": res["unserved"], "iterations": res["iterations"], "seed": seed}
        keep = True
        if os.path.exists(ref_path):
            old = json.load(open(ref_path))
            lex = inst.objective == "lexi"
            keep = (new["unserved"], new["vehicles"] if lex else 0, new["distance"]) < \
                   (old["unserved"], old["vehicles"] if lex else 0, old["distance"])
        if keep:
            with open(ref_path + f".tmp{os.getpid()}", "w") as f:
                json.dump(new, f)
            os.replace(ref_path + f".tmp{os.getpid()}", ref_path)
        with open(ref_path[:-5] + f".s{seed}.json", "w") as f:      # every reference run is kept (for consensus guides)
            json.dump(new, f)
    bks = _bks_tables().get(inst.name.lower().replace("hetiuy", ""))
    gap = None if bks is None else 100.0 * (res["distance"] - bks[1]) / bks[1]
    return {
        "instance": name, "family": inst.family, "n": inst.n, "algo": algo, "seed": seed,
        "budget": budget, "distance": res["distance"], "vehicles": res["vehicles"],
        "unserved": res["unserved"], "objective": res["objective"],
        "bks_vehicles": None if bks is None else bks[0], "bks_distance": None if bks is None else bks[1],
        "gap_pct": gap, "iterations": res["iterations"], "elapsed": res["elapsed"],
        "t_guide_setup": res.get("t_guide_setup", 0.0), "guide_calls": res.get("guide_calls", 0),
        "insert_evals": res["insert_evals"], "checker_ok": ok, "checker_errors": errs[:5],
        "trajectory": res["trajectory"], "destroy_weights": res["destroy_weights"],
        "repair_weights": res.get("repair_weights"), "iter_clock": cfg.iter_clock,
        "max_iters": cfg.max_iters if cfg.iter_clock else None, "real_elapsed": res.get("real_elapsed"),
        "diag": res.get("diag"),
        "sp": res.get("sp"),
        "wall": time.time() - t0, "host": socket.gethostname(),
    }


def _iname(family, name):
    # generated sets reuse file names across families (valopt_00000 exists in both)
    return (family + "/" + name) if str(family).startswith("gen-") else name


def key_of(r):
    return (_iname(r.get("family", ""), r["instance"]), r["algo"], int(r["seed"]), round(float(r["budget"]), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N jobs (pilot)")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    import sys
    sys.path.insert(0, os.path.join(REPO, "src"))
    from hgalns.io import list_instances, load_instance
    cfg = yaml.safe_load(open(a.config))
    jobs = []
    for block in cfg["blocks"]:
        iters = json.load(open(os.path.join(REPO, block["iter_file"]))) if block.get("iter_file") else None
        paths = []
        for spec in block["instances"]:
            fam, size = spec.get("family"), spec.get("size")
            ps = list_instances(fam, size)
            if spec.get("names"):
                ps = [p for p in ps if os.path.splitext(os.path.basename(p))[0] in spec["names"]]
            paths += ps
        for p in paths:
            n = load_instance(p).n
            for b in block["budgets"]:
                budget = float(b[:-1]) * n if isinstance(b, str) and b.endswith("n") else float(b)
                for algo, nseeds in block["algorithms"].items():
                    for s in range(nseeds):
                        opts = {"diag": bool(block.get("diag", False)), "fam": fam}
                        if iters is not None:
                            it_ = iters[os.path.splitext(os.path.basename(p))[0]]
                            opts["max_iters"] = it_[algo] if isinstance(it_, dict) else it_   # per-configuration budgets
                        jobs.append((p, algo, s, round(budget, 3), cfg.get("models", {}), cfg.get("tune", {}), opts))
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(key_of(json.loads(line)))
            except Exception:
                pass
    todo = [j for j in jobs if (_iname(j[6].get("fam", ""), os.path.splitext(os.path.basename(j[0]))[0]), j[1], j[2], j[3]) not in done]
    random.Random(12345).shuffle(todo)          # interleave algorithms -> shared conditions
    if a.limit:
        todo = todo[: a.limit]
    est = sum(j[3] for j in todo)
    print(f"{len(jobs)} jobs in config, {len(done)} done, {len(todo)} to run; "
          f"~{est / 3600:.1f} CPU-h -> ~{est / 3600 / max(1, a.workers):.1f} h wall with {a.workers} workers", flush=True)
    if a.dry:
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    meta = {"_meta": True, "time": time.ctime(), "host": socket.gethostname(), "platform": platform.platform(),
            "cpu": platform.processor(), "workers": a.workers, "config": cfg}
    try:
        meta["git"] = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    with open(a.out + ".meta.jsonl", "a") as f:
        f.write(json.dumps(meta) + "\n")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in todo}
        for i, f in enumerate(as_completed(futs)):
            try:
                r = f.result()
            except Exception as e:  # keep going; log the failure
                j = futs[f]
                print(f"FAILED {j[0]} {j[1]} {j[2]}: {e!r}", flush=True)
                continue
            with open(a.out, "a") as fo:
                fo.write(json.dumps(r) + "\n")
            g = "" if r["gap_pct"] is None else f"gap={r['gap_pct']:6.2f}%"
            print(f"[{i + 1}/{len(todo)} {time.time() - t0:6.0f}s] {r['instance']:16s} {r['algo']:20s} s{r['seed']} "
                  f"veh={r['vehicles']} dist={r['distance']:.2f} {g} it={r['iterations']} ok={r['checker_ok']}", flush=True)


if __name__ == "__main__":
    main()
