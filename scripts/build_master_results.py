"""Master results table: one row per run of every experiment (original result files are only read, never changed).

Output (in ../Research_Package/02_results/):
  master_results.csv        one row per run
  results_summary.json      aggregates per experiment x benchmark x algorithm
  EXPERIMENT_CATALOG.md     one section per experiment: objective, data, configuration, runs, seeds, files, status, outcome
"""
import collections
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_main import REPO, bks_tables, score_of        # noqa: E402

OUT = os.path.abspath(os.path.join(REPO, "..", "Research_Package", "02_results"))
SUBSET = set()
try:
    sub = json.load(open(os.path.join(REPO, "results", "followup_subset.json")))
    SUBSET = set(sub if isinstance(sub, list) else [x for v in sub.values() for x in v])
except Exception:
    pass

LABEL = {"B0_regret2": "Regret-2 construction only", "B1_alns": "Classical ALNS", "B1_sp": "Classical ALNS + set partitioning",
         "P_hgnn": "First method: HGNN route prior + pruning + learned removal", "B5_homgnn": "First method with homogeneous GNN",
         "A_INT1_prune": "First method, pruning only", "A_INT2_prior": "First method, prior only",
         "A_INT3_noGNNdestroy": "First method without learned removal", "REF_long": "Classical ALNS, 4x iterations (reference run)",
         "REF_x1": "Classical ALNS, 1x iterations, other seed (reference run)", "REF_x06": "Classical ALNS, 0.6x iterations (reference run)",
         "O_oracle": "Oracle from 4x reference, first-method setting", "O_ref1": "Oracle from 1x reference", "O_ref06": "Oracle from 0.6x reference",
         "O_oracle@prior": "Oracle (4x), prior only, lambda 0.2", "O_oracle@prior02": "Oracle (4x), prior only, lambda 0.2",
         "O_oracle@prior1": "Oracle (4x), prior only, lambda 1.0", "O_oracle@destroy": "Oracle (4x), removal operator only",
         "O_oracle@prune": "Oracle (4x), pruning only", "O_soft": "Mean of three 4x runs as guide, first-method setting",
         "O_soft@prior1": "Mean of three 4x runs, prior only, lambda 1.0", "C_shaw": "Static relatedness as guide, first-method setting",
         "C_shaw@prior1": "Static relatedness, prior only, lambda 1.0", "C_copred": "Old co-routing head as guide", "C_self": "Own best solution as guide",
         "C2_co@hgnn_l02": "Teacher-trained HGNN, soft hint, lambda 0.2", "C2_co@hgnn_l1": "Teacher-trained HGNN, soft hint, lambda 1.0",
         "C2_co@hom_l1": "Teacher-trained homogeneous GNN, soft hint, lambda 1.0", "C2_co@mlp_l1": "Teacher-trained model without message passing, lambda 1.0",
         "C2_co@old_l1": "Old co-routing head, soft hint, lambda 1.0", "C2_sp@hgnn_l02": "Teacher-trained HGNN soft hint (lambda 0.2) + set partitioning",
         "S_best2": "Best of 2 moves, same iterations", "S_best4": "Best of 4 moves, same iterations", "W_best2": "Best of 2 moves, same work",
         "W_best4": "Best of 4 moves, same work", "W_best8": "Best of 8 moves, same work"}

EXP = collections.OrderedDict([
    ("E01", ("Pilot of the pipeline", "Does the whole pipeline run end to end?", "INFO: led to share-based pruning and more H-DARP training data")),
    ("E02", ("Initial experiment (Phase 3)", "First comparison on 10 instances", "INFO: no significant difference (p = 0.35)")),
    ("E03", ("Tuning", "Which ALNS and guidance parameters are best on generated tuning instances?", "NEGATIVE for guidance: every guided setting worse than tuned classical ALNS")),
    ("E04", ("Main experiment, equal time", "RQ1/RQ2: first method beats classical ALNS at 0.5 s per request", "NEGATIVE")),
    ("E04-RQ3", ("Main experiment, fixed deadlines", "RQ3: feasibility and quality at 10 s and 20 s", "All feasible; first method worse")),
    ("E05", ("Equal iterations", "The loss is only the time overhead of the network", "NEGATIVE: still worse")),
    ("E06", ("Components of the first method", "Which component hurts?", "INFO: pruning")),
    ("E07", ("First oracle", "The mechanism helps when the advice is good", "POSITIVE")),
    ("E08", ("Pilot baseline", "Classical ALNS on the tuning instances (reference for E08-E14)", "INFO")),
    ("E08a", ("Limit of neighbourhood selection", "A perfect choice among m random moves leaves room for a learned selector", "NEGATIVE: at most 0.7%")),
    ("E08b", ("Other guides", "Old co-routing head / own best solution as guide", "NEGATIVE")),
    ("E08c", ("Teacher strength", "Advice helps only from a stronger teacher", "POSITIVE on H-DARP style")),
    ("E09a", ("More reference runs", "Data for the consensus guide", "INFO")),
    ("E09b", ("Use of the advice", "Prior / removal operator / pruning; consensus; static relatedness", "POSITIVE for prior only; others negative")),
    ("E10", ("Set partitioning, tuning instances", "Set partitioning improves classical ALNS", "POSITIVE")),
    ("E11", ("Per-instance configuration", "The best configuration is instance-specific", "NEGATIVE")),
    ("E14", ("Search with trained models", "Gate G4: teacher-trained model as soft hint beats classical ALNS", "NEGATIVE")),
    ("E15", ("Set partitioning, benchmarks", "Set partitioning improves classical ALNS on 140 instances", "POSITIVE, small")),
    ("E16", ("Oracle and trained model, 36 benchmark instances", "Confirmation of E09b and E14 on benchmarks", "Confirmed")),
    ("E17", ("Soft hint and set partitioning, all benchmarks", "2 x 2 ablation: hint on/off x set partitioning on/off", "see ABLATION_STUDIES.md")),
    ("E18a", ("Reference runs for all benchmarks", "Data for the oracle on all instances", "INFO")),
    ("E18b", ("Oracle on all benchmarks", "Oracle soft hint on 140 instances", "see ABLATION_STUDIES.md")),
])


def exp_of(fname, r):
    a, base = r["algo"], r["algo"].split("@")[0]
    if fname == "core.jsonl":
        return "E04-RQ3" if r["budget"] in (10, 20) else "E04"
    if fname.startswith("tune_"):
        return "E03"
    if fname == "initial.jsonl":
        return "E02"
    if fname.startswith("pilot") and fname not in ("pilot_headroom.jsonl", "pilot_isac.jsonl"):
        return "E01"
    if fname == "pilot_isac.jsonl":
        return "E11"
    if fname == "followup.jsonl":
        if a in ("B1_alns", "P_hgnn"):
            return "E05"
        if base.startswith("A_INT"):
            return "E06"
        if a == "REF_long":
            return "E07" if r["instance"] in SUBSET else "E18a"
        if a == "O_oracle":
            return "E07"
        if a == "B1_sp":
            return "E15"
        if a in ("O_oracle@prior02", "C2_co@hgnn_l02"):
            return "E16"
        if a == "O_oracle@prior1":
            return "E16" if r["instance"] in SUBSET else "E18b"
        if base == "C2_sp":
            return "E17"
    if fname == "pilot_headroom.jsonl":
        if a == "B1_alns":
            return "E08"
        if base[:2] in ("S_", "W_"):
            return "E08a"
        if a in ("P_hgnn", "C_copred", "C_self"):
            return "E08b"
        if a in ("REF_x1", "REF_x06", "O_ref1", "O_ref06", "O_oracle"):
            return "E08c"
        if a == "REF_long":
            return "E08c" if r["seed"] == 0 else "E09a"
        if a in ("O_soft", "C_shaw", "O_oracle@prior", "O_oracle@prior1", "O_oracle@destroy", "O_oracle@prune"):
            return "E09b"
        if a == "B1_sp":
            return "E10"
        if base == "C2_co" or a in ("O_soft@prior1", "C_shaw@prior1"):
            return "E14"
    return "other"


def bench_of(r):
    f = r.get("family", "")
    if f == "gen-lilim":
        return "Generated, Li & Lim style (tuning)"
    if f == "gen-hdarp":
        return "Generated, H-DARP style (tuning)"
    if "lilim" in f:
        return "Li & Lim 100" if r["n"] < 80 else "Li & Lim 200"
    return "H-DARP (Malheiros)"


COLS = ["experiment", "source_file", "benchmark", "instance", "n_requests", "algorithm", "algorithm_label", "seed", "clock", "budget_s", "max_iterations",
        "iterations", "run_time_s", "vehicles", "objective", "unserved", "feasible", "bks_vehicles", "bks_objective", "gap_pct", "score",
        "set_partitioning_improved", "set_partitioning_time_s", "derived"]


def main():
    os.makedirs(OUT, exist_ok=True)
    bks = bks_tables()
    rows = []
    files = [f for f in sorted(glob.glob(os.path.join(REPO, "results", "*.jsonl"))) if not f.endswith(".meta.jsonl")]
    for path in files:
        fname = os.path.basename(path)
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "algo" not in r or r.get("_meta"):
                continue
            b = bks.get(r["instance"].lower().replace("hetiuy", "")) if not str(r.get("family", "")).startswith("gen-") else None

            def rec(algo, veh, obj, derived, spi="", spt=""):
                gap = score = ""
                if b is not None and r["unserved"] == 0:
                    r2 = dict(r, bks_vehicles=b[0], bks_distance=b[1])
                    score = round(score_of(r2, veh, obj), 4); gap = round(100.0 * (obj - b[1]) / b[1], 4)
                return [exp_of(fname, r), fname, bench_of(r), r["instance"], r["n"], algo, LABEL.get(algo, LABEL.get(algo.split("@")[0], algo)), r["seed"],
                        "iteration" if r.get("iter_clock") else "time", r["budget"], r.get("max_iters") or "", r["iterations"],
                        round(r.get("real_elapsed") or r["elapsed"], 3), veh, round(obj, 4), r["unserved"], int(bool(r["checker_ok"]) and r["unserved"] == 0),
                        "" if b is None or b[0] is None else b[0], "" if b is None else b[1], gap, score, spi, spt, derived]
            sp = r.get("sp") or {}
            rows.append(rec(r["algo"], r["vehicles"], r["distance"], 0, "" if not sp else int(bool(sp.get("improved"))), "" if not sp else round(sp.get("time", 0.0), 3)))
            if r["algo"].startswith("C2_sp") and "before_distance" in sp:      # the same run before the set-partitioning step
                rows.append(rec(r["algo"].replace("C2_sp", "C2_co"), sp["before_vehicles"], sp["before_distance"], 1))
    with open(os.path.join(OUT, "master_results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(COLS); w.writerows(rows)
    # ---- summary
    agg = collections.defaultdict(list)
    for x in rows:
        agg[(x[0], x[2], x[5])].append(x)
    summ = []
    for (e, bn, al), v in sorted(agg.items()):
        per = collections.defaultdict(list)
        for x in v:
            per[x[3]].append(x)
        sc = [np.mean([y[20] for y in u if y[20] != ""]) for u in per.values() if any(y[20] != "" for y in u)]
        summ.append({"experiment": e, "benchmark": bn, "algorithm": al, "label": v[0][6], "instances": len(per), "runs": len(v),
                     "seeds": sorted({x[7] for x in v}), "clock": v[0][8], "feasible_share": float(np.mean([x[16] for x in v])),
                     "mean_objective": float(np.mean([np.mean([y[14] for y in u]) for u in per.values()])),
                     "mean_vehicles": float(np.mean([np.mean([y[13] for y in u]) for u in per.values()])),
                     "mean_score": float(np.mean(sc)) if sc else None, "std_score_over_instances": float(np.std(sc)) if sc else None,
                     "mean_run_time_s": float(np.mean([x[12] for x in v])), "mean_iterations": float(np.mean([x[11] for x in v])),
                     "budget_s_min_max": [float(min(x[9] for x in v)), float(max(x[9] for x in v))]})
    json.dump({"rows": len(rows), "files": [os.path.basename(f) for f in files], "groups": summ}, open(os.path.join(OUT, "results_summary.json"), "w"), indent=1)
    # ---- catalogue
    out = ["# Experiment catalogue\n", f"Built from the original result files by `scripts/build_master_results.py`. {len(rows)} rows in `master_results.csv` "
           "(rows with `derived = 1` are the value of a run before its set-partitioning step). Original files are in `vrp-hgnn-alns/results/` and are never changed.\n",
           "Runs that are not searches (offline evaluations E07b, E09c, E13; teacher runs E12) are listed at the end.\n"]
    bye = collections.defaultdict(list)
    for s in summ:
        bye[s["experiment"]].append(s)
    for e, (title, objective, outcome) in EXP.items():
        g = bye.get(e, [])
        if not g:
            out.append(f"\n## {e}. {title}\n\n- Objective: {objective}\n- **Status: no runs found**\n"); continue
        runs = sum(s["runs"] for s in g); feas = np.mean([s["feasible_share"] for s in g])
        out.append(f"\n## {e}. {title}\n")
        out.append(f"- **Objective / hypothesis:** {objective}")
        out.append(f"- **Outcome:** {outcome}")
        out.append(f"- **Status:** completed; {runs} runs; share of feasible runs with all requests served: {100 * feas:.1f}%")
        out.append(f"- **Clock:** {', '.join(sorted({s['clock'] for s in g}))}; budget per run {min(s['budget_s_min_max'][0] for s in g):.1f}–{max(s['budget_s_min_max'][1] for s in g):.1f} s")
        out.append(f"- **Source file(s):** {', '.join(sorted({x[1] for x in rows if x[0] == e}))}\n")
        out.append("| Benchmark | Algorithm / configuration | Inst. | Runs | Seeds | Mean objective | Mean vehicles | Mean score | Mean run time (s) | Mean iterations |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in g:
            sd = s["seeds"]; sdt = f"{min(sd)}–{max(sd)}" if len(sd) > 1 else str(sd[0])
            out.append(f"| {s['benchmark']} | {s['label']} (`{s['algorithm']}`) | {s['instances']} | {s['runs']} | {sdt} | {s['mean_objective']:.2f} | {s['mean_vehicles']:.2f} | "
                       f"{'–' if s['mean_score'] is None else format(s['mean_score'], '.2f')} | {s['mean_run_time_s']:.1f} | {s['mean_iterations']:.0f} |")
    nt = len(glob.glob(os.path.join(REPO, "data", "teachers", "*", "m4", "*.json")))
    out.append("\n## Experiments without search runs in the table\n")
    out.append("| ID | What | Size | Files | Status |\n|---|---|---|---|---|")
    out.append("| E07b | Label accuracy: exact insertion cost against the network | 60,561 removed requests on 130 held-out instances | `results/insertion_baseline.json` | completed |")
    out.append("| E09c | Predictability of long-run co-routing (existing predictors) | 48 tuning instances, three 4x runs each | `results/co_eval/base.json` | completed |")
    out.append(f"| E12 | Teacher runs for training | {nt} runs found (3 seeds, 4x iterations) | `data/teachers/*/m4/` | completed |")
    out.append("| E13 | Training of 6 models and offline gate G3 | 300 + 300 training instances | `models/v2/`, `logs/train_v2/`, `results/co_eval/v2.json` | completed |")
    other = [s for s in summ if s["experiment"] == "other"]
    if other:
        out.append("\nRuns not assigned to an experiment: " + ", ".join(f"{s['algorithm']} ({s['runs']})" for s in other))
    open(os.path.join(OUT, "EXPERIMENT_CATALOG.md"), "w").write("\n".join(out) + "\n")
    c = collections.Counter(x[0] for x in rows)
    print(len(rows), "rows;", dict(sorted(c.items())))


if __name__ == "__main__":
    main()
