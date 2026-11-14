"""Pilot for the revised methodology (tuning instances only, equal iterations, seeds 0-1).

Reads results/pilot_headroom.jsonl and compares every variant with classical ALNS (B1) on the same instance:
  S_best m : perfect neighbourhood selection among m random moves, same number of iterations (m times the work)
  W_best m : the same, but with iterations / m (same work as B1)
  P_hgnn   : the current method;  C_copred : guide = the network's static co-routing head;
  C_self   : guide = the run's own best solution;  O_oracle : guide = a 4x-iteration reference;  REF_long : that reference
Difference = 100 x (vehicles difference, Li & Lim style only) + % distance difference; positive = worse than B1.
"""
import collections
import json
import os
import sys

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
NAME = {"S_best2": "Best of 2 moves, equal iterations", "S_best4": "Best of 4 moves, equal iterations",
        "W_best2": "Best of 2 moves, equal work", "W_best4": "Best of 4 moves, equal work", "W_best8": "Best of 8 moves, equal work",
        "P_hgnn": "HGNN-ALNS (current method)", "C_copred": "Guide = network co-routing head (static)",
        "C_self": "Guide = own best solution (no network)", "O_oracle": "Guide = oracle (4x reference)",
        "B1_sp": "Classical ALNS + set partitioning at the end",
        "REF_long": "Classical ALNS, 4x iterations", "REF_x1": "Classical ALNS, 1x iterations, other seed",
        "REF_x06": "Classical ALNS, 0.6x iterations, other seed",
        "O_ref1": "Guide = oracle from a 1x reference", "O_ref06": "Guide = oracle from a 0.6x reference",
        "O_soft": "Guide = soft consensus of three 4x runs", "C_shaw": "Guide = static relatedness (no learning)",
        "O_oracle@prior": "Oracle (4x), prior only (lambda 0.2)", "O_oracle@prior1": "Oracle (4x), prior only (lambda 1.0)",
        "O_oracle@destroy": "Oracle (4x), destroy operator only", "O_oracle@prune": "Oracle (4x), pruning only (top 50%)"}
NAME.update({"C2_co@hgnn_l1": "Teacher-trained HGNN, prior only (lambda 1.0)", "C2_co@hgnn_l02": "Teacher-trained HGNN, prior only (lambda 0.2)",
             "C2_co@hom_l1": "Teacher-trained homogeneous GNN, prior only (lambda 1.0)",
             "C2_co@mlp_l1": "Teacher-trained model without message passing, prior only (lambda 1.0)",
             "C2_co@old_l1": "Old co-routing head (short-run teacher), prior only (lambda 1.0)",
             "O_soft@prior1": "Soft consensus of three 4x runs, prior only (lambda 1.0)",
             "C_shaw@prior1": "Static relatedness (no learning), prior only (lambda 1.0)"})
REFS = ("REF_long", "REF_x1", "REF_x06")


def main():
    rows = [json.loads(l) for l in open(os.path.join(REPO, "results", "pilot_headroom.jsonl"))]
    by = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        by[(r["family"], r["instance"])][r["algo"]][r["seed"]] = r
    out = ["# Pilot on the tuning instances (equal iterations)\n",
           f"Runs: {len(rows)}; all feasible: {all(r['checker_ok'] for r in rows)}; unserved: {sum(r['unserved'] > 0 for r in rows)}.\n",
           "Positive difference = worse than classical ALNS (B1) on the same instance.\n"]
    J = []
    for fam, lex, title in (("gen-lilim", True, "Li & Lim style (20 instances)"), ("gen-hdarp", False, "H-DARP style (30 instances)")):
        t = []
        for al in NAME:
            d, tr, imp, impb = [], [], [], []
            nuns = 0
            for (f, i), v in by.items():
                if f != fam or al not in v or "B1_alns" not in v:
                    continue
                a, b = v[al], v["B1_alns"]
                sa = sorted(a) if al in REFS else sorted(set(a) & set(b))
                sb = sorted(b) if al in REFS else sa
                if not sa or not sb:
                    continue
                va, vb = np.mean([a[s]["vehicles"] for s in sa]), np.mean([b[s]["vehicles"] for s in sb])
                da, db = np.mean([a[s]["distance"] for s in sa]), np.mean([b[s]["distance"] for s in sb])
                ua, ub = np.mean([a[s]["unserved"] for s in sa]), np.mean([b[s]["unserved"] for s in sb])
                if ua > 0 or ub > 0:          # distances are not comparable when requests are left unserved
                    nuns += 1
                    continue
                d.append((100.0 * (va - vb) if lex else 0.0) + 100.0 * (da - db) / db)
                tr.append(np.mean([a[s]["real_elapsed"] for s in sa]) / np.mean([b[s]["real_elapsed"] for s in sb]))
                imp.append(np.mean([a[s]["diag"]["improved"] / max(1, a[s]["diag"]["iters"]) for s in sa]))
                impb.append(np.mean([b[s]["diag"]["improved"] / max(1, b[s]["diag"]["iters"]) for s in sb]))
            if not d:
                continue
            d = np.array(d)
            p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            t.append(f"| {NAME[al]} | {len(d)} | {d.mean():+.2f} | {np.median(d):+.2f} | {int((d < 0).sum())} / {int((d == 0).sum())} / "
                     f"{int((d > 0).sum())} | {'<0.001' if p < 0.001 else f'{p:.3f}'} | {np.median(tr):.2f} | {100 * np.mean(imp):.1f}% | "
                     f"{100 * np.mean(impb):.1f}% | {nuns} |")
            J.append({"family": fam, "algo": al, "n": len(d), "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
                      "better": int((d < 0).sum()), "worse": int((d > 0).sum()), "p": float(p), "time_ratio": float(np.median(tr)),
                      "improving": float(np.mean(imp)), "improving_b1": float(np.mean(impb))})
        out.append(f"\n## {title}\n")
        out.append("| Variant | Inst. | Mean diff. | Median diff. | Better / same / worse | Wilcoxon p | Time / time of B1 | Improving iterations | B1 improving | Inst. left out (unserved requests) |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        out += t
    os.makedirs(os.path.join(REPO, "results", "pilot_analysis"), exist_ok=True)
    open(os.path.join(REPO, "results", "pilot_analysis", "report.md"), "w").write("\n".join(out) + "\n")
    json.dump(J, open(os.path.join(REPO, "results", "pilot_analysis", "summary.json"), "w"), indent=1)
    print("\n".join(out))


if __name__ == "__main__":
    main()
