"""Benchmark confirmation runs (equal iterations, iteration clock): E15 set partitioning, E16 oracle / trained model as prior."""
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_main import REPO, GROUPS, GNAME, bks_tables, compare, md_table, f2, fp   # noqa: E402
from analyse_followup import prep, index                                             # noqa: E402

NAME = {"B1_sp": "Classical ALNS + set partitioning", "O_oracle@prior1": "Oracle (4x run), prior only, lambda 1.0",
        "O_oracle@prior02": "Oracle (4x run), prior only, lambda 0.2", "O_oracle": "Oracle (4x run), pruning + prior + destroy (E07)",
        "C2_co@hgnn_l02": "Teacher-trained HGNN, prior only, lambda 0.2", "P_hgnn": "First method (HGNN-ALNS)"}


def main():
    F = prep(os.path.join(REPO, "results", "followup.jsonl"), bks_tables())
    by = index(F)
    insts = collections.defaultdict(set)
    for r in F:
        insts[r["group"]].add(r["instance"])
    out = ["# Benchmark confirmation (equal iterations)\n",
           "Positive difference = worse than classical ALNS (B1) with the same seeds and the same number of iterations.\n"]
    for title, algos in (("E15. Set partitioning over the route pool (all instances, 5 seeds)", ["B1_sp"]),
                         ("E16. Advice as prior only (36-instance subset, 3 seeds)", ["O_oracle@prior1", "O_oracle@prior02", "O_oracle", "C2_co@hgnn_l02", "P_hgnn"])):
        t = []
        for al in algos:
            for g in GROUPS + ["ALL"]:
                ii = sorted(i for gg in (GROUPS if g == "ALL" else [g]) for i in insts[gg])
                if al != "B1_sp":      # subset: only instances that have every variant of this table
                    ii = [i for i in ii if all((i, a) in by for a in ("O_oracle@prior1", "O_oracle", "B1_alns"))]
                sub = {k: ({s: r for s, r in v.items() if s in (0, 1, 2)} if al != "B1_sp" else v) for k, v in by.items()}
                c = compare(sub, ii, al, "B1_alns")
                if c is None:
                    continue
                extra = ""
                if al == "B1_sp":
                    rs = [r for i in ii for r in by.get((i, al), {}).values() if r.get("sp")]
                    if rs:
                        extra = (f"{100 * np.mean([r['sp']['improved'] for r in rs]):.0f}% of runs improved; "
                                 f"solver {np.mean([r['sp']['time'] for r in rs]):.1f} s on average; pool {np.mean([r['sp']['pool'] for r in rs]):.0f} routes")
                t.append([NAME.get(al, al), GNAME.get(g, "All instances"), c["n"], f2(c["mean_a"]), f2(c["mean_b"]), f"{c['diff']:+.2f}",
                          f"{c['win']} / {c['tie']} / {c['loss']}", fp(c["p"]), f2(c["a12"]), extra])
        if t:
            out.append(f"\n## {title}\n")
            out.append(md_table(["Variant", "Group", "Inst.", "Mean score", "B1 mean score", "Diff.", "Better / same / worse", "Wilcoxon p", "A12", "Note"], t))
    d = os.path.join(REPO, "results", "confirm_analysis")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "report.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
