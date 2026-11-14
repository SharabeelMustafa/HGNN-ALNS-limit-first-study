"""Final ablation and baseline tables for the research package (benchmarks, equal iterations unless stated).

Reads results/followup.jsonl and results/core.jsonl (never changes them). Writes
  ../Research_Package/04_ablations/ABLATION_STUDIES.md, ablation_summary.json, ablation_per_instance.csv
  ../Research_Package/05_baselines/per_instance_benchmark.csv, baseline_group_summary.json
"""
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_main import REPO, GROUPS, GNAME, bks_tables, compare, md_table, f2, fp, score_of   # noqa: E402
from analyse_followup import prep, index                                                        # noqa: E402

PK = os.path.abspath(os.path.join(REPO, "..", "Research_Package"))
HINT, HSP, B1, BSP, ORA, P1 = "C2_co@hgnn_l02", "C2_sp@hgnn_l02", "B1_alns", "B1_sp", "O_oracle@prior1", "P_hgnn"
NAME = {B1: "Classical ALNS", BSP: "Classical ALNS + set partitioning", HINT: "ALNS + HGNN soft hint", HSP: "ALNS + HGNN soft hint + set partitioning",
        ORA: "ALNS + oracle soft hint (4x reference)", P1: "First method (route prior + pruning)", "A_INT2_prior": "First method, prior only",
        "A_INT1_prune": "First method, pruning only"}


def main():
    bks = bks_tables()
    F = prep(os.path.join(REPO, "results", "followup.jsonl"), bks)
    extra = []
    for r in F:                                    # the same run before its set-partitioning step = soft hint alone
        sp = r.get("sp") or {}
        if r["algo"] == HSP and "before_distance" in sp:
            q = dict(r); q["algo"] = HINT + "#pre"; q["distance"] = sp["before_distance"]; q["vehicles"] = sp["before_vehicles"]
            q["score"] = score_of(q, q["vehicles"], q["distance"]); q["real_elapsed"] = (r.get("real_elapsed") or 0) - sp.get("time", 0.0)
            extra.append(q)
    by = index(F + extra)
    # hint alone: use the direct runs where they exist (36 instances, seeds 0-2) only as a consistency check; the table uses the pre-step values
    H = HINT + "#pre"
    NAME[H] = NAME[HINT]
    same = tot = 0
    for (i, a), v in list(by.items()):
        if a == HINT:
            for s, r in v.items():
                q = by.get((i, H), {}).get(s)
                if q is not None:
                    tot += 1; same += abs(q["distance"] - r["distance"]) < 1e-6 and q["vehicles"] == r["vehicles"]
    insts = collections.defaultdict(set)
    for r in F:
        insts[r["group"]].add(r["instance"])
    GA = GROUPS + ["ALL"]
    def ii(g, need):
        base = sorted(i for gg in (GROUPS if g == "ALL" else [g]) for i in insts[gg])
        return [i for i in base if all((i, a) in by for a in need)]
    def seeds(d, ss):
        return {k: {s: r for s, r in v.items() if s in ss} for k, v in d.items()}
    out = ["# Ablation studies\n",
           "All tables: real benchmark instances, the same seeds and the same number of ALNS iterations for every variant (iteration clock), so runs are paired and deterministic. "
           "Score: Li & Lim = 100 x extra vehicles + % distance gap to the best-known solution; H-DARP = % cost gap. **Lower is better. A positive difference means the first variant is worse.** "
           "B/S/W = instances where the first variant is better / the same / worse. p = paired Wilcoxon signed-rank test on instance means.\n",
           f"Consistency check: the soft-hint-only result is read from the hint + set-partitioning runs before their final step. Where direct hint-only runs exist, both agree in {same} of {tot} runs.\n"]
    J = {"consistency": {"same": int(same), "pairs": int(tot)}, "tables": {}}

    def table(title, key, pairs, ss, note=None):
        t = []
        for a, b in pairs:
            for g in GA:
                I = ii(g, (a, b))
                c = compare(seeds(by, ss), I, a, b)
                if c is None:
                    continue
                t.append([NAME.get(a, a), NAME.get(b, b), GNAME.get(g, "All instances"), c["n"], f2(c["mean_a"]), f2(c["mean_b"]), f"{c['diff']:+.2f}",
                          f"{c['win']} / {c['tie']} / {c['loss']}", fp(c["p"])])
                J["tables"].setdefault(key, []).append({"first": a, "second": b, "group": g, **c})
        out.append(f"\n## {title}\n")
        if note:
            out.append(note + "\n")
        out.append(md_table(["First variant", "Compared with", "Benchmark", "Inst.", "Mean score (first)", "Mean score (second)", "Diff.", "B / S / W", "p"], t))

    S5, S3 = (0, 1, 2, 3, 4), (0, 1, 2)
    # ---------------- overview 2 x 2
    out.append("\n## Overview: soft hint x set partitioning (all 140 instances, 5 seeds)\n")
    t = []
    for g in GA:
        I = ii(g, (B1, BSP, H, HSP))
        if not I:
            continue
        m = {a: np.mean([np.mean([by[(i, a)][s]["score"] for s in S5 if s in by[(i, a)]]) for i in I]) for a in (B1, BSP, H, HSP)}
        t.append([GNAME.get(g, "All instances"), len(I), f2(m[B1]), f2(m[BSP]), f2(m[H]), f2(m[HSP])])
        J["tables"].setdefault("overview", []).append({"group": g, "n": len(I), **{NAME[a]: float(v) for a, v in m.items()}})
    out.append(md_table(["Benchmark", "Inst.", "Classical ALNS", "+ set partitioning", "+ HGNN soft hint", "+ HGNN soft hint + set partitioning"], t))
    # ---------------- ablation 1
    C = prep(os.path.join(REPO, "results", "core.jsonl"), bks)
    cby = index([r for r in C if abs(r["budget"] - 0.5 * r["n"]) < 1e-6])
    t = []
    for g in GA:
        I = sorted(i for gg in (GROUPS if g == "ALL" else [g]) for i in insts[gg])
        c = compare(cby, I, P1, B1)
        if c:
            t.append(["First method, equal time (0.5 s per request, 10 seeds)", GNAME.get(g, "All instances"), c["n"], f2(c["mean_a"]), f2(c["mean_b"]), f"{c['diff']:+.2f}",
                      f"{c['win']} / {c['tie']} / {c['loss']}", fp(c["p"])])
            J["tables"].setdefault("ablation1", []).append({"variant": "first method, equal time", "group": g, **c})
    for lab, a, ss in (("First method, equal iterations (5 seeds)", P1, S5), ("Revised method: HGNN soft hint, equal iterations (5 seeds)", H, S5)):
        for g in GA:
            c = compare(seeds(by, ss), ii(g, (a, B1)), a, B1)
            if c:
                t.append([lab, GNAME.get(g, "All instances"), c["n"], f2(c["mean_a"]), f2(c["mean_b"]), f"{c['diff']:+.2f}", f"{c['win']} / {c['tie']} / {c['loss']}", fp(c["p"])])
                J["tables"].setdefault("ablation1", []).append({"variant": lab, "group": g, **c})
    out.append("\n## Ablation 1. Classical ALNS against HGNN-guided ALNS\n")
    out.append(md_table(["HGNN-guided variant", "Benchmark", "Inst.", "Mean score (guided)", "Mean score (classical ALNS)", "Diff.", "B / S / W", "p"], t))
    # ---------------- ablations 2-4
    table("Ablation 2. Soft hint enabled against disabled", "ablation2", [(H, B1), (HSP, BSP), ("A_INT2_prior", B1)], S5,
          "Rows 1-2: the teacher-trained HGNN as soft hint (lambda 0.2), without and with set partitioning. Row 3: the prior of the first method alone (no pruning, no learned removal).")
    table("Ablation 3. Set partitioning enabled against disabled", "ablation3", [(BSP, B1), (HSP, H)], S5)
    t = []
    for a in (BSP, HSP):
        for g in GA:
            rs = [r for i in ii(g, (a,)) for s, r in by[(i, a)].items() if s in S5 and r.get("sp")]
            if rs:
                t.append([NAME[a], GNAME.get(g, "All instances"), len(rs), f"{100 * np.mean([r['sp']['improved'] for r in rs]):.0f}%",
                          f"{np.mean([r['sp']['time'] for r in rs]):.2f}", f"{np.max([r['sp']['time'] for r in rs]):.1f}", f"{np.mean([r['sp']['pool'] for r in rs]):.0f}",
                          f"{100 * np.mean([r['sp']['time'] / max(r['real_elapsed'], 1e-9) for r in rs]):.1f}%"])
    out.append("\nCost of the set-partitioning step:\n")
    out.append(md_table(["Variant", "Benchmark", "Runs", "Runs improved", "Mean solver time (s)", "Max solver time (s)", "Mean pool size (routes)", "Share of the run time"], t))
    table("Ablation 4. Predicted HGNN guidance against oracle guidance", "ablation4", [(H, B1), (ORA, B1), (H, ORA)], S3,
          "Instances with a 4x reference run; seeds 0-2. The oracle reads the co-routing of one reference solution (prior only, lambda 1.0). It is an upper limit, not a usable method.")
    t = []
    for g in GA:
        I = ii(g, (H, ORA, B1))
        if not I:
            continue
        m = {a: np.mean([np.mean([by[(i, a)][s]["score"] for s in S3 if s in by[(i, a)]]) for i in I]) for a in (B1, H, ORA)}
        og, hg = m[B1] - m[ORA], m[B1] - m[H]
        t.append([GNAME.get(g, "All instances"), len(I), f2(m[B1]), f2(m[H]), f2(m[ORA]), f"{og:+.2f}", f"{hg:+.2f}", "–" if og <= 0 else f"{100 * hg / og:.0f}%"])
    out.append("\nHow much of the oracle's gain does the trained model reach? (gain = classical score minus guided score; positive = better than classical)\n")
    out.append(md_table(["Benchmark", "Inst.", "Classical ALNS", "HGNN soft hint", "Oracle soft hint", "Gain of the oracle", "Gain of the HGNN", "Share reached"], t))
    out.append("\nCaution for this table: it compares means. On Li & Lim a mean is moved by a few instances where one vehicle more or less changes the score by 100 points. "
               "The B / S / W counts above are the safer reading: with the HGNN hint more instances get worse than better on every set, and the difference to classical ALNS is "
               "significant only on H-DARP, where the hint is worse.\n")
    # ---------------- overhead
    t = []
    for a, ss in ((BSP, S5), (H, S5), (HSP, S5), (ORA, S3), (P1, S5)):
        row = [NAME.get(a, a)]
        for g in GROUPS:
            rr = []
            for i in ii(g, (a, B1)):
                sa = [s for s in ss if s in by[(i, a)] and s in by[(i, B1)]]
                if sa:
                    rr.append(np.mean([by[(i, a)][s]["real_elapsed"] + (by[(i, a)][s].get("sp") or {}).get("time", 0.0) * (a == H) * 0 for s in sa]) / np.mean([by[(i, B1)][s]["real_elapsed"] for s in sa]))
            row.append(f"x {np.median(rr):.2f}" if rr else "–")
        t.append(row)
    out.append("\n## Computational overhead at equal iterations\n\nMedian over instances of (run time of the variant) / (run time of classical ALNS). "
               "Timings were taken with 12 runs in parallel on a laptop and on different days, so they are approximate (about ±10%).\n")
    out.append(md_table(["Variant"] + [GNAME[g] for g in GROUPS], t))
    # ---------------- vehicles on Li & Lim, run level
    from scipy import stats as _st
    t = []
    for g in ("LL100", "LL200"):
        for a, ss in ((BSP, S5), (H, S5), (HSP, S5), (P1, S5), (ORA, S3)):
            fewer = more = same = 0; fi, mi = set(), set()
            for i in ii(g, (a, B1)):
                for s_ in ss:
                    if s_ in by[(i, a)] and s_ in by[(i, B1)]:
                        d = by[(i, a)][s_]["vehicles"] - by[(i, B1)][s_]["vehicles"]
                        if d < 0:
                            fewer += 1; fi.add(i)
                        elif d > 0:
                            more += 1; mi.add(i)
                        else:
                            same += 1
            pv = _st.binomtest(fewer, fewer + more, 0.5).pvalue if fewer + more else 1.0
            t.append([GNAME[g], NAME.get(a, a), fewer, more, same, f"{len(fi)} / {len(mi)}", fp(pv)])
            J["tables"].setdefault("vehicles", []).append({"group": g, "algo": a, "fewer": fewer, "more": more, "same": same, "inst_fewer": len(fi), "inst_more": len(mi), "p_sign_runs": float(pv)})
    out.append("\n## Li & Lim: where the mean scores come from (number of vehicles, run by run)\n\n"
               "One vehicle more or less changes the score by 100 points, so a few runs decide the mean. The table counts paired runs (same instance, same seed) "
               "with fewer or more vehicles than classical ALNS. The sign test treats runs as independent, which they are not within an instance, so it is only a rough guide.\n")
    out.append(md_table(["Benchmark", "Variant", "Runs with fewer vehicles", "Runs with more vehicles", "Runs with the same number", "Instances with fewer / more", "Sign test p (runs)"], t))
    out.append("\nReading: on Li & Lim 100 the lower mean score of the hint comes from 4 of the 56 instances, where some runs save one vehicle. "
               "On Li & Lim 200 the hint saves a vehicle as often as it loses one. So there is no reliable effect of the trained hint on the number of vehicles.\n")
    os.makedirs(os.path.join(PK, "04_ablations"), exist_ok=True); os.makedirs(os.path.join(PK, "05_baselines"), exist_ok=True)
    open(os.path.join(PK, "04_ablations", "ABLATION_STUDIES.md"), "w").write("\n".join(out) + "\n")
    json.dump(J, open(os.path.join(PK, "04_ablations", "ablation_summary.json"), "w"), indent=1, default=float)
    # ---------------- per-instance table
    from hgalns.io import list_instances, load_instance
    K = {}
    for fam, size in (("lilim", 100), ("lilim", 200), ("malheiros", None)):
        for p in list_instances(fam, size):
            K[os.path.splitext(os.path.basename(p))[0]] = load_instance(p).K
    algos = [(B1, "classical_alns", S5), (BSP, "alns_sp", S5), (H, "alns_hint", S5), (HSP, "alns_hint_sp", S5), (ORA, "alns_oracle_hint", S3), (P1, "first_method", S5)]
    head = ["benchmark", "instance", "n_requests", "fleet_size", "bks_vehicles", "bks_objective"]
    for _, nm, _ in algos:
        head += [f"{nm}_mean_objective", f"{nm}_std_objective", f"{nm}_best_objective", f"{nm}_worst_objective", f"{nm}_mean_vehicles", f"{nm}_gap_pct", f"{nm}_score",
                 f"{nm}_mean_time_s", f"{nm}_feasible_runs"]
    rows = []
    for g in GROUPS:
        for i in sorted(insts[g]):
            r0 = next(iter(by[(i, B1)].values()))
            row = [GNAME[g], i, r0["n"], K.get(i, ""), "" if r0["bks_vehicles"] is None else r0["bks_vehicles"], r0["bks_distance"]]
            for a, nm, ss in algos:
                rs = [r for s, r in by.get((i, a), {}).items() if s in ss]
                if not rs:
                    row += [""] * 9; continue
                d = [r["distance"] for r in rs]
                row += [round(np.mean(d), 3), round(np.std(d), 3), round(min(d), 3), round(max(d), 3), round(np.mean([r["vehicles"] for r in rs]), 2),
                        round(100 * (np.mean(d) - r0["bks_distance"]) / r0["bks_distance"], 3), round(np.mean([r["score"] for r in rs]), 3),
                        round(np.mean([r["real_elapsed"] for r in rs]), 1), f"{sum(bool(r['checker_ok']) and r['unserved'] == 0 for r in rs)}/{len(rs)}"]
            rows.append(row)
    with open(os.path.join(PK, "05_baselines", "per_instance_benchmark.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(head); w.writerows(rows)
    print("\n".join(out)[:6000])


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(REPO, "src"))
    main()
