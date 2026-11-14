"""Full analysis of the main experiment (configs/core.yaml -> results/core.jsonl).

Blocks
  main      budget 0.5 s per request : B0, B1 (classical ALNS), P (HGNN-ALNS), B5 (homogeneous GNN),
                                       A_INT1 (pruning only), A_INT2 (prior only)
  deadline  budget 10 s and 20 s     : B1 vs P

Score of a run (lower is better):
  Li & Lim : 100 * (vehicles - BKS vehicles) + % distance above BKS      (lexicographic objective)
  H-DARP   : % cost above BKS
Statistics: the instance is the unit of analysis (seeds are replicates). For every comparison the two
algorithms use the same seeds. Wilcoxon signed-rank on per-instance mean scores, Holm correction over the
comparisons against the reference inside each group, Vargha-Delaney A12 (mean over instances), Friedman test.

Usage: python scripts/analyse_main.py results/core.jsonl --out results/core_analysis
"""
import argparse
import collections
import csv
import json
import os
import re
import warnings

import numpy as np
from scipy import stats

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..")
COLORS = {"B1_alns": "#1baf7a", "P_hgnn": "#2a78d6", "B5_homgnn": "#eb6834", "A_INT2_prior": "#6250d6",
          "A_INT1_prune": "#eda100", "B0_regret2": "#8a8984"}
LABEL = {"B0_regret2": "Regret-2 construction (B0)", "B1_alns": "Classical ALNS (B1)", "P_hgnn": "HGNN-ALNS (P)",
         "B5_homgnn": "Homogeneous GNN-ALNS (B5)", "A_INT1_prune": "Pruning only (A-INT1)",
         "A_INT2_prior": "Prior only (A-INT2)"}
SHORT = {"B0_regret2": "B0", "B1_alns": "B1", "P_hgnn": "P", "B5_homgnn": "B5", "A_INT1_prune": "A-INT1",
         "A_INT2_prior": "A-INT2"}
GROUPS = ["LL100", "LL200", "DARP"]
GNAME = {"LL100": "Li & Lim 100", "LL200": "Li & Lim 200", "DARP": "H-DARP (Malheiros)", "ALL": "All instances"}


def bks_tables():
    out = {}
    for r in csv.DictReader(open(os.path.join(REPO, "data", "bks", "lilim_bks.csv"))):
        m = re.match(r'[" ]*(l(?:rc|r|c)\d(?:_\d+_\d+|\d\d))', r["instance"].lower())   # drop footnote marks (*, q, k, ...)
        if m:
            out[m.group(1)] = (int(r["vehicles"]), float(r["distance"]))
    for r in csv.DictReader(open(os.path.join(REPO, "data", "bks", "malheiros2021_mdhdarp_bks.csv"))):
        out[r["instance"].lower()] = (None, float(r["bks"]))
    return out


def load(path):
    rows, bad = [], 0
    for line in open(path):
        if not line.strip() or line.startswith('{"_meta"'):
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            bad += 1
    return rows, bad


def score_of(r, veh, dist):
    gap = 100.0 * (dist - r["bks_distance"]) / r["bks_distance"]
    return gap if r["bks_vehicles"] is None else 100.0 * (veh - r["bks_vehicles"]) + gap


def curve(r, grid):
    """Incumbent score on a normalised time grid (0..1); 100 before the first complete solution."""
    c = np.full(len(grid), 100.0)
    for t, d, v, u in r["trajectory"]:
        if u == 0:
            c[grid >= t / r["budget"]] = min(score_of(r, v, d), 100.0)
    return c


def a12(x, y):
    return sum((xi < yj) + 0.5 * (xi == yj) for xi in x for yj in y) / (len(x) * len(y))


def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def group_of(r):
    if r["bks_vehicles"] is None:
        return "DARP"
    return "LL100" if r["n"] < 80 else "LL200"


def klass(inst):
    m = re.match(r"^(l[rc]c?)(\d)", inst.lower())
    return (m.group(1) + m.group(2)).upper() if m else "DARP"


def compare(by, insts, a, b, key="score"):
    """Algorithm a against reference b on the given instances, same seeds for both."""
    ma, mb, a12s = [], [], []
    for i in insts:
        ra, rb = by.get((i, a), {}), by.get((i, b), {})
        seeds = sorted(set(ra) & set(rb))
        if not seeds:
            continue
        xa, xb = [ra[s][key] for s in seeds], [rb[s][key] for s in seeds]
        ma.append(np.mean(xa)); mb.append(np.mean(xb)); a12s.append(a12(xa, xb))
    ma, mb = np.array(ma), np.array(mb)
    d = ma - mb
    if len(d) == 0:
        return None
    p = stats.wilcoxon(d).pvalue if np.any(np.abs(d) > 1e-9) else 1.0
    return {"n": len(d), "mean_a": float(ma.mean()), "mean_b": float(mb.mean()), "diff": float(d.mean()),
            "median_diff": float(np.median(d)), "win": int((d < -1e-6).sum()), "tie": int((np.abs(d) <= 1e-6).sum()),
            "loss": int((d > 1e-6).sum()), "p": float(p), "a12": float(np.mean(a12s))}


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def f2(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def fp(p):
    return "–" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="B1_alns")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, bad = load(a.results)
    bks = bks_tables()
    R = []
    for r in rows:
        b = bks.get(r["instance"].lower().replace("hetiuy", ""))
        if b is None:
            continue
        r["bks_vehicles"], r["bks_distance"] = b
        r["gap_pct"] = 100.0 * (r["distance"] - b[1]) / b[1]
        r["score"] = score_of(r, r["vehicles"], r["distance"]) if r["unserved"] == 0 else 1000.0
        r["extra_veh"] = 0 if b[0] is None else r["vehicles"] - b[0]
        r["ips"] = r["iterations"] / max(r["elapsed"], 1e-9)
        r["group"] = group_of(r)
        r["block"] = "deadline" if r["budget"] in (10.0, 20.0) else "main"
        R.append(r)
    grid = np.linspace(0, 1, 101)
    for r in R:
        r["curve"] = curve(r, grid)
        r["pi"] = float(np.trapz(r["curve"], grid))
    out = [f"# Main experiment analysis\n\nRuns loaded: {len(R)} (unreadable lines skipped: {bad}; runs without BKS: {len(rows) - len(R)}).",
           f"Checker: {sum(r['checker_ok'] for r in R)} of {len(R)} runs feasible. Runs with unserved requests: {sum(r['unserved'] > 0 for r in R)}."]
    M = [r for r in R if r["block"] == "main"]
    D = [r for r in R if r["block"] == "deadline"]
    algos = [x for x in LABEL if any(r["algo"] == x for r in M)]
    by = collections.defaultdict(dict)
    for r in M:
        by[(r["instance"], r["algo"])][r["seed"]] = r
    inst_group = {r["instance"]: r["group"] for r in M}
    inst_n = {r["instance"]: r["n"] for r in M}
    insts_of = {g: sorted(i for i, gg in inst_group.items() if gg == g) for g in GROUPS}
    insts_of["ALL"] = sorted(inst_group)
    J = {"runs": len(R), "main": {}, "deadline": {}}

    # ------------------------------------------------------------------ T1: summary per group (all seeds)
    out.append("\n## Table 1. Summary per benchmark group (budget 0.5 s per request; all seeds)\n")
    t = []
    for g in GROUPS + ["ALL"]:
        for al in algos:
            rs = [r for i in insts_of[g] for r in by.get((i, al), {}).values()]
            if not rs:
                continue
            per_inst = [np.mean([r["score"] for r in by[(i, al)].values()]) for i in insts_of[g] if (i, al) in by]
            ll = [r for r in rs if r["bks_vehicles"] is not None]
            row = {"group": g, "algo": al, "instances": len(per_inst), "runs": len(rs), "score": float(np.mean(per_inst)),
                   "gap": float(np.mean([r["gap_pct"] for r in rs])),
                   "extra_veh": float(np.mean([r["extra_veh"] for r in ll])) if ll else None,
                   "bks_veh_share": float(np.mean([r["extra_veh"] <= 0 for r in ll])) if ll else None,
                   "pi": float(np.mean([r["pi"] for r in rs])), "ips": float(np.mean([r["ips"] for r in rs])),
                   "setup": float(np.mean([r["t_guide_setup"] for r in rs]))}
            J["main"].setdefault("summary", []).append(row)
            t.append([GNAME[g], LABEL[al], row["instances"], row["runs"], f2(row["score"]), f2(row["extra_veh"]),
                      "–" if row["bks_veh_share"] is None else f"{100 * row['bks_veh_share']:.0f}%", f2(row["gap"]),
                      f2(row["pi"]), f"{row['ips']:.0f}", f2(row["setup"], 3)])
    out.append(md_table(["Group", "Algorithm", "Inst.", "Runs", "Mean score", "Extra vehicles vs BKS", "Runs at BKS vehicles",
                         "Distance/cost gap %", "Primal integral", "Iter./s", "Guide set-up (s)"], t))

    # ------------------------------------------------------------------ T2: Li & Lim cumulative vehicles / distance
    out.append("\n## Table 2. Li & Lim in the standard form: cumulative number of vehicles (CNV) and cumulative distance (CTD), mean over seeds\n")
    t = []
    for g in ("LL100", "LL200"):
        bv = sum(bks[i.lower()][0] for i in insts_of[g]); bd = sum(bks[i.lower()][1] for i in insts_of[g])
        t.append([GNAME[g], "Best known (BKS)", bv, f"{bd:.0f}"])
        for al in algos:
            if not all((i, al) in by for i in insts_of[g]):
                continue
            cnv = sum(np.mean([r["vehicles"] for r in by[(i, al)].values()]) for i in insts_of[g])
            ctd = sum(np.mean([r["distance"] for r in by[(i, al)].values()]) for i in insts_of[g])
            J["main"].setdefault("cnv", []).append({"group": g, "algo": al, "cnv": float(cnv), "ctd": float(ctd), "bks_cnv": bv, "bks_ctd": bd})
            t.append([GNAME[g], LABEL[al], f"{cnv:.1f}", f"{ctd:.0f}"])
    out.append(md_table(["Group", "Algorithm", "CNV", "CTD"], t))

    # ------------------------------------------------------------------ T3: tests against the reference
    out.append(f"\n## Table 3. Statistical comparison against {LABEL[a.ref]} (instance-level, same seeds; negative difference = better than the reference)\n")
    t = []
    for g in GROUPS + ["ALL"]:
        cs = [(al, compare(by, insts_of[g], al, a.ref)) for al in algos if al not in (a.ref, "B0_regret2")]
        cs = [(al, c) for al, c in cs if c]
        adj = holm([c["p"] for _, c in cs])
        for (al, c), pa in zip(cs, adj):
            c["p_holm"] = pa
            J["main"].setdefault("vs_ref", []).append({"group": g, "algo": al, **c})
            t.append([GNAME[g], LABEL[al], c["n"], f2(c["mean_a"]), f2(c["mean_b"]), f"{c['diff']:+.2f}", f"{c['median_diff']:+.2f}",
                      f"{c['win']} / {c['tie']} / {c['loss']}", fp(c["p"]), fp(pa), f2(c["a12"])])
    out.append(md_table(["Group", "Algorithm", "Inst.", "Mean score", "Ref. mean score", "Mean diff.", "Median diff.",
                         "Win / tie / loss", "Wilcoxon p", "Holm p", "A12"], t))

    # ------------------------------------------------------------------ T4: other pairwise questions
    out.append("\n## Table 4. Pairwise comparisons for RQ2 and RQ4 (instance-level, same seeds; negative difference = first is better)\n")
    t = []
    pairs = [("P_hgnn", "B5_homgnn", "RQ2: heterogeneous vs homogeneous GNN"),
             ("P_hgnn", "A_INT1_prune", "full guidance vs pruning only"),
             ("P_hgnn", "A_INT2_prior", "full guidance vs prior only"),
             ("A_INT2_prior", "A_INT1_prune", "prior only vs pruning only")]
    for x, y, what in pairs:
        for g in GROUPS + ["ALL"]:
            c = compare(by, insts_of[g], x, y)
            if not c:
                continue
            J["main"].setdefault("pairs", []).append({"group": g, "a": x, "b": y, **c})
            t.append([what, GNAME[g], c["n"], f2(c["mean_a"]), f2(c["mean_b"]), f"{c['diff']:+.2f}",
                      f"{c['win']} / {c['tie']} / {c['loss']}", fp(c["p"]), f2(c["a12"])])
    out.append(md_table(["Comparison", "Group", "Inst.", "Mean score (first)", "Mean score (second)", "Mean diff.",
                         "Win / tie / loss", "Wilcoxon p", "A12"], t))

    # ------------------------------------------------------------------ Friedman
    out.append("\n## Friedman tests (iterative algorithms; instances as blocks; common seeds)\n")
    it_algos = [al for al in algos if al != "B0_regret2"]
    for g in GROUPS + ["ALL"]:
        ii = [i for i in insts_of[g] if all((i, al) in by for al in it_algos)]
        if len(ii) < 3:
            continue
        Mx = []
        for i in ii:
            seeds = sorted(set.intersection(*[set(by[(i, al)]) for al in it_algos]))
            Mx.append([np.mean([by[(i, al)][s]["score"] for s in seeds]) for al in it_algos])
        Mx = np.array(Mx)
        fr = stats.friedmanchisquare(*Mx.T)
        ranks = np.mean([stats.rankdata(np.round(row, 6)) for row in Mx], axis=0)
        J["main"].setdefault("friedman", []).append({"group": g, "n": len(ii), "chi2": float(fr.statistic), "p": float(fr.pvalue),
                                                     "ranks": dict(zip(it_algos, map(float, ranks)))})
        out.append(f"- **{GNAME[g]}** ({len(ii)} instances): chi2 = {fr.statistic:.2f}, p = {fp(fr.pvalue)}; mean ranks: " +
                   ", ".join(f"{SHORT[al]} {rk:.2f}" for al, rk in zip(it_algos, ranks)))

    # ------------------------------------------------------------------ T5: by instance class (RQ4)
    out.append("\n## Table 5. When does guidance help? HGNN-ALNS (P) against classical ALNS (B1) by instance class\n")
    t = []
    cls = collections.defaultdict(list)
    for i in insts_of["ALL"]:
        k = klass(i)
        cls[(inst_group[i], k if k != "DARP" else ("DARP n<=120" if inst_n[i] <= 120 else "DARP n>120"))].append(i)
    for (g, k), ii in sorted(cls.items(), key=lambda kv: (GROUPS.index(kv[0][0]), kv[0][1])):
        c = compare(by, ii, "P_hgnn", a.ref)
        if not c:
            continue
        routes = np.mean([np.mean([r["vehicles"] for r in by[(i, a.ref)].values()]) for i in ii if (i, a.ref) in by])
        ev = compare(by, ii, "P_hgnn", a.ref, key="extra_veh")
        J["main"].setdefault("by_class", []).append({"group": g, "class": k, "routes": float(routes), **c, "extra_veh_diff": ev["diff"]})
        t.append([GNAME[g], k, c["n"], f"{routes:.1f}", f2(c["mean_a"]), f2(c["mean_b"]), f"{c['diff']:+.2f}", f"{ev['diff']:+.3f}",
                  f"{c['win']} / {c['tie']} / {c['loss']}", fp(c["p"])])
    out.append(md_table(["Group", "Class", "Inst.", "Mean routes", "P mean score", "B1 mean score", "Mean diff.",
                         "Diff. in vehicles", "Win / tie / loss", "Wilcoxon p"], t))

    # per-instance table (CSV) and correlation of the P - B1 difference with the number of routes and size
    per = []
    for i in insts_of["ALL"]:
        row = {"instance": i, "group": inst_group[i], "class": klass(i), "n": inst_n[i],
               "bks_vehicles": bks[i.lower().replace("hetiuy", "")][0], "bks_distance": bks[i.lower().replace("hetiuy", "")][1]}
        for al in algos:
            rr = list(by.get((i, al), {}).values())
            if rr:
                row[f"{SHORT[al]}_score"] = float(np.mean([r["score"] for r in rr]))
                row[f"{SHORT[al]}_veh"] = float(np.mean([r["vehicles"] for r in rr]))
                row[f"{SHORT[al]}_gap"] = float(np.mean([r["gap_pct"] for r in rr]))
                row[f"{SHORT[al]}_ips"] = float(np.mean([r["ips"] for r in rr]))
        per.append(row)
    keys = sorted({k for r in per for k in r}, key=lambda k: (k not in ("instance", "group", "class", "n", "bks_vehicles", "bks_distance"), k))
    with open(os.path.join(a.out, "per_instance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(per)
    out.append("\n## Correlation of the P − B1 score difference with instance properties (Spearman, per instance)\n")
    for g in GROUPS + ["ALL"]:
        pp = [r for r in per if (g == "ALL" or r["group"] == g) and "P_score" in r and "B1_score" in r]
        if len(pp) < 5:
            continue
        d = [r["P_score"] - r["B1_score"] for r in pp]
        c1 = stats.spearmanr(d, [r["B1_veh"] for r in pp]); c2 = stats.spearmanr(d, [r["n"] for r in pp])
        J["main"].setdefault("corr", []).append({"group": g, "routes_rho": float(c1.statistic), "routes_p": float(c1.pvalue),
                                                 "n_rho": float(c2.statistic), "n_p": float(c2.pvalue)})
        out.append(f"- **{GNAME[g]}**: with number of routes rho = {c1.statistic:+.2f} (p = {fp(c1.pvalue)}); "
                   f"with number of requests rho = {c2.statistic:+.2f} (p = {fp(c2.pvalue)})")

    # ------------------------------------------------------------------ T6: deadline block (RQ3)
    if D:
        out.append("\n## Table 6. Short deadlines (RQ3): fixed budget of 10 s and 20 s per instance\n")
        t = []
        for bud in (10.0, 20.0):
            byd = collections.defaultdict(dict)
            for r in D:
                if r["budget"] == bud:
                    byd[(r["instance"], r["algo"])][r["seed"]] = r
            for g in GROUPS + ["ALL"]:
                c = compare(byd, insts_of[g], "P_hgnn", a.ref)
                if not c:
                    continue
                rs = [r for i in insts_of[g] for al in ("P_hgnn", a.ref) for r in byd.get((i, al), {}).values()]
                ok = sum(r["checker_ok"] for r in rs); uns = sum(r["unserved"] > 0 for r in rs)
                late = max(r["elapsed"] for r in rs) - bud
                J["deadline"].setdefault("rows", []).append({"budget": bud, "group": g, **c, "feasible": ok, "runs": len(rs),
                                                             "unserved_runs": uns, "max_overrun_s": float(late)})
                t.append([f"{bud:.0f} s", GNAME[g], c["n"], f2(c["mean_a"]), f2(c["mean_b"]), f"{c['diff']:+.2f}",
                          f"{c['win']} / {c['tie']} / {c['loss']}", fp(c["p"]), f2(c["a12"]), f"{ok}/{len(rs)}", uns, f"{late:.2f}"])
        out.append(md_table(["Budget", "Group", "Inst.", "P mean score", "B1 mean score", "Mean diff.", "Win / tie / loss",
                             "Wilcoxon p", "A12", "Feasible runs", "Runs with unserved", "Max overrun (s)"], t))

    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(J, f, indent=1)
    print("\n".join(out))

    # ------------------------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "grid.color": "#e6e5e0", "axes.grid": True,
                         "grid.linewidth": 0.6, "legend.frameon": False})
    ialgos = [al for al in algos if al != "B0_regret2"]

    # F1 convergence: mean incumbent score over normalised time
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, g in zip(axs, GROUPS):
        for al in ialgos:
            cs = [np.mean([r["curve"] for r in by[(i, al)].values()], axis=0) for i in insts_of[g] if (i, al) in by]
            if cs:
                ax.plot(grid * 100, np.mean(cs, axis=0), color=COLORS[al], lw=1.6, label=LABEL[al])
        ax.set_title(GNAME[g], fontsize=9, loc="left"); ax.set_xlabel("share of the time budget used (%)")
        ax.set_yscale("log")
    axs[0].set_ylabel("mean score of the best solution so far")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(h), fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(a.out, "fig_convergence.png"), dpi=200); plt.close(fig)

    # F2 per-instance difference P - B1 against the number of routes
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, g in zip(axs, GROUPS):
        pp = [r for r in per if r["group"] == g and "P_score" in r and "B1_score" in r]
        d = np.array([r["P_score"] - r["B1_score"] for r in pp]); x = np.array([r["B1_veh"] for r in pp])
        lim = np.percentile(np.abs(d), 95) * 1.3 + 1e-6 if len(d) else 1
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.scatter(x, np.clip(d, -lim, lim), s=14, color=COLORS["P_hgnn"], alpha=0.75, linewidths=0)
        ax.set_ylim(-lim * 1.08, lim * 1.08)
        ax.set_title(f"{GNAME[g]}: {int((d < -1e-6).sum())} better, {int((np.abs(d) <= 1e-6).sum())} equal, {int((d > 1e-6).sum())} worse",
                     fontsize=8.5, loc="left")
        ax.set_xlabel("number of routes (classical ALNS)")
    axs[0].set_ylabel("score of HGNN-ALNS minus classical ALNS\n(below 0 = HGNN-ALNS better)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_diff_vs_routes.png"), dpi=200); plt.close(fig)

    # F3 score by class for all algorithms
    order = sorted(cls, key=lambda k: (GROUPS.index(k[0]), k[1]))
    fig, ax = plt.subplots(figsize=(10, 3.4))
    x = np.arange(len(order)); wd = 0.8 / len(ialgos)
    for j, al in enumerate(ialgos):
        v = [np.mean([np.mean([r["score"] for r in by[(i, al)].values()]) for i in cls[k] if (i, al) in by] or [np.nan]) for k in order]
        ax.bar(x - 0.4 + wd * (j + 0.5), v, wd * 0.88, color=COLORS[al], label=LABEL[al])
    ax.set_xticks(x); ax.set_xticklabels([k[1].replace("DARP ", "H-DARP\n").replace("<=", "≤ ").replace(">", "> ") if k[0] == "DARP" else
                                       f"{k[1]}\n{'100' if k[0] == 'LL100' else '200'}" for k in order], fontsize=7.5)
    ax.grid(axis="x", visible=False); ax.set_axisbelow(True)
    ax.set_ylabel("mean score (lower is better)"); ax.legend(ncol=3, fontsize=7.5, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_score_by_class.png"), dpi=200); plt.close(fig)

    # F4 iterations per second relative to classical ALNS
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(GROUPS)); gal = [al for al in ialgos if al != a.ref]; wd = 0.8 / max(1, len(gal))
    for j, al in enumerate(gal):
        v = []
        for g in GROUPS:
            rat = [np.mean([r["ips"] for r in by[(i, al)].values()]) / np.mean([r["ips"] for r in by[(i, a.ref)].values()])
                   for i in insts_of[g] if (i, al) in by and (i, a.ref) in by]
            v.append(np.mean(rat) if rat else np.nan)
        ax.bar(x - 0.4 + wd * (j + 0.5), v, wd * 0.88, color=COLORS[al], label=LABEL[al])
    ax.axhline(1, color="#52514e", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([GNAME[g] for g in GROUPS])
    ax.set_ylabel("iterations per second\nrelative to classical ALNS"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_speed.png"), dpi=200); plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
