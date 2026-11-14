"""Statistics, tables and convergence figures for an experiment results file (Final Methodology 10.3-10.5).

Score of a run (lower is better), relative to the best-known solution (BKS):
  Li & Lim (lexicographic): 100 * (vehicles - BKS vehicles) + % distance gap
  H-DARP (cost):            % cost gap
Primal integral: time-average of the score of the best solution found so far over the budget.

Usage:
  python scripts/analyse.py results/initial.jsonl --out results/initial_analysis --ref B1_alns
Outputs: summary.csv, per_instance.csv, stats.md, fig_convergence.png, fig_gap_per_instance.png
"""
import argparse
import collections
import csv
import json
import os

import numpy as np
from scipy import stats

COLORS = {"B1_alns": "#1baf7a", "P_hgnn": "#2a78d6", "B5_homgnn": "#eb6834", "A_INT2_prior": "#6250d6",
          "A_INT1_prune": "#eda100", "A_INT3_noGNNdestroy": "#e87ba4", "B0_regret2": "#8a8984",
          "B2_alns_uniform": "#008300", "A_ALNS3_rrt": "#e34948", "A_ALNS4_itersa": "#52514e"}
LABEL = {"B0_regret2": "Regret-2 construction (B0)", "B1_alns": "Classical ALNS (B1)", "P_hgnn": "HGNN-ALNS (P)",
         "B5_homgnn": "Homogeneous GNN-ALNS (B5)", "A_INT2_prior": "HGNN prior only (A-INT2)",
         "A_INT1_prune": "HGNN pruning only (A-INT1)", "A_INT3_noGNNdestroy": "HGNN, no GNN destroy (A-INT3)"}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip() and not l.startswith('{"_meta"')]


def score_of(r, veh, dist):
    gap = 100.0 * (dist - r["bks_distance"]) / r["bks_distance"]
    if r["bks_vehicles"] is not None:
        return 100.0 * (veh - r["bks_vehicles"]) + gap
    return gap


def primal_integral(r):
    """Average score over [0, budget] of the incumbent; 100 before the first feasible full solution."""
    T = r["budget"]
    pts = [(t, score_of(r, v, d)) for t, d, v, u in r["trajectory"] if u == 0]
    if not pts:
        return 100.0
    area, t_prev, s_prev = 0.0, 0.0, 100.0
    for t, s in pts:
        t = min(t, T)
        area += (t - t_prev) * min(s_prev, 100.0)
        t_prev, s_prev = t, s
    area += (T - t_prev) * min(s_prev, 100.0)
    return area / T


def a12(x, y):
    """Vargha-Delaney A12: probability that a run of x is better (lower) than a run of y."""
    gt = sum((xi < yj) + 0.5 * (xi == yj) for xi in x for yj in y)
    return gt / (len(x) * len(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="B1_alns")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = [r for r in load(a.results) if r.get("bks_distance")]
    for r in R:
        r["score"] = score_of(r, r["vehicles"], r["distance"]) if r["unserved"] == 0 else 1000.0
        r["pi"] = primal_integral(r)
        r["ips"] = r["iterations"] / max(r["elapsed"], 1e-9)
    algos = [x for x in LABEL if any(r["algo"] == x for r in R)] + \
        sorted({r["algo"] for r in R} - set(LABEL))
    insts = sorted({r["instance"] for r in R}, key=lambda i: (("het" in i), next(r["n"] for r in R if r["instance"] == i), i))
    by = collections.defaultdict(list)
    for r in R:
        by[(r["instance"], r["algo"])].append(r)

    # ---------------------------------------------------------------- per-instance table
    rows = []
    for i in insts:
        for al in algos:
            rr = by.get((i, al))
            if not rr:
                continue
            sc = [r["score"] for r in rr]
            rows.append({"instance": i, "n": rr[0]["n"], "family": "lilim" if rr[0]["bks_vehicles"] is not None else "hdarp",
                         "algo": al, "runs": len(rr), "vehicles_mean": np.mean([r["vehicles"] for r in rr]),
                         "bks_vehicles": rr[0]["bks_vehicles"], "gap_mean": np.mean([r["gap_pct"] for r in rr]),
                         "gap_best": min(r["gap_pct"] for r in rr), "score_mean": np.mean(sc), "score_sd": np.std(sc),
                         "primal_integral": np.mean([r["pi"] for r in rr]), "iters_per_s": np.mean([r["ips"] for r in rr]),
                         "guide_setup_s": np.mean([r["t_guide_setup"] for r in rr]),
                         "checker_ok": all(r["checker_ok"] for r in rr)})
    with open(os.path.join(a.out, "per_instance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---------------------------------------------------------------- summary + statistics
    lines = ["| Algorithm | Family | Runs | Mean score | Mean gap % | Mean primal integral | Iterations/s | Wins / ties / losses vs ref (instances) | Wilcoxon p (paired runs) | A12 vs ref |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    summary = []
    for fam in ("lilim", "hdarp", "all"):
        for al in algos:
            rs = [r for r in R if r["algo"] == al and (fam == "all" or (r["bks_vehicles"] is not None) == (fam == "lilim"))]
            if not rs:
                continue
            # paired by (instance, seed) against the reference algorithm
            ref = {(r["instance"], r["seed"]): r for r in R if r["algo"] == a.ref}
            pairs = [(r["score"], ref[(r["instance"], r["seed"])]["score"]) for r in rs if (r["instance"], r["seed"]) in ref]
            p = a12v = None
            wtl = ""
            if al != a.ref and pairs:
                d = np.array([x - y for x, y in pairs])
                p = stats.wilcoxon(d).pvalue if np.any(np.abs(d) > 1e-9) else 1.0
                a12v = a12([x for x, _ in pairs], [y for _, y in pairs])
                W = T = L = 0
                for i in {r["instance"] for r in rs}:
                    m1 = np.mean([r["score"] for r in by[(i, al)]])
                    m0 = np.mean([r["score"] for r in by.get((i, a.ref), [])] or [np.nan])
                    if np.isnan(m0):
                        continue
                    if abs(m1 - m0) < 1e-6:
                        T += 1
                    elif m1 < m0:
                        W += 1
                    else:
                        L += 1
                wtl = f"{W} / {T} / {L}"
            s = {"family": fam, "algo": al, "runs": len(rs), "score": np.mean([r["score"] for r in rs]),
                 "gap": np.mean([r["gap_pct"] for r in rs]), "pi": np.mean([r["pi"] for r in rs]),
                 "ips": np.mean([r["ips"] for r in rs]), "wtl": wtl, "p": p, "a12": a12v}
            summary.append(s)
            lines.append(f"| {LABEL.get(al, al)} | {fam} | {s['runs']} | {s['score']:.2f} | {s['gap']:.2f} | {s['pi']:.2f} | "
                         f"{s['ips']:.0f} | {wtl or '-'} | {'-' if p is None else f'{p:.3f}'} | {'-' if a12v is None else f'{a12v:.2f}'} |")
    # Friedman over instances (blocks) on mean score, iterative algorithms present on every instance
    it_algos = [al for al in algos if al != "B0_regret2" and all((i, al) in by for i in insts)]
    fr = None
    if len(it_algos) >= 3:
        M = np.array([[np.mean([r["score"] for r in by[(i, al)]]) for al in it_algos] for i in insts])
        fr = stats.friedmanchisquare(*M.T)
        ranks = np.mean([stats.rankdata(row) for row in M], axis=0)
        lines += ["", f"Friedman test over {len(insts)} instances ({', '.join(it_algos)}): "
                      f"chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.3f}. Mean ranks: " +
                  ", ".join(f"{al} {rk:.2f}" for al, rk in zip(it_algos, ranks))]
    with open(os.path.join(a.out, "stats.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("\n".join(lines))

    # ---------------------------------------------------------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "grid.color": "#e6e5e0", "axes.grid": True, "grid.linewidth": 0.6})
    ialgos = [al for al in algos if al != "B0_regret2"]
    # score per instance: bars = mean over seeds, dots = individual seeds; Li & Lim and H-DARP on separate scales
    fams = [("Li & Lim (score: % above BKS distance, +100 per extra vehicle)", [i for i in insts if "het" not in i]),
            ("H-DARP (score: % above BKS cost)", [i for i in insts if "het" in i])]
    fams = [f for f in fams if f[1]]
    fig, axs = plt.subplots(1, len(fams), figsize=(10, 3.6), gridspec_kw={"width_ratios": [len(f[1]) for f in fams]})
    rng = np.random.default_rng(0)
    for ax, (ttl, ii) in zip(np.atleast_1d(axs), fams):
        x = np.arange(len(ii))
        wd = 0.8 / len(ialgos)
        for j, al in enumerate(ialgos):
            xs = x - 0.4 + wd * (j + 0.5)
            m = [np.mean([r["score"] for r in by[(i, al)]]) if (i, al) in by else np.nan for i in ii]
            ax.bar(xs, m, wd * 0.88, color=COLORS.get(al, "#999"), label=LABEL.get(al, al))
            for xx, i in zip(xs, ii):
                v = [r["score"] for r in by.get((i, al), [])]
                ax.scatter(xx + rng.uniform(-wd * 0.2, wd * 0.2, len(v)), v, s=6, color="#0b0b0b", zorder=3, linewidths=0)
        ax.set_xticks(x)
        ax.set_xticklabels([i.replace("hetIUY", "") for i in ii])
        ax.set_title(ttl, fontsize=8.5, loc="left")
    np.atleast_1d(axs)[0].set_ylabel("score (lower is better)")
    h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(ialgos), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(a.out, "fig_gap_per_instance.png"), dpi=200)
    plt.close(fig)
    # convergence: median incumbent score over time for the H-DARP instances and the 200-request Li & Lim ones
    show = [i for i in insts if "het" in i] + [i for i in insts if "het" not in i and next(r["n"] for r in R if r["instance"] == i) >= 100]
    fig, axs = plt.subplots(1, len(show), figsize=(2.1 * len(show), 2.8))
    for ax, i in zip(np.atleast_1d(axs), show):
        for al in ialgos:
            rr = by.get((i, al))
            if not rr:
                continue
            T = rr[0]["budget"]
            grid = np.linspace(0, T, 200)
            curves = []
            for r in rr:
                pts = [(t, score_of(r, v, d)) for t, d, v, u in r["trajectory"] if u == 0]
                c = np.full(len(grid), np.nan)
                for t, s in pts:
                    c[grid >= t] = s
                curves.append(c)
            with np.errstate(all="ignore"):
                import warnings
                warnings.simplefilter("ignore")
                med = np.nanmedian(np.array(curves), axis=0)
            ax.step(grid, med, where="post", color=COLORS.get(al, "#999"), lw=1.6, label=LABEL.get(al, al))
        ax.set_title(i.replace("hetIUY", ""), fontsize=9)
        ax.set_xlabel("time (s)")
        lo = np.nanmin([np.nanmin([s["score_mean"] for s in rows if s["instance"] == i])])
        ax.set_ylim(max(lo - 2, -1), lo + 25)
    np.atleast_1d(axs)[0].set_ylabel("median incumbent score")
    h, l = np.atleast_1d(axs)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(h), frameon=False, fontsize=8)
    fig.suptitle("Convergence: median over seeds of the best solution found so far (lower is better)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(a.out, "fig_convergence.png"), dpi=200)
    plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
