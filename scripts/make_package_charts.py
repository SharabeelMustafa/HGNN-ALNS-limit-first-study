"""Performance and ablation charts for the research package (PNG 300 dpi + PDF). Data: 04_ablations/ablation_summary.json,
05_baselines/per_instance_benchmark.csv and published_malheiros2021_table6.csv. Existing analysis figures are copied next to them."""
import csv
import json
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PK = os.path.abspath(os.path.join(REPO, "..", "Research_Package"))
OUT = os.path.join(PK, "10_figures")
C = {"b1": "#1baf7a", "sp": "#c98a00", "hint": "#2a78d6", "hsp": "#6250d6", "oracle": "#d6409f", "first": "#eb6834", "pub": "#8a8984"}
INK, MUTED, GRID = "#1f1f1e", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984", "xtick.color": MUTED,
                     "ytick.color": MUTED, "grid.color": GRID, "axes.axisbelow": True, "grid.linewidth": 0.6, "legend.frameon": False, "font.family": "DejaVu Sans"})
G = [("LL100", "Li & Lim 100"), ("LL200", "Li & Lim 200"), ("DARP", "H-DARP")]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300); fig.savefig(os.path.join(OUT, name + ".pdf")); plt.close(fig)


def c1_overview(J):
    ov = {o["group"]: o for o in J["tables"]["overview"]}
    keys = [("Classical ALNS", "b1", "Classical ALNS"), ("Classical ALNS + set partitioning", "sp", "+ set partitioning"),
            ("ALNS + HGNN soft hint", "hint", "+ HGNN soft hint"), ("ALNS + HGNN soft hint + set partitioning", "hsp", "+ HGNN soft hint + set partitioning")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, (g, ttl) in zip(axs, G):
        v = [ov[g][k] for k, _, _ in keys]
        xs = np.arange(4)
        ax.bar(xs, v, 0.7, color=[C[c] for _, c, _ in keys])
        for x, y in zip(xs, v):
            ax.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=7, color=MUTED)
        ax.set_title(f"{ttl} ({ov[g]['n']} instances)", fontsize=8.5); ax.set_xticks([]); ax.grid(axis="y"); ax.set_ylim(0, max(v) * 1.16)
    axs[0].set_ylabel("mean score (lower is better)")
    h = [plt.Rectangle((0, 0), 1, 1, color=C[c]) for _, c, _ in keys]
    fig.legend(h, [lab for _, _, lab in keys], loc="lower center", ncol=4, fontsize=7.5, handlelength=1.0, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.09, 1, 1)); save(fig, "C1_benchmark_comparison")


def c2_ablation(J):
    rows = [("ablation2", "C2_co@hgnn_l02#pre", "B1_alns", "HGNN soft hint", "hint"), ("ablation3", "B1_sp", "B1_alns", "Set partitioning", "sp"),
            ("ablation1", None, None, "First method (route prior + pruning)", "first"), ("ablation4", "O_oracle@prior1", "B1_alns", "Oracle soft hint (upper limit)", "oracle")]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    ys = np.arange(len(rows))[::-1]
    for ax, (g, ttl) in zip(axs, G):
        for y, (tab, a, b, lab, col) in zip(ys, rows):
            if a is None:
                e = [x for x in J["tables"]["ablation1"] if x["group"] == g and x["variant"].startswith("First method, equal iterations")]
            else:
                e = [x for x in J["tables"][tab] if x["group"] == g and x["first"] == a and x["second"] == b]
            if not e:
                continue
            e = e[0]
            ax.barh(y, e["diff"], 0.62, color=C[col], zorder=2)
            star = "*" if e["p"] < 0.05 else ""
            ax.text(e["diff"] + (0.02 if e["diff"] >= 0 else -0.02) * 1, y, f" {e['diff']:+.2f}{star} ", va="center", ha="left" if e["diff"] >= 0 else "right", fontsize=7, color=MUTED)
        ax.axvline(0, color=MUTED, lw=0.8); ax.set_title(ttl, fontsize=8.5); ax.grid(axis="x")
        lo, hi = ax.get_xlim(); pad = 0.38 * (hi - lo); ax.set_xlim(lo - pad, hi + pad)
        ax.set_yticks(ys); ax.set_yticklabels([r[3] for r in rows]); ax.tick_params(axis="y", length=0, labelcolor=INK)
    fig.text(0.62, 0.02, "score difference to classical ALNS at equal iterations (left of 0 = better);  * p < 0.05", ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, "C2_ablation_effects")


def c3_published():
    R = [r for r in csv.DictReader(open(os.path.join(PK, "05_baselines", "per_instance_benchmark.csv"))) if r["benchmark"].startswith("H-DARP")]
    T = {r["instance"]: r for r in csv.DictReader(open(os.path.join(PK, "05_baselines", "published_malheiros2021_table6.csv")))}
    R.sort(key=lambda r: (int(r["n_requests"]), r["instance"]))
    lab = [r["instance"].lower().replace("hetiuy", "") for r in R]
    xs = np.arange(len(R))
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.plot(xs, [float(r["classical_alns_gap_pct"]) for r in R], "-o", ms=4, lw=1.4, color=C["b1"], label="Classical ALNS [ours]")
    ax.plot(xs, [float(r["alns_sp_gap_pct"]) for r in R], "-s", ms=3.5, lw=1.2, color=C["sp"], label="ALNS + set partitioning [ours]")
    if all(r["alns_hint_sp_gap_pct"] != "" for r in R):
        ax.plot(xs, [float(r["alns_hint_sp_gap_pct"]) for r in R], "-^", ms=3.5, lw=1.2, color=C["hsp"], label="ALNS + HGNN soft hint + set partitioning [ours]")
    ax.plot(xs, [float(T[k]["da_gap"]) for k in lab], "--D", ms=3, lw=1.1, color=C["pub"], label="Deterministic annealing [published]")
    ax.plot(xs, [float(T[k]["ilssp_gap"]) for k in lab], ":v", ms=3.5, lw=1.3, color=INK, label="ILS + set partitioning [published]")
    ax.set_xticks(xs); ax.set_xticklabels(lab, rotation=60, ha="right", fontsize=6.8); ax.grid(axis="y"); ax.set_ylim(0, None)
    ax.set_ylabel("gap to best-known cost (%)"); ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_xlabel("H-DARP instance (vehicles-requests), sorted by number of requests", fontsize=7.5)
    fig.tight_layout(); save(fig, "C3_hdarp_against_published")


if __name__ == "__main__":
    J = json.load(open(os.path.join(PK, "04_ablations", "ablation_summary.json")))
    c1_overview(J); c2_ablation(J); c3_published()
    for src, dst in (("results/followup_analysis/fig_components.png", "C4_first_method_components.png"), ("results/followup_analysis/fig_label_accuracy.png", "C5_label_accuracy.png"),
                     ("results/followup_analysis/fig_equal_iter.png", "C6_equal_time_vs_equal_iterations.png"), ("results/core_analysis/fig_convergence.png", "C7_convergence_main_experiment.png"),
                     ("results/core_analysis/fig_score_by_class.png", "C8_score_by_instance_class.png"), ("../Final_Paper/figures/fig_advice.png", "C9_sources_of_advice.png"),
                     ("../Final_Paper/figures/fig_advice.pdf", "C9_sources_of_advice.pdf"), ("../Final_Paper/figures/fig_auc.png", "C10_predictability_of_corouting.png"),
                     ("../Final_Paper/figures/fig_auc.pdf", "C10_predictability_of_corouting.pdf")):
        p = os.path.join(REPO, src)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(OUT, dst))
    print(sorted(x for x in os.listdir(OUT) if x.startswith("C") and x.endswith(".png")))
