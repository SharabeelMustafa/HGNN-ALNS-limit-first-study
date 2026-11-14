"""Figures for the paper (single-column width). Data: results/pilot_headroom.jsonl and results/co_eval/v2.json."""
import collections
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.abspath(os.path.join(REPO, "..", "Final_Paper", "figures"))
C = {"oracle": "#d6409f", "learned": "#2a78d6", "free": "#eda100", "classical": "#1baf7a", "ceiling": "#8a8984"}
INK, MUTED, GRID = "#1f1f1e", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "xtick.color": MUTED, "ytick.color": INK, "grid.color": GRID, "axes.axisbelow": True, "grid.linewidth": 0.6,
                     "legend.frameon": False, "font.family": "DejaVu Sans"})


def diffs(fam, al, lex):
    rows = [json.loads(l) for l in open(os.path.join(REPO, "results", "pilot_headroom.jsonl"))]
    by = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        if r["family"] == fam:
            by[r["instance"]][r["algo"]][r["seed"]] = r
    d = []
    for i, v in by.items():
        if al not in v or "B1_alns" not in v:
            continue
        a, b = v[al], v["B1_alns"]
        sa = sorted(a) if al == "REF_long" else sorted(set(a) & set(b))
        sb = sorted(b) if al == "REF_long" else sa
        if not sa or any(a[s]["unserved"] for s in sa) or any(b[s]["unserved"] for s in sb):
            continue
        va, vb = np.mean([a[s]["vehicles"] for s in sa]), np.mean([b[s]["vehicles"] for s in sb])
        da, db = np.mean([a[s]["distance"] for s in sa]), np.mean([b[s]["distance"] for s in sb])
        d.append((100.0 * (va - vb) if lex else 0.0) + 100.0 * (da - db) / db)
    return np.array(d)


def boot(d, n=4000, seed=1):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, size=(n, len(d)), replace=True).mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def fig_advice():
    rows = [("4x run, strong prior", "O_oracle@prior1", "oracle"),
            ("4x run, weak prior", "O_oracle@prior", "oracle"),
            ("1x run", "O_ref1", "oracle"),
            ("0.6x run", "O_ref06", "oracle"),
            ("Mean of three 4x runs", "O_soft@prior1", "oracle"),
            ("HGNN, weak prior", "C2_co@hgnn_l02", "learned"),
            ("HGNN, strong prior", "C2_co@hgnn_l1", "learned"),
            ("First method", "P_hgnn", "learned"),
            ("Static relatedness", "C_shaw@prior1", "free"),
            ("4x iterations, no advice", "REF_long", "classical")]
    fig, ax = plt.subplots(figsize=(3.45, 3.1))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, al, cat) in zip(ys, rows):
        d = diffs("gen-hdarp", al, False)
        m = d.mean(); lo, hi = boot(d)
        ax.barh(y, m, 0.62, color=C[cat], zorder=2)
        ax.plot([lo, hi], [y, y], color=INK, lw=0.9, zorder=3)
        if m >= 0:
            ax.text(hi + 0.2, y, f"{m:+.2f}", va="center", ha="left", fontsize=6.8, color=MUTED)
        else:
            ax.text(lo - 0.2, y, f"{m:+.2f}", va="center", ha="right", fontsize=6.8, color=MUTED)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows]); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("cost difference to classical ALNS (%)")
    ax.grid(axis="x"); ax.set_xlim(-4.6, 12.4); ax.set_xticks([-2, 0, 2, 4, 6, 8])
    h = [plt.Rectangle((0, 0), 1, 1, color=C[k]) for k in ("oracle", "learned", "free", "classical")]
    fig.legend(h, ["advice read from search runs", "learned model", "rule without learning", "more search"], fontsize=6.6, loc="lower center",
               ncol=2, handlelength=1.0, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.40, right=0.985, top=0.985, bottom=0.25)
    fig.savefig(os.path.join(OUT, "fig_advice.pdf")); fig.savefig(os.path.join(OUT, "fig_advice.png"), dpi=300); plt.close(fig)


def fig_auc():
    d = json.load(open(os.path.join(REPO, "results", "co_eval", "v2.json")))
    rows = [("Other 4x runs", "other long runs (ceiling)", "ceiling"), ("One 1x run", "1x run (other seed)", "oracle"),
            ("One 0.6x run", "0.6x run (other seed)", "oracle"), ("HGNN, 4x teacher", "teacher-trained_hgnn", "learned"),
            ("Hom. GNN, 4x teacher", "teacher-trained_hom", "learned"), ("No message passing", "teacher-trained_mlp", "learned"),
            ("HGNN, short teacher", "old HGNN co-routing head", "learned"), ("Static relatedness", "static relatedness (free)", "free")]
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.2), sharey=True)
    ys = np.arange(len(rows))[::-1]
    for ax, fam, ttl in zip(axs, ("hdarp", "lilim"), ("H-DARP style", "Li & Lim style")):
        for y, (lab, key, cat) in zip(ys, rows):
            v = d[fam]["predictors"][key]["auc"]
            ax.barh(y, v - 0.5, 0.62, left=0.5, color=C[cat], zorder=2)
            ax.text(v + 0.012, y, f"{v:.2f}", va="center", ha="left", fontsize=6.5, color=MUTED)
        ax.set_xlim(0.5, 1.0); ax.set_xticks([0.5, 0.7, 0.9]); ax.grid(axis="x"); ax.set_title(ttl, fontsize=7.5, pad=3)
        ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows]); ax.tick_params(axis="y", length=0)
    fig.text(0.67, 0.02, "AUC (0.5 = chance)", ha="center", fontsize=7.2)
    fig.subplots_adjust(left=0.36, right=0.985, top=0.92, bottom=0.17, wspace=0.12)
    fig.savefig(os.path.join(OUT, "fig_auc.pdf")); fig.savefig(os.path.join(OUT, "fig_auc.png"), dpi=300); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_advice(); fig_auc()
    print("wrote", OUT)
