"""Training graphs and model-evaluation figures.

  fig_training_loss.png     train / validation loss per epoch, 4 models (2 families x HGNN / homogeneous GNN)
  fig_training_metrics.png  validation top-1 accuracy and recall@5 per epoch
  fig_hdarp_v1_v2.png       H-DARP training run v1 (n<=100 data) vs v2 (n<=200 data): validation curves
  fig_ranking_baselines.png held-out ranking quality: random / nearest-route / homogeneous GNN / HGNN
  fig_tuning.png            two-stage tuning scores (stage A: classical ALNS; stage B: HGNN guidance)

Usage: python scripts/make_figures.py --out results/figures
"""
import argparse
import collections
import json
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#8a8984"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GREY, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "grid.color": "#e6e5e0", "axes.grid": True, "grid.linewidth": 0.6, "legend.frameon": False})
FAM = {"lilim": "Li & Lim PDPTW (homogeneous fleet)", "hdarp": "H-DARP (heterogeneous fleet)"}


def hist(path):
    if path.endswith(".json"):
        return json.load(open(path))
    out = []
    for line in open(path):
        if line.startswith("{\"epoch\""):
            out.append(json.loads(line))
    return out


def training(models, out):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.1), sharey=False)
    for ax, fam in zip(axs, ("lilim", "hdarp")):
        for m, c, lab in (("hgnn", BLUE, "HGNN"), ("homgnn", ORANGE, "Homogeneous GNN")):
            h = hist(os.path.join(models, f"{fam}_{m}.pt.history.json"))
            e = [r["epoch"] + 1 for r in h]
            ax.plot(e, [r["train_loss"] for r in h], color=c, lw=1.6, label=f"{lab} - training")
            ax.plot(e, [r["loss"] for r in h], color=c, lw=1.6, ls="--", label=f"{lab} - validation")
        ax.set_title(FAM[fam], fontsize=9, loc="left")
        ax.set_xlabel("epoch")
    axs[0].set_ylabel("loss  (route CE + 0.5 x co-routing focal)")
    axs[1].legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_training_loss.png"), dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 3.1))
    for ax, fam in zip(axs, ("lilim", "hdarp")):
        for m, c, lab in (("hgnn", BLUE, "HGNN"), ("homgnn", ORANGE, "Homogeneous GNN")):
            h = hist(os.path.join(models, f"{fam}_{m}.pt.history.json"))
            e = [r["epoch"] + 1 for r in h]
            ax.plot(e, [r["recall@5"] for r in h], color=c, lw=1.6, label=f"{lab} - recall@5")
            ax.plot(e, [r["top1"] for r in h], color=c, lw=1.6, ls="--", label=f"{lab} - top-1")
        ax.set_ylim(0, 1)
        ax.set_title(FAM[fam], fontsize=9, loc="left")
        ax.set_xlabel("epoch")
    axs[0].set_ylabel("validation accuracy")
    axs[1].legend(fontsize=7.5, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_training_metrics.png"), dpi=200)
    plt.close(fig)


def v1v2(logs, models, out):
    v1 = hist(os.path.join(logs, "train_hdarp_hgnn.log"))
    v2 = hist(os.path.join(models, "hdarp_hgnn.pt.history.json"))
    if not v1:
        return
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for h, c, lab in ((v1, GREY, "v1: trained on n <= 100 (validation n <= 100)"),
                      (v2, BLUE, "v2: trained on n <= 200 (validation n <= 200)")):
        e = [r["epoch"] + 1 for r in h]
        axs[0].plot(e, [r["train_loss"] for r in h], color=c, lw=1.6, label=lab + " - training")
        axs[0].plot(e, [r["loss"] for r in h], color=c, lw=1.6, ls="--", label=lab + " - validation")
        axs[1].plot(e, [r["recall@5"] for r in h], color=c, lw=1.6, label=lab)
    axs[0].set_ylabel("loss")
    axs[1].set_ylabel("validation recall@5")
    for ax in axs:
        ax.set_xlabel("epoch")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7)
    axs[1].set_ylim(0.5, 1)
    fig.suptitle("H-DARP HGNN: first training run (v1) vs retraining on larger instances (v2). "
                 "The validation sets differ, so see Fig. 7.1 of the report for a comparison on the same data", fontsize=8, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(out, "fig_hdarp_v1_v2.png"), dpi=200)
    plt.close(fig)


def ranking(path, out):
    R = json.load(open(path))
    groups = [("lilim/all", "Li & Lim\n(all val.)"), ("hdarp/n<=100", "H-DARP\nn <= 100"), ("hdarp/n>100", "H-DARP\nn > 100")]
    series = [("random", "Random ranking", GREY), ("nearest_route", "Nearest-route heuristic", YELLOW),
              ("hgnn_v1", "HGNN v1 (H-DARP only, before retraining)", VIOLET), ("homgnn", "Homogeneous GNN", ORANGE), ("hgnn", "HGNN", BLUE)]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.3))
    for ax, met, ttl in ((axs[0], "top1", "top-1 accuracy"), (axs[1], "recall@5", "recall@5 (true route kept by top-5 pruning)")):
        x = np.arange(len(groups))
        wd = 0.8 / len(series)
        for j, (key, lab, c) in enumerate(series):
            v = [R[g].get(key, {}).get(met, np.nan) for g, _ in groups]
            b = ax.bar(x - 0.4 + wd * (j + 0.5), v, wd * 0.88, color=c, label=lab)
            for rect, val in zip(b, v):
                if not np.isnan(val):
                    ax.text(rect.get_x() + rect.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=5.5, color="#52514e")
        ax.set_xticks(x)
        ax.set_xticklabels([g[1] for g in groups])
        ax.set_title(ttl, fontsize=9, loc="left")
        ax.set_ylim(0, 1.05)
    axs[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ranking_baselines.png"), dpi=200)
    plt.close(fig)


def tuning_scores(path):
    rows = [json.loads(l) for l in open(path) if l.strip() and not l.startswith('{"_meta"')]
    best = {}
    for r in rows:
        k = (r["family"], r["instance"])
        if r["unserved"] == 0:
            key = (r["vehicles"] if "lilim" in r["family"] else 0, r["distance"])
            best[k] = min(best.get(k, key), key)
    sc = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        k = (r["family"], r["instance"])
        if k not in best:            # no configuration served every request: carries no ranking information
            continue
        fam = "lilim" if "lilim" in r["family"] else "hdarp"
        bv, bd = best[k]
        s = 1000.0 if r["unserved"] else (100.0 * (r["vehicles"] - bv) if fam == "lilim" else 0) + 100 * (r["distance"] - bd) / bd
        sc[fam][r["algo"].split("@")[1]].append(s)
    return {f: {a: float(np.mean(v)) for a, v in d.items()} for f, d in sc.items()}


def tuning(resdir, out):
    fig, axs = plt.subplots(2, 2, figsize=(9, 4.6))
    for col, (stage, ttl) in enumerate((("A", "Stage A: classical ALNS (removal share xi, noise eta)"),
                                        ("B", "Stage B: HGNN guidance (top-K, prior weight lambda)"))):
        p = os.path.join(resdir, f"tune_{stage}.jsonl")
        if not os.path.exists(p):
            continue
        S = tuning_scores(p)
        for row, fam in enumerate(("lilim", "hdarp")):
            ax = axs[row, col]
            items = sorted(S[fam].items(), key=lambda kv: kv[1])
            lab = [k.replace("eta", " eta ").replace("xi", "xi ").replace("_lam", "  lambda ").replace("K999", "K all")
                   .replace("K0.5", "K 50%").replace("K5", "K 5").replace("_", " ") for k, _ in items]
            v = [s for _, s in items]
            ax.barh(range(len(v)), v, color=[BLUE if i == 0 else GREY for i in range(len(v))], height=0.7)
            ax.set_yticks(range(len(v)))
            ax.set_yticklabels(lab, fontsize=7)
            ax.invert_yaxis()
            for i, s in enumerate(v):
                ax.text(s, i, f" {s:.2f}", va="center", fontsize=6.5, color="#52514e")
            ax.set_title(f"{ttl}\n{FAM[fam]}" if row == 0 else FAM[fam], fontsize=8, loc="left")
            ax.set_xlabel("mean relative score (lower is better)" if row == 1 else "")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_tuning.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    training(a.models, a.out)
    v1v2(a.logs, a.models, a.out)
    ranking(os.path.join(a.results, "ranking_eval.json"), a.out)
    tuning(a.results, a.out)
    print("wrote figures to", a.out)


if __name__ == "__main__":
    main()
