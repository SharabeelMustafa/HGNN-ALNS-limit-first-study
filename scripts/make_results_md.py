"""Build RESULTS.md from the analysis reports (re-run after every stage; nothing is typed by hand here)."""
import datetime
import json
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def section(path, start, level="## "):
    """Text of the section whose heading starts with `start` (up to the next heading of the same level)."""
    p = os.path.join(REPO, path)
    if not os.path.exists(p):
        return None
    lines = open(p).read().split("\n")
    out, on = [], False
    for ln in lines:
        if ln.startswith(level):
            if on:
                break
            on = ln[len(level):].startswith(start)
            if on:
                continue
        if on:
            out.append(ln)
    return "\n".join(out).strip() if out else None


def add(out, title, body, note=None):
    if body:
        out.append(f"\n### {title}\n")
        if note:
            out.append(note + "\n")
        out.append(body)


def main():
    out = ["# Results\n", f"Built by `scripts/make_results_md.py` on {datetime.datetime.now():%Y-%m-%d %H:%M} from the analysis reports. "
           "Score (lower is better): Li & Lim = 100 × extra vehicles + % distance gap to the best-known solution; H-DARP = % cost gap. "
           "Outcome marks: **POSITIVE** / **NEGATIVE** / **INFO** as in `EXPERIMENT_LOG.md`.\n",
           "\n## 1. Baseline: the first method (E04) — NEGATIVE\n",
           "Classical ALNS (B1) against HGNN-ALNS (P) and its variants; 140 benchmark instances, 0.5 s per request, one CPU thread. "
           "Full report: `results/core_analysis/report.md`.\n"]
    add(out, "Summary per benchmark", section("results/core_analysis/report.md", "Table 1."))
    add(out, "Statistical comparison against classical ALNS", section("results/core_analysis/report.md", "Table 3."))
    out.append("\n## 2. Diagnostics of the first method (E05–E07)\n\nFull report: `results/followup_analysis/report.md`.\n")
    add(out, "Equal iterations (E05) — NEGATIVE: the loss is not the time overhead", section("results/followup_analysis/report.md", "B."))
    add(out, "Components (E06) — INFO: pruning is the harmful part", section("results/followup_analysis/report.md", "D."))
    add(out, "Oracle guidance (E07a) — POSITIVE: the mechanism helps when the advice is good", section("results/followup_analysis/report.md", "E."))
    add(out, "What predicts the label best (E07b) — POSITIVE: exact insertion cost beats the network", section("results/followup_analysis/report.md", "G."))
    out.append("\n## 3. Pilots for the revised method (E08–E09), tuning instances only\n\nFull report: `results/pilot_analysis/report.md`. "
               "All runs use the same number of iterations as classical ALNS unless the row says otherwise.\n")
    for t in ("Li & Lim style", "H-DARP style"):
        add(out, t, section("results/pilot_analysis/report.md", t))
    q = os.path.join(REPO, "results", "pilot_analysis", "isac.md")
    if os.path.exists(q):
        out.append("\n### Per-instance configuration (E11) — NEGATIVE\n")
        out.append(re.sub(r"^## ", "#### ", open(q).read().split("\n", 1)[1], flags=re.M).strip())
    p = os.path.join(REPO, "results", "co_eval")
    if os.path.isdir(p):
        out.append("\n## 4. Offline check: predicting the co-routing of long runs (gate G3)\n\n"
                   "AUC and average precision over all request pairs, mean over the tuning instances; target = one held-out 4× run.\n")
        for f in sorted(os.listdir(p)):
            if not f.endswith(".json"):
                continue
            d = json.load(open(os.path.join(p, f)))
            for fam, r in d.items():
                rows = sorted(r["predictors"].items(), key=lambda kv: -kv[1]["ap"])
                out.append(f"\n### {f[:-5]} — {'Li & Lim style' if fam == 'lilim' else 'H-DARP style'} ({r['instances']} instances)\n")
                out.append("| Predictor | AUC | Average precision |\n|---|---|---|")
                out += [f"| {k} | {v['auc']:.3f} | {v['ap']:.3f} |" for k, v in rows]
    for extra in ("results/confirm_analysis/report.md", "../Research_Package/04_ablations/ABLATION_STUDIES.md"):
        q = os.path.join(REPO, extra)
        if os.path.exists(q):
            out.append("\n" + re.sub(r"^# ", "## ", open(q).read(), count=1, flags=re.M))
    open(os.path.join(REPO, "RESULTS.md"), "w").write("\n".join(out) + "\n")
    print("wrote RESULTS.md", len(out), "blocks")


if __name__ == "__main__":
    main()
