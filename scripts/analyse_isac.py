"""E11: is the best ALNS configuration instance-specific?

Data: 6 configurations (removal share xi x noise eta) x 3 seeds on the 50 tuning instances, with iteration budgets equal
to what each configuration reaches in the same time. For each family:
  global  : the configuration with the best mean over all instances (chosen on two seeds), evaluated on the third seed
  specific: for each instance the configuration that is best on two seeds, evaluated on the third seed
The difference specific - global (three rotations of the held-out seed) is the room for a per-instance choice that a
perfect predictor of "best on two seeds" could use. Scores are relative to the default configuration (positive = worse).
"""
import collections
import json
import os

import numpy as np
from scipy import stats

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT = {"gen-lilim": "B1_alns@xi0.4_eta0.025", "gen-hdarp": "B1_alns@xi0.2_eta0.0"}


def main():
    rows = [json.loads(l) for l in open(os.path.join(REPO, "results", "pilot_isac.jsonl"))]
    out = ["# E11: instance-specific configuration\n", f"Runs: {len(rows)}; feasible: {all(r['checker_ok'] for r in rows)}.\n"]
    for fam, lex in (("gen-lilim", True), ("gen-hdarp", False)):
        by = collections.defaultdict(dict)
        for r in rows:
            if r["family"] == fam:
                by[r["instance"]][(r["algo"], r["seed"])] = r
        algos = sorted({r["algo"] for r in rows if r["family"] == fam})
        insts = [i for i, v in by.items() if len(v) == len(algos) * 3 and all(x["unserved"] == 0 for x in v.values())]
        if not insts:
            continue
        S = np.zeros((len(insts), len(algos), 3))               # score relative to the default configuration, seed mean
        for a, i in enumerate(insts):
            ref_v = np.mean([by[i][(DEFAULT[fam], s)]["vehicles"] for s in range(3)])
            ref_d = np.mean([by[i][(DEFAULT[fam], s)]["distance"] for s in range(3)])
            for b, al in enumerate(algos):
                for s in range(3):
                    r = by[i][(al, s)]
                    S[a, b, s] = (100.0 * (r["vehicles"] - ref_v) if lex else 0.0) + 100.0 * (r["distance"] - ref_d) / ref_d
        out.append(f"\n## {'Li & Lim style' if lex else 'H-DARP style'} ({len(insts)} instances)\n")
        out.append("| Configuration | Mean score vs default | Instances where it is the best (3-seed mean) |\n|---|---|---|")
        m3 = S.mean(2)
        best = m3.argmin(1)
        for b, al in enumerate(algos):
            out.append(f"| {al.split('@')[1]}{' (default)' if al == DEFAULT[fam] else ''} | {m3[:, b].mean():+.2f} | {int((best == b).sum())} |")
        g, sp = [], []
        for h in range(3):                                        # held-out seed
            tr = [s for s in range(3) if s != h]
            mt = S[:, :, tr].mean(2)
            gb = mt.mean(0).argmin()
            g.append(S[:, gb, h]); sp.append(S[np.arange(len(insts)), mt.argmin(1), h])
        g, sp = np.mean(g, 0), np.mean(sp, 0)
        d = sp - g
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        naive = m3.min(1).mean() - m3.mean(0).min()
        out.append(f"\n- Global best configuration, held-out seed: {g.mean():+.2f}")
        out.append(f"- Per-instance best configuration (chosen on two seeds), held-out seed: {sp.mean():+.2f}")
        out.append(f"- **Room for a per-instance choice: {d.mean():+.2f}** (negative = better; {int((d < 0).sum())} better / {int((d > 0).sum())} worse, Wilcoxon p = {p:.3f})")
        out.append(f"- For comparison, the biased estimate without a held-out seed (best of 6 on the same seeds): {naive:+.2f}")
    os.makedirs(os.path.join(REPO, "results", "pilot_analysis"), exist_ok=True)
    open(os.path.join(REPO, "results", "pilot_analysis", "isac.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
