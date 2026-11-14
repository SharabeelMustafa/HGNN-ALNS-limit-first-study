"""Baseline comparison for the research package: our variants against best-known values and published methods.

Sources are kept apart: (a) reproduced by our implementation, (b) best-known values from official tables, (c) numbers taken from published
papers, obtained on other computers. Reads ../Research_Package/05_baselines/per_instance_benchmark.csv (from analyse_final.py) and the PDF of
Malheiros et al. (2021) for their Table 6.
"""
import csv
import os
import re
import subprocess

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PK = os.path.abspath(os.path.join(REPO, "..", "Research_Package", "05_baselines"))
PDF = os.path.abspath(os.path.join(REPO, "..", "research_kb", "papers", "malheiros2021_mdhdarp.pdf"))


def table6():
    txt = subprocess.run(["pdftotext", "-layout", PDF, "-"], capture_output=True, text=True).stdout
    rows = {}
    for line in txt.split("\n"):
        m = re.match(r"\s*(a\d+-\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if m:
            v = [float(x) for x in m.groups()[1:]]
            rows[m.group(1)] = dict(bks=v[0], da_best=v[1], da_avg=v[2], da_gap=v[3], da_cpu=v[4], ilssp_t_best=v[5], ilssp_t_avg=v[6], ilssp_t_gap=v[7],
                                    ilssp_best=v[8], ilssp_avg=v[9], ilssp_gap=v[10], ilssp_cpu=v[11])
    return rows


def main():
    T = table6()
    with open(os.path.join(PK, "published_malheiros2021_table6.csv"), "w", newline="") as f:
        w = csv.writer(f); keys = list(next(iter(T.values())).keys()); w.writerow(["instance"] + keys)
        for k, v in T.items():
            w.writerow([k] + [v[x] for x in keys])
    R = list(csv.DictReader(open(os.path.join(PK, "per_instance_benchmark.csv"))))
    fl = lambda x: float(x) if x not in ("", None) else float("nan")      # noqa: E731
    V = [("classical_alns", "Classical ALNS"), ("alns_sp", "ALNS + set partitioning"), ("alns_hint", "ALNS + HGNN soft hint"),
         ("alns_hint_sp", "ALNS + HGNN soft hint + set partitioning"), ("alns_oracle_hint", "ALNS + oracle soft hint (upper limit)"), ("first_method", "First method")]
    out = ["# Baseline comparison\n",
           "Three kinds of numbers appear here and must not be mixed:\n",
           "- **[ours]** reproduced by our implementation on our laptop (Intel i7-12700H, one thread per run), same seeds and same iterations for all our variants;",
           "- **[BKS]** best-known values from the official tables (SINTEF for Li & Lim; Malheiros et al. 2021 for H-DARP). They are not proven optima;",
           "- **[published]** results printed in a paper, obtained with other code (C++), other computers and other stopping rules. They show the level of the state of the art. "
           "They are **not** a controlled comparison with our runs.\n",
           "## 1. Our variants: summary per benchmark [ours]\n",
           "Mean over instances of the mean over seeds. Gap = distance or cost gap to the best-known value (can be negative on Li & Lim when more vehicles are used). "
           "Time = mean real run time with 12 runs in parallel.\n",
           "| Benchmark | Variant | Inst. | Mean score | Mean extra vehicles | Mean gap % | Mean run time (s) | Feasible runs |", "|---|---|---|---|---|---|---|---|"]
    for g in ("Li & Lim 100", "Li & Lim 200", "H-DARP (Malheiros)"):
        rs = [r for r in R if r["benchmark"] == g]
        for key, lab in V:
            ok = [r for r in rs if r[key + "_score"] != ""]
            if not ok:
                continue
            ev = np.mean([fl(r[key + "_mean_vehicles"]) - fl(r["bks_vehicles"]) for r in ok]) if g != "H-DARP (Malheiros)" else float("nan")
            fe = [r[key + "_feasible_runs"].split("/") for r in ok]
            out.append(f"| {g} | {lab} | {len(ok)} | {np.mean([fl(r[key + '_score']) for r in ok]):.2f} | {'–' if np.isnan(ev) else format(ev, '+.3f')} | "
                       f"{np.mean([fl(r[key + '_gap_pct']) for r in ok]):+.2f} | {np.mean([fl(r[key + '_mean_time_s']) for r in ok]):.1f} | {sum(int(a) for a, b in fe)}/{sum(int(b) for a, b in fe)} |")
    # ---- H-DARP per instance against published
    out += ["\n## 2. Multi-depot heterogeneous DARP: per instance, against published methods\n",
            "Published columns are copied from Table 6 of Malheiros et al. (2021): DA = deterministic annealing of Braekers et al. (2014) as re-run by Malheiros et al.; "
            "ILS-SP = their iterated local search with set partitioning. Both: C++, one thread of an Intel i5-9600KF (3.7 GHz), 10 runs, gap of the average to the best-known value. "
            "Our columns: gap of the mean over 5 seeds, iteration budget equal to what our classical ALNS reaches in 0.5 s per request.\n",
            "| Instance | Requests | Vehicles | BKS | DA gap % [published] | DA time (s) | ILS-SP gap % [published] | ILS-SP time (s) | Classical ALNS gap % [ours] | + set part. gap % [ours] | + hint + set part. gap % [ours] | Our time (s) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    hd = [r for r in R if r["benchmark"] == "H-DARP (Malheiros)"]
    hd.sort(key=lambda r: (int(r["n_requests"]), r["instance"]))
    acc = {k: [] for k in ("da", "da_t", "ils", "ils_t", "b1", "sp", "hsp", "t")}
    for r in hd:
        key = r["instance"].lower().replace("hetiuy", "")
        p = T.get(key)
        if not p:
            continue
        g = lambda k: fl(r[k + "_gap_pct"])      # noqa: E731
        out.append(f"| {key} | {r['n_requests']} | {r['fleet_size']} | {p['bks']:.2f} | {p['da_gap']:.2f} | {p['da_cpu']:.1f} | {p['ilssp_gap']:.2f} | {p['ilssp_cpu']:.1f} | "
                   f"{g('classical_alns'):.2f} | {g('alns_sp'):.2f} | {'–' if np.isnan(g('alns_hint_sp')) else format(g('alns_hint_sp'), '.2f')} | {fl(r['classical_alns_mean_time_s']):.1f} |")
        for k, v in (("da", p["da_gap"]), ("da_t", p["da_cpu"]), ("ils", p["ilssp_gap"]), ("ils_t", p["ilssp_cpu"]), ("b1", g("classical_alns")), ("sp", g("alns_sp")),
                     ("hsp", g("alns_hint_sp")), ("t", fl(r["classical_alns_mean_time_s"]))):
            acc[k].append(v)
    m = {k: np.nanmean(v) for k, v in acc.items()}
    out.append(f"| **Mean** | | | | **{m['da']:.2f}** | {m['da_t']:.1f} | **{m['ils']:.2f}** | {m['ils_t']:.1f} | **{m['b1']:.2f}** | **{m['sp']:.2f}** | **{m['hsp']:.2f}** | {m['t']:.1f} |")
    out.append("\nReading: the published methods are about 5 percentage points closer to the best-known values than our ALNS, in a similar amount of time. "
               "Our implementation makes about 2,000-4,000 iterations per run; published ALNS and local-search codes make an order of magnitude more.\n")
    # ---- Li & Lim cumulative
    out += ["## 3. Li & Lim PDPTW: cumulative number of vehicles and distance\n",
            "Standard form of reporting for this benchmark: sum over the instances of a set. Our values: mean over 5 seeds per instance, then summed.\n",
            "| Set | Source | Cumulative vehicles | Cumulative distance | Note |", "|---|---|---|---|---|"]
    for g, pub in (("Li & Lim 100", [("Ropke & Pisinger 2006, best of 10 runs [published]", 402, 58060, "25,000 iterations, 66 s per run on a 1.5 GHz Pentium IV"),
                                     ("Ropke & Pisinger 2006, average of the runs [published]", 403, 58249, "same")]),
                   ("Li & Lim 200", [("Ropke & Pisinger 2006, best of 10 runs [published]", 606, 180931, "25,000 iterations, 264 s per run"),
                                     ("Ropke & Pisinger 2006, average of the runs [published]", 608, 181707, "same")])):
        rs = [r for r in R if r["benchmark"] == g]
        out.append(f"| {g} | Best-known solutions, SINTEF [BKS] | {sum(fl(r['bks_vehicles']) for r in rs):.0f} | {sum(fl(r['bks_objective']) for r in rs):.0f} | {len(rs)} instances |")
        for lab, v, d, note in pub:
            out.append(f"| {g} | {lab} | {v} | {d} | {note} |")
        for key, lab in V[:4]:
            ok = [r for r in rs if r[key + "_score"] != ""]
            if len(ok) == len(rs):
                out.append(f"| {g} | {lab} [ours] | {sum(fl(r[key + '_mean_vehicles']) for r in ok):.1f} | {sum(fl(r[key + '_mean_objective']) for r in ok):.0f} | "
                           f"mean of 5 seeds; {np.mean([fl(r[key + '_mean_time_s']) for r in ok]):.0f} s per run |")
    out.append("\nThe Ropke & Pisinger numbers are from the thesis version of their paper (Tables 7-9), as collected in our literature notes; "
               "the best-known totals have improved since 2006, so their 2006 results are above today's best-known values on the 200 set. "
               "Other published methods (Curtois et al. 2018; Sartori & Buriol 2020) report matched and new best-known solutions with 5-15 minutes per run; "
               "we did not find their cumulative values in a source we could read.\n")
    out += ["## 4. Methods we did not run\n",
            "| Method | Why it is not in the tables as a reproduced baseline |", "|---|---|",
            "| Malheiros et al. ILS-SP, Braekers et al. DA | No public code that we could run; published values are given in Section 2 |",
            "| Curtois et al., Sartori & Buriol (PDPTW) | Different stopping rule (5-15 min); only published statements are used |",
            "| Google OR-Tools | Planned in the proposal; not run. It does not model the DARP ride-time and upgrade rules without extra work |",
            "| Neural constructive solvers | No published model for the multi-depot heterogeneous DARP or PDPTW with these constraints |"]
    open(os.path.join(PK, "BASELINE_COMPARISON.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out[:22])); print("...", len(T), "published rows parsed; H-DARP mean gaps:", {k: round(float(v), 2) for k, v in m.items()})


if __name__ == "__main__":
    main()
