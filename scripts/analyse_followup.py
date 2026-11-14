"""Analysis of the follow-up diagnostic experiments (run after the main experiment).

Questions:
  A. Did the guidance help on the tuning instances (generated, in-distribution)?        [tune_A/B.jsonl]
  B. Is the loss of HGNN-ALNS caused by its time overhead? -> same number of iterations.   [followup.jsonl]
  C. How does the guidance change the search (undo rate, repeated solutions, improving steps)?
  D. Which component hurts (pruning, prior, GNN destroy)?  All instances, equal iterations.
  E. Can the mechanism help at all? Oracle guidance read from a reference solution (4x iterations).
  F. Extra checks on the main experiment (construction quality, GNN destroy weight, pure overhead).
  G. What predicts the training label best: the network or the exact insertion cost?    [insertion_baseline.json]

Usage: python scripts/analyse_followup.py [--out results/followup_analysis]
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyse_main import (REPO, GROUPS, GNAME, bks_tables, load, score_of, group_of, compare, md_table, f2, fp)  # noqa: E402

NAME = {"B1_alns": "Classical ALNS (B1)", "P_hgnn": "HGNN-ALNS (P)", "A_INT1_prune": "Pruning only (A-INT1)",
        "A_INT2_prior": "Prior only (A-INT2)", "A_INT3_noGNNdestroy": "Pruning + prior, no GNN destroy (A-INT3)",
        "O_oracle": "Oracle-guided ALNS", "REF_long": "Classical ALNS, 4x iterations (reference)"}
COL = {"B1_alns": "#1baf7a", "P_hgnn": "#2a78d6", "A_INT1_prune": "#eda100", "A_INT2_prior": "#6250d6",
       "A_INT3_noGNNdestroy": "#eb6834", "O_oracle": "#d6409f", "REF_long": "#8a8984"}
GALL = GROUPS + ["ALL"]


def prep(path, bks):
    rows, _ = load(path)
    out = []
    for r in rows:
        b = bks.get(r["instance"].lower().replace("hetiuy", ""))
        if b is None:
            continue
        r["bks_vehicles"], r["bks_distance"] = b
        r["score"] = score_of(r, r["vehicles"], r["distance"]) if r["unserved"] == 0 else 1000.0
        r["group"] = group_of(r)
        out.append(r)
    return out


def index(rows):
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["instance"], r["algo"])][r["seed"]] = r
    return by


def rate(rs, num, den):
    a = sum(r["diag"][num] for r in rs); b = sum(r["diag"][den] for r in rs)
    return a / b if b else float("nan")


def irate(by, ii, al, num, den):
    """Mean over instances of the per-instance rate (the averaging used in section C)."""
    v = [rate(list(by[(i, al)].values()), num, den) for i in ii if by.get((i, al))]
    return float(np.nanmean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "results", "followup_analysis"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = lambda *p: os.path.join(REPO, *p)   # noqa: E731
    bks = bks_tables()
    out, J = ["# Follow-up analysis: why does the guidance not help?\n"], {}

    # ------------------------------------------------------------------ A: tuning instances
    trows = []
    for f in ("tune_A.jsonl", "tune_B.jsonl"):
        trows += load(R("results", f))[0]
    tby = collections.defaultdict(dict)
    for r in trows:
        tby[(r["family"], r["instance"])][r["algo"]] = r
    out.append("## A. Tuning instances (generated, same distribution as the training data; 1 seed, 0.25 s per request)\n")
    out.append("Score difference against the tuned classical ALNS on the same instance (positive = worse than classical ALNS).\n")
    t, J["tuning"] = [], []
    for fam, ref, lex in (("gen-lilim", "B1_alns@xi0.4_eta0.025", True), ("gen-hdarp", "B1_alns@xi0.2_eta0.0", False)):
        for K in ("K999", "K0.5", "K5"):
            for lam in ("lam0.2", "lam1.0"):
                al = f"P_hgnn@{K}_{lam}"
                d = []
                for (fm, _), v in tby.items():
                    if fm == fam and al in v and ref in v:
                        x, y = v[al], v[ref]
                        d.append((100.0 * (x["vehicles"] - y["vehicles"]) if lex else 0.0)
                                 + 100.0 * (x["distance"] - y["distance"]) / y["distance"])
                if not d:
                    continue
                d = np.array(d)
                p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                kn = {"K999": "no pruning", "K0.5": "top 50%", "K5": "top 5"}[K]
                t.append(["Li & Lim style" if lex else "H-DARP style", kn, lam[3:], len(d), f"{d.mean():+.2f}", f"{np.median(d):+.2f}",
                          f"{int((d < 0).sum())} / {int((d == 0).sum())} / {int((d > 0).sum())}", fp(p)])
                J["tuning"].append({"family": fam, "K": K, "lam": lam, "n": len(d), "mean_diff": float(d.mean()),
                                    "median_diff": float(np.median(d)), "p": float(p)})
    out.append(md_table(["Family", "Pruning", "lambda", "Inst.", "Mean diff.", "Median diff.", "Win / tie / loss", "Wilcoxon p"], t))

    # ------------------------------------------------------------------ load follow-up + main
    F = prep(R("results", "followup.jsonl"), bks)
    C = [r for r in prep(R("results", "core.jsonl"), bks) if r["budget"] not in (10.0, 20.0)]
    fby, cby = index(F), index(C)
    insts = {g: sorted({r["instance"] for r in F if r["group"] == g and r["algo"] == "B1_alns"}) for g in GROUPS}
    insts["ALL"] = sum((insts[g] for g in GROUPS), [])
    sub = json.load(open(R("results", "followup_subset.json")))
    subs = {g: [i for i in insts[g] if i in sub] for g in GALL}
    out.append(f"\nFollow-up runs loaded: {len(F)} (all feasible: {all(r['checker_ok'] for r in F)}; "
               f"runs with unserved requests: {sum(r['unserved'] > 0 for r in F)}).\n")

    # ------------------------------------------------------------------ B: equal iterations
    out.append("## B. Equal number of iterations: HGNN-ALNS (P) against classical ALNS (B1)\n")
    out.append("Each run gets the median number of iterations that classical ALNS made on that instance in the main "
               "experiment. The time overhead of the network is therefore removed. Positive difference = P is worse.\n")
    t, J["equal_iter"] = [], []
    for g in GALL:
        ci = compare(fby, insts[g], "P_hgnn", "B1_alns")
        ct = compare(cby, insts[g], "P_hgnn", "B1_alns")
        if ci is None:
            continue
        tr = [np.mean([r["real_elapsed"] for r in fby[(i, "P_hgnn")].values()]) / np.mean([r["real_elapsed"] for r in fby[(i, "B1_alns")].values()])
              for i in insts[g] if (i, "P_hgnn") in fby and (i, "B1_alns") in fby]
        t.append([GNAME[g], ci["n"], f2(ct["mean_a"]), f2(ct["mean_b"]), f"{ct['diff']:+.2f}", fp(ct["p"]),
                  f2(ci["mean_a"]), f2(ci["mean_b"]), f"{ci['diff']:+.2f}", f"{ci['win']} / {ci['tie']} / {ci['loss']}", fp(ci["p"]),
                  f2(ci["a12"]), f2(float(np.mean(tr)))])
        J["equal_iter"].append({"group": g, "equal_time": ct, "equal_iter": ci, "time_ratio": float(np.mean(tr))})
    out.append(md_table(["Group", "Inst.", "P (equal time)", "B1 (equal time)", "Diff.", "p", "P (equal iter.)", "B1 (equal iter.)",
                         "Diff.", "Win / tie / loss", "Wilcoxon p", "A12", "Time of P / time of B1"], t))

    # ------------------------------------------------------------------ C: search behaviour
    out.append("\n## C. How the guidance changes the search (equal iterations; mean over instances)\n")
    out.append("- *Back to origin*: share of removed requests that the repair step puts back into the route they were taken from.\n"
               "- *Network top-1 = origin*: share of removed requests for which the network ranks the origin route first.\n"
               "- *Same solution*: share of iterations where the repair step rebuilds exactly the solution it started from.\n"
               "- *Seen before*: share of iterations whose result was already visited.\n"
               "- *Improving*: share of iterations whose result is better than the current solution.\n")
    t, J["behaviour"] = [], []
    for g in GALL:
        for al in ("B1_alns", "P_hgnn"):
            v = collections.defaultdict(list)
            for i in insts[g]:
                rs = list(fby.get((i, al), {}).values())
                if not rs:
                    continue
                v["undo"].append(rate(rs, "undo", "defined")); v["top1"].append(rate(rs, "top1_origin", "top1_defined"))
                for k in ("same", "revisit", "accepted", "improved"):
                    v[k].append(rate(rs, k, "iters"))
                v["best"].append(1000.0 * rate(rs, "new_best", "iters"))
            if not v:
                continue
            m = {k: float(np.nanmean(x)) for k, x in v.items()}
            t.append([GNAME[g], NAME[al], f"{100 * m['undo']:.1f}%", "–" if al == "B1_alns" else f"{100 * m['top1']:.1f}%",
                      f"{100 * m['same']:.1f}%", f"{100 * m['revisit']:.1f}%", f"{100 * m['improved']:.1f}%",
                      f"{100 * m['accepted']:.1f}%", f2(m["best"], 1)])
            J["behaviour"].append({"group": g, "algo": al, **m})
    out.append(md_table(["Group", "Algorithm", "Back to origin", "Network top-1 = origin", "Same solution", "Seen before",
                         "Improving", "Accepted", "New best per 1000 iter."], t))
    # per-instance link between the change of behaviour and the change of the result
    out.append("\nSpearman correlation, per instance, between the score difference P − B1 (equal iterations) and the change of "
               "behaviour P − B1:\n")
    J["behaviour_corr"] = []
    for g in GALL:
        dx = collections.defaultdict(list); dy = []
        for i in insts[g]:
            ra, rb = list(fby.get((i, "P_hgnn"), {}).values()), list(fby.get((i, "B1_alns"), {}).values())
            if not ra or not rb:
                continue
            dy.append(np.mean([r["score"] for r in ra]) - np.mean([r["score"] for r in rb]))
            dx["back to origin"].append(rate(ra, "undo", "defined") - rate(rb, "undo", "defined"))
            dx["seen before"].append(rate(ra, "revisit", "iters") - rate(rb, "revisit", "iters"))
            dx["improving"].append(rate(ra, "improved", "iters") - rate(rb, "improved", "iters"))
        if len(dy) < 5:
            continue
        parts = []
        for k, x in dx.items():
            rho, p = stats.spearmanr(x, dy)
            parts.append(f"{k} rho = {rho:+.2f} (p = {fp(p)})")
            J["behaviour_corr"].append({"group": g, "what": k, "rho": float(rho), "p": float(p)})
        out.append(f"- **{GNAME[g]}**: " + "; ".join(parts))

    # ------------------------------------------------------------------ D: components
    out.append("\n## D. Components at equal iterations (all instances, seeds 0–4)\n")
    # On H-DARP the tuned setting has no pruning: A-INT1 is classical ALNS and A-INT3 is A-INT2. Check it on the runs
    # that exist for both, then reuse the identical runs for the remaining H-DARP instances.
    same = tot = 0
    for i in insts["DARP"]:
        for x, y in (("A_INT1_prune", "B1_alns"), ("A_INT3_noGNNdestroy", "A_INT2_prior")):
            for sd, r in fby.get((i, x), {}).items():
                if sd in fby.get((i, y), {}):
                    tot += 1
                    same += abs(r["distance"] - fby[(i, y)][sd]["distance"]) < 1e-9 and r["vehicles"] == fby[(i, y)][sd]["vehicles"]
    dby = dict(fby)
    for i in insts["DARP"]:
        if (i, "B1_alns") in fby:
            dby[(i, "A_INT1_prune")] = fby[(i, "B1_alns")]
        if (i, "A_INT2_prior") in fby:
            dby[(i, "A_INT3_noGNNdestroy")] = fby[(i, "A_INT2_prior")]
    out.append(f"On H-DARP the tuned setting has no pruning, so A-INT1 is identical to classical ALNS and A-INT3 is identical to "
               f"A-INT2 (checked: {same} of {tot} paired runs give exactly the same solution). Those rows reuse the identical runs.\n")
    J["hdarp_identity_check"] = {"same": int(same), "pairs": int(tot)}
    t, J["components"] = [], []
    for g in GALL:
        for al in ("A_INT1_prune", "A_INT2_prior", "A_INT3_noGNNdestroy", "P_hgnn"):
            c = compare(dby, insts[g], al, "B1_alns")
            if c is None:
                continue
            ii = [i for i in insts[g] if dby.get((i, al)) and dby.get((i, "B1_alns"))]
            t.append([GNAME[g], NAME[al], c["n"], f2(c["mean_a"]), f2(c["mean_b"]), f"{c['diff']:+.2f}",
                      f"{c['win']} / {c['tie']} / {c['loss']}", fp(c["p"]), f2(c["a12"]),
                      f"{100 * irate(dby, ii, al, 'undo', 'defined'):.1f}%", f"{100 * irate(dby, ii, 'B1_alns', 'undo', 'defined'):.1f}%",
                      f"{100 * irate(dby, ii, al, 'improved', 'iters'):.1f}%", f"{100 * irate(dby, ii, 'B1_alns', 'improved', 'iters'):.1f}%"])
            J["components"].append({"group": g, "algo": al, **c})
    out.append(md_table(["Group", "Algorithm", "Inst.", "Mean score", "B1 mean score", "Diff.", "Win / tie / loss", "Wilcoxon p", "A12",
                         "Back to origin", "B1 back to origin", "Improving", "B1 improving"], t))
    out.append("\nEffect of one component when the others stay the same (negative difference = the first is better):\n")
    t, J["component_pairs"] = [], []
    for name, x, y in (("adding the GNN destroy operator: P vs A-INT3", "P_hgnn", "A_INT3_noGNNdestroy"),
                       ("adding pruning to the prior: A-INT3 vs A-INT2", "A_INT3_noGNNdestroy", "A_INT2_prior"),
                       ("adding the prior to pruning: A-INT3 vs A-INT1", "A_INT3_noGNNdestroy", "A_INT1_prune")):
        for g in GALL:
            if g == "DARP" and "pruning" in name.split(":")[0] and "adding pruning" in name:
                continue                                  # no pruning on H-DARP: the two are the same runs
            c = compare(dby, insts[g], x, y)
            if c is None:
                continue
            t.append([name, GNAME[g], c["n"], f2(c["mean_a"]), f2(c["mean_b"]), f"{c['diff']:+.2f}",
                      f"{c['win']} / {c['tie']} / {c['loss']}", fp(c["p"]), f2(c["a12"])])
            J["component_pairs"].append({"comparison": name, "group": g, **c})
    out.append(md_table(["Comparison", "Group", "Inst.", "Mean score (first)", "Mean score (second)", "Mean diff.", "Win / tie / loss",
                         "Wilcoxon p", "A12"], t))

    # where the Li & Lim loss comes from: more vehicles or longer routes
    out.append("\nLi & Lim only: the score mixes vehicles and distance, so the two parts are shown separately (equal iterations, "
               "mean over seeds, then over instances; positive = worse than classical ALNS). *Distance, same vehicles* uses only "
               "the instances where both algorithms have the same mean number of vehicles.\n")
    t, J["vehicles_distance"] = [], []
    for g in ("LL100", "LL200"):
        for al in ("A_INT1_prune", "A_INT2_prior", "A_INT3_noGNNdestroy", "P_hgnn"):
            dv, dd, ds = [], [], []
            for i in insts[g]:
                a_, b_ = dby.get((i, al), {}), dby.get((i, "B1_alns"), {})
                sd = sorted(set(a_) & set(b_))
                if not sd:
                    continue
                va, vb = np.mean([a_[s]["vehicles"] for s in sd]), np.mean([b_[s]["vehicles"] for s in sd])
                da, db = np.mean([a_[s]["distance"] for s in sd]), np.mean([b_[s]["distance"] for s in sd])
                dv.append(va - vb); dd.append(100.0 * (da - db) / db)
                if abs(va - vb) < 1e-9:
                    ds.append(100.0 * (da - db) / db)
            if not dv:
                continue
            dv, dd, ds = np.array(dv), np.array(dd), np.array(ds)
            pv = stats.wilcoxon(dv).pvalue if np.any(dv != 0) else 1.0
            ps = stats.wilcoxon(ds).pvalue if len(ds) and np.any(ds != 0) else 1.0
            t.append([GNAME[g], NAME[al], len(dv), f"{dv.mean():+.3f}", f"{int((dv > 0).sum())} / {int((dv < 0).sum())}", fp(pv),
                      f"{dd.mean():+.2f}%", len(ds), f"{ds.mean():+.2f}%" if len(ds) else "–", fp(ps)])
            J["vehicles_distance"].append({"group": g, "algo": al, "n": len(dv), "veh_diff": float(dv.mean()), "p_veh": float(pv),
                                           "dist_diff_pct": float(dd.mean()), "n_same_veh": len(ds),
                                           "dist_diff_same_veh_pct": float(ds.mean()) if len(ds) else None, "p_dist_same_veh": float(ps)})
    out.append(md_table(["Group", "Algorithm", "Inst.", "Vehicles, diff. to B1", "Inst. with more / fewer vehicles", "Wilcoxon p (vehicles)",
                         "Distance, diff. to B1", "Inst. with same vehicles", "Distance, same vehicles", "Wilcoxon p (distance)"], t))

    # ------------------------------------------------------------------ E: oracle
    if any(r["algo"] == "O_oracle" for r in F):
        out.append("\n## E. Oracle guidance (36-instance subset, seeds 0–2, equal iterations)\n")
        out.append("The reference solution comes from one classical ALNS run with 4 times the iterations. The oracle replaces the "
                   "network: a request prefers the route that holds its route-mates of the reference solution. Pruning rule and "
                   "prior weight are the same as for HGNN-ALNS. This is an upper bound for what correct route advice can give.\n")
        t, J["oracle"] = [], []
        for g in GALL:
            ii = [i for i in subs[g] if (i, "O_oracle") in fby and (i, "REF_long") in fby and (i, "B1_alns") in fby and (i, "P_hgnn") in fby]
            if not ii:
                continue
            m = {}
            for al in ("B1_alns", "P_hgnn", "O_oracle"):
                m[al] = np.array([np.mean([fby[(i, al)][s]["score"] for s in (0, 1, 2) if s in fby[(i, al)]]) for i in ii])
            m["REF_long"] = np.array([fby[(i, "REF_long")][0]["score"] for i in ii])
            co = compare(fby, ii, "O_oracle", "B1_alns")
            cr = stats.wilcoxon(m["O_oracle"] - m["REF_long"]).pvalue if np.any(m["O_oracle"] != m["REF_long"]) else 1.0
            t.append([GNAME[g], len(ii), f2(m["B1_alns"].mean()), f2(m["P_hgnn"].mean()), f2(m["O_oracle"].mean()), f2(m["REF_long"].mean()),
                      f"{co['diff']:+.2f}", f"{co['win']} / {co['tie']} / {co['loss']}", fp(co["p"]),
                      f"{(m['O_oracle'] - m['REF_long']).mean():+.2f}", fp(cr), f"{100 * irate(fby, ii, 'O_oracle', 'undo', 'defined'):.1f}%",
                      f"{100 * irate(fby, ii, 'O_oracle', 'improved', 'iters'):.1f}%"])
            J["oracle"].append({"group": g, "n": len(ii), **{k: float(v.mean()) for k, v in m.items()}, "oracle_vs_b1": co, "p_oracle_vs_ref": float(cr)})
        out.append(md_table(["Group", "Inst.", "B1", "P", "Oracle", "Reference (4x)", "Oracle − B1", "Win / tie / loss", "Wilcoxon p",
                             "Oracle − reference", "p", "Back to origin", "Improving"], t))

    # ------------------------------------------------------------------ F: extra checks on the main experiment
    out.append("\n## F. Extra checks on the main experiment (equal time, 0.5 s per request)\n")
    t, J["extra"] = [], []
    for g in GALL:
        ii = [i for i in insts[g] if (i, "P_hgnn") in cby and (i, "B1_alns") in cby]
        if not ii:
            continue
        def first(i, al):  # construction result: first point of the trajectory
            rs = cby[(i, al)].values()
            return np.mean([r["trajectory"][0][1] for r in rs]), np.mean([r["trajectory"][0][2] for r in rs])
        dd = [100.0 * (first(i, "P_hgnn")[0] - first(i, "B1_alns")[0]) / first(i, "B1_alns")[0] for i in ii]
        dv = [first(i, "P_hgnn")[1] - first(i, "B1_alns")[1] for i in ii]
        pc = stats.wilcoxon(dd).pvalue if np.any(np.array(dd) != 0) else 1.0
        w = [r["destroy_weights"]["gnn"] / sum(r["destroy_weights"].values()) for i in ii for r in cby[(i, "P_hgnn")].values()
             if "gnn" in r["destroy_weights"]]
        t.append([GNAME[g], len(ii), f"{np.mean(dd):+.2f}%", f"{np.mean(dv):+.2f}", fp(pc), f"{100 * np.mean(w):.1f}%"])
        J["extra"].append({"group": g, "construction_dist_diff_pct": float(np.mean(dd)), "construction_veh_diff": float(np.mean(dv)),
                           "p_construction": float(pc), "gnn_destroy_weight_share": float(np.mean(w))})
    out.append(md_table(["Group", "Inst.", "Construction distance, P vs B1", "Construction vehicles, P − B1", "Wilcoxon p (distance)",
                         "Final weight share of the GNN destroy operator (uniform = 16.7%)"], t))
    c = compare(cby, insts["DARP"], "A_INT1_prune", "B1_alns")
    if c:
        out.append(f"\nPure overhead on H-DARP: with the tuned H-DARP setting, A-INT1 has no pruning and no prior, so it is classical "
                   f"ALNS that also pays for the network. Equal time: A-INT1 {c['mean_a']:.2f} vs B1 {c['mean_b']:.2f} "
                   f"(diff {c['diff']:+.2f}, Wilcoxon p = {fp(c['p'])}).")
        J["overhead_hdarp"] = c

    # ------------------------------------------------------------------ G: what predicts the label?
    ib_path, re_path = R("results", "insertion_baseline.json"), R("results", "ranking_eval.json")
    if os.path.exists(ib_path):
        IB = json.load(open(ib_path)); RE = json.load(open(re_path)) if os.path.exists(re_path) else {}
        out.append("\n## G. What predicts the training label best? (held-out Validation-ML instances, same destroyed states)\n")
        out.append("The label of a removed request is its origin route in the elite solution. *Exact insertion cost* ranks the "
                   "candidate routes by the cheapest feasible insertion in the destroyed state. This is the quantity that every "
                   "repair operator of classical ALNS computes anyway.\n")
        t, J["label_accuracy"] = [], []
        for key, rk, nm in (("lilim", "lilim/all", "Li & Lim style"), ("hdarp n<=100", "hdarp/n<=100", "H-DARP style, n ≤ 100"),
                            ("hdarp n>100", "hdarp/n>100", "H-DARP style, n > 100")):
            if key not in IB:
                continue
            d = IB[key]; nr = RE.get(rk, {}).get("nearest_route", {})
            cell = lambda m: f"{d[m]['top1']:.2f} / {d[m]['recall@5']:.2f}"   # noqa: E731
            t.append([nm, d["instances"], d["requests"], f"{d['mean_candidates']:.1f}", cell("random"),
                      f"{nr['top1']:.2f} / {nr['recall@5']:.2f}" if nr else "–", cell("homgnn"), cell("hgnn"), cell("insertion")])
            J["label_accuracy"].append({"data": key, **d, "nearest_route": nr})
        out.append(md_table(["Data", "Inst.", "Removed requests", "Mean candidates", "Random", "Nearest route", "Homogeneous GNN", "HGNN",
                             "Exact insertion cost"], t))
        out.append("\nCells: top-1 accuracy / recall@5. The nearest-route numbers come from the first ranking evaluation "
                   "(same instances, a different sample of destroyed states).")
        if all("recall@half" in IB[k]["hgnn"] for k in IB):
            out.append("\nPruning rule used on Li & Lim (keep the top 50% of the candidate routes, at least 3): share of removed "
                       "requests whose label route is **cut away** by the rule.\n")
            t = []
            for key, nm in (("lilim", "Li & Lim style"), ("hdarp n<=100", "H-DARP style, n ≤ 100"), ("hdarp n>100", "H-DARP style, n > 100")):
                if key in IB:
                    d = IB[key]
                    t.append([nm] + [f"{100 * (1 - d[m]['recall@half']):.1f}%" for m in ("random", "homgnn", "hgnn", "insertion")])
            out.append(md_table(["Data", "Random", "Homogeneous GNN", "HGNN", "Exact insertion cost"], t))
            out.append("\n(The tuned H-DARP setting does not prune; the H-DARP rows are shown for reference only.)")

    open(os.path.join(a.out, "report.md"), "w").write("\n".join(out) + "\n")
    json.dump(J, open(os.path.join(a.out, "summary.json"), "w"), indent=1)

    # ------------------------------------------------------------------ figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "grid.color": "#e6e5e0", "axes.grid": True,
                         "axes.axisbelow": True, "grid.linewidth": 0.6, "legend.frameon": False})
    x = np.arange(len(GROUPS)); gl = [GNAME[g] for g in GROUPS]

    def bars(ax, series, fmt="{:.1f}"):
        wd = 0.8 / len(series)
        for j, (lab, col, v) in enumerate(series):
            xs = x[:len(v)] - 0.4 + wd * (j + 0.5)
            ax.bar(xs, v, wd * 0.86, color=col, label=lab)
            for xx, vv in zip(xs, v):
                if np.isfinite(vv):
                    ax.annotate(fmt.format(vv), (xx, vv), xytext=(0, 2 if vv >= 0 else -2), textcoords="offset points",
                                va="bottom" if vv >= 0 else "top",
                                ha="center", fontsize=7, color="#52514e")
        ax.set_xticks(x); ax.set_xticklabels(gl); ax.grid(axis="x", visible=False)

    # F1: P - B1 at equal time and at equal iterations
    E = {e["group"]: e for e in J["equal_iter"]}
    if all(g in E for g in GROUPS):
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        bars(ax, [("equal time", "#8a8984", [E[g]["equal_time"]["diff"] for g in GROUPS]),
                  ("equal iterations", COL["P_hgnn"], [E[g]["equal_iter"]["diff"] for g in GROUPS])], "{:+.2f}")
        ax.axhline(0, color="#52514e", lw=0.8)
        ax.set_ylabel("score difference:\nHGNN-ALNS − classical ALNS", fontsize=8.5)
        ax.set_title("Above 0: HGNN-ALNS is worse", fontsize=8.5, loc="left", color="#52514e")
        lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_ - 0.08 * (hi_ - lo_), hi_ + 0.06 * (hi_ - lo_))
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_equal_iter.png"), dpi=200); plt.close(fig)

    # F2: search behaviour, small multiples
    B = {(b["group"], b["algo"]): b for b in J["behaviour"]}
    if all((g, al) in B for g in GROUPS for al in ("B1_alns", "P_hgnn")):
        fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
        for ax, (k, ttl) in zip(axs, (("undo", "Removed requests put back\ninto their origin route (%)"),
                                      ("revisit", "Iterations that reach a solution\nseen before (%)"),
                                      ("improved", "Iterations that improve\nthe current solution (%)"))):
            bars(ax, [(NAME[al], COL[al], [100 * B[(g, al)][k] for g in GROUPS]) for al in ("B1_alns", "P_hgnn")])
            ax.set_xticklabels(["Li & Lim\n100", "Li & Lim\n200", "H-DARP"])
            ax.set_title(ttl, fontsize=9)
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(a.out, "fig_behaviour.png"), dpi=200); plt.close(fig)

    # F3: oracle
    if J.get("oracle"):
        O = {o["group"]: o for o in J["oracle"]}
        gs = [g for g in GROUPS if g in O]
        fig, axs = plt.subplots(1, len(gs), figsize=(3.3 * len(gs), 3.2))
        for ax, g in zip(np.atleast_1d(axs), gs):
            als = ["B1_alns", "P_hgnn", "O_oracle", "REF_long"]
            v = [O[g][al] for al in als]
            xs = np.arange(len(als))
            ax.bar(xs, v, 0.7, color=[COL[al] for al in als])
            for xx, vv in zip(xs, v):
                ax.annotate(f"{abs(vv):.2f}", (xx, vv), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color="#52514e")
            ax.set_xticks(xs); ax.set_xticklabels(["Classical\nALNS", "HGNN-\nALNS", "Oracle-\nguided", "Classical,\n4x iter."], fontsize=7.5)
            ax.set_title(f"{GNAME[g]} ({O[g]['n']} instances)", fontsize=9); ax.grid(axis="x", visible=False)
        np.atleast_1d(axs)[0].set_ylabel("mean score (lower is better)")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_oracle.png"), dpi=200); plt.close(fig)
    # F4: label accuracy
    if J.get("label_accuracy"):
        L = J["label_accuracy"]
        fig, ax = plt.subplots(figsize=(7.2, 3.3))
        xs = np.arange(len(L))
        ser = [("Random", "#8a8984", [d["random"]["top1"] for d in L]),
               ("Nearest route", "#eda100", [d["nearest_route"].get("top1", np.nan) for d in L]),
               ("Homogeneous GNN", "#eb6834", [d["homgnn"]["top1"] for d in L]),
               ("HGNN", COL["P_hgnn"], [d["hgnn"]["top1"] for d in L]),
               ("Exact insertion cost", COL["B1_alns"], [d["insertion"]["top1"] for d in L])]
        wd = 0.8 / len(ser)
        for j, (lab, col, v) in enumerate(ser):
            xx = xs - 0.4 + wd * (j + 0.5)
            ax.bar(xx, v, wd * 0.86, color=col, label=lab)
            for q, vv in zip(xx, v):
                if np.isfinite(vv):
                    ax.annotate(f"{vv:.2f}", (q, vv), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color="#52514e")
        ax.set_xticks(xs); ax.set_xticklabels([{"lilim": "Li & Lim style", "hdarp n<=100": "H-DARP style, n ≤ 100",
                                                "hdarp n>100": "H-DARP style, n > 100"}[d["data"]] for d in L])
        ax.set_ylabel("top-1 accuracy on the origin route"); ax.set_ylim(0, 0.85); ax.grid(axis="x", visible=False)
        ax.legend(fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12), columnspacing=1.2, handlelength=1.2)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_label_accuracy.png"), dpi=200); plt.close(fig)
    # F5: components at equal iterations (difference to classical ALNS), one panel per group
    C = {(c["group"], c["algo"]): c for c in J.get("components", [])}
    als = ["A_INT1_prune", "A_INT2_prior", "A_INT3_noGNNdestroy", "P_hgnn"]
    if all((g, al) in C for g in GROUPS for al in als):
        fig, axs = plt.subplots(1, len(GROUPS), figsize=(3.4 * len(GROUPS), 3.3))
        for ax, g in zip(axs, GROUPS):
            v = [C[(g, al)]["diff"] for al in als]
            xs = np.arange(len(als))
            ax.bar(xs, v, 0.7, color=[COL[al] for al in als])
            for xx, al, vv in zip(xs, als, v):
                pv = C[(g, al)]["p"]
                star = " *" if pv < 0.05 else ""
                ax.annotate(f"{vv:+.2f}{star}", (xx, vv), xytext=(0, 2 if vv >= 0 else -2), textcoords="offset points",
                            va="bottom" if vv >= 0 else "top", ha="center", fontsize=7, color="#52514e")
            ax.axhline(0, color="#52514e", lw=0.8)
            ax.set_xticks(xs); ax.set_xticklabels(["Pruning\nonly", "Prior\nonly", "Pruning\n+ prior", "Full\nmethod"], fontsize=7.5)
            ax.set_title(f"{GNAME[g]} ({C[(g, als[0])]['n']} instances)", fontsize=9); ax.grid(axis="x", visible=False)
            lo_, hi_ = ax.get_ylim(); pad = 0.12 * (hi_ - lo_); ax.set_ylim(lo_ - pad, hi_ + pad)
        axs[0].set_ylabel("score difference to classical ALNS\n(above 0 = worse)", fontsize=8.5)
        fig.text(0.5, 0.01, "* Wilcoxon p < 0.05.  Equal iterations, 5 seeds.  On H-DARP the tuned setting has no pruning.",
                 ha="center", fontsize=7.5, color="#52514e")
        fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(a.out, "fig_components.png"), dpi=200); plt.close(fig)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
