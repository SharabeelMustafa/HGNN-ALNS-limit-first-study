"""Write tuned settings (from tune_select results) into the 'tune' section of experiment configs.
Usage: python scripts/apply_tuning.py results/tune_A.jsonl results/tune_B.jsonl configs/core.yaml configs/pilot.yaml"""
import json, re, subprocess, sys, yaml

def best(path):
    out = subprocess.run([sys.executable, "scripts/tune_select.py", path], capture_output=True, text=True).stdout
    print(out)
    return json.loads(out.strip().splitlines()[-1])

def parse(name):
    p = {}
    for m in re.finditer(r"(xi|eta|K|lam)([0-9.]+)", name.split("@")[1]):
        p[m.group(1)] = float(m.group(2))
    return p

A, B = best(sys.argv[1]), best(sys.argv[2])
core, guide = {}, {}
for fam in ("lilim", "hdarp"):
    a = parse(A[fam]); core[fam] = {"xi": a["xi"], "noise": a["eta"]}
    b = parse(B[fam]); K = b["K"]; guide[fam] = {"K": int(K) if K >= 1 else K, "lam": b["lam"]}
tune = {
    "B1_alns": {"by_family": {f: core[f] for f in core}},
    "B2_alns_uniform": {"by_family": {f: core[f] for f in core}},
    "A_ALNS3_rrt": {"by_family": {f: core[f] for f in core}},
    "A_ALNS4_itersa": {"by_family": {f: core[f] for f in core}},
    "P_hgnn": {"by_family": {f: dict(core[f], guide=guide[f]) for f in core}},
    "A_INT3_noGNNdestroy": {"by_family": {f: dict(core[f], guide=guide[f]) for f in core}},
    "B5_homgnn": {"by_family": {f: dict(core[f], guide=guide[f]) for f in core}},
    "A_INT1_prune": {"by_family": {f: dict(core[f], guide={"K": guide[f]["K"], "lam": 0.0}) for f in core}},
    "A_INT2_prior": {"by_family": {f: dict(core[f], guide={"K": 999, "lam": guide[f]["lam"]}) for f in core}},
}
for cfg in sys.argv[3:]:
    c = yaml.safe_load(open(cfg)); c["tune"] = tune
    yaml.safe_dump(c, open(cfg, "w"), sort_keys=False)
json.dump({"core": core, "guide": guide}, open("results/tuned_settings.json", "w"), indent=1)
print("tuned:", json.dumps({"core": core, "guide": guide}))
