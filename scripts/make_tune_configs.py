"""Write the two tuning configs (Final Methodology Section 11.7: small grid, equal effort).
Stage A tunes the shared ALNS core on classical ALNS; Stage B tunes guidance on HGNN-ALNS."""
import itertools, yaml

models = {"hgnn": {"lilim": "models/lilim_hgnn.pt", "hdarp": "models/hdarp_hgnn.pt"},
          "homgnn": {"lilim": "models/lilim_homgnn.pt", "hdarp": "models/hdarp_homgnn.pt"}}
inst = [{"family": "gen-lilim", "size": "valopt"}, {"family": "gen-hdarp", "size": "valopt"}]

A = {}
for xi, eta in itertools.product([0.2, 0.4], [0.0, 0.025, 0.05]):
    A[f"B1_alns@xi{xi}_eta{eta}"] = {"xi": xi, "noise": eta}
yaml.safe_dump({"models": models, "tune": A,
                "blocks": [{"name": "tuneA", "instances": inst, "budgets": ["0.25n"],
                            "algorithms": {k: 1 for k in A}}]}, open("configs/tune_A.yaml", "w"), sort_keys=False)

def stageB(core_by_family):
    B = {}
    for K, lam in itertools.product([5, 0.5, 999], [0.2, 1.0]):
        B[f"P_hgnn@K{K}_lam{lam}"] = {"by_family": {f: dict(c, guide={"K": K, "lam": lam})
                                                    for f, c in core_by_family.items()}}
    yaml.safe_dump({"models": models, "tune": B,
                    "blocks": [{"name": "tuneB", "instances": inst, "budgets": ["0.25n"],
                                "algorithms": {k: 1 for k in B}}]}, open("configs/tune_B.yaml", "w"), sort_keys=False)


def core_from_stageA(path):
    import json, re, subprocess, sys
    out = subprocess.run([sys.executable, "scripts/tune_select.py", path], capture_output=True, text=True).stdout
    best = json.loads(out.strip().splitlines()[-1])
    core = {}
    for fam, name in best.items():
        p = dict((m.group(1), float(m.group(2))) for m in re.finditer(r"(xi|eta)([0-9.]+)", name))
        core[fam] = {"xi": p["xi"], "noise": p["eta"]}
    return core


if __name__ == "__main__":
    import sys
    core = core_from_stageA(sys.argv[1]) if len(sys.argv) > 1 else {"lilim": {}, "hdarp": {}}
    stageB(core)
    print("wrote configs/tune_A.yaml and configs/tune_B.yaml; stage-A core =", core)
