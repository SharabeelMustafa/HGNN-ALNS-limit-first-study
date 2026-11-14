import sys, csv
sys.path.insert(0, "src")
from hgalns.io import list_instances, load_instance
from hgalns.alns import ALNSConfig, run_alns
from hgalns.checker import check_solution

bks = {r["instance"].lower(): (int(r["vehicles"]), float(r["distance"])) for r in csv.DictReader(open("data/bks/lilim_bks.csv"))}
bks.update({r["instance"]: (None, float(r["bks"])) for r in csv.DictReader(open("data/bks/malheiros2021_mdhdarp_bks.csv"))})
cases = [(p, b) for p, b in [(list_instances('lilim',100)[30], float(sys.argv[1])), (list_instances('lilim',100)[5], float(sys.argv[1])),
                             (list_instances('malheiros')[2], float(sys.argv[2]))]]
for path, budget in cases:
    inst = load_instance(path)
    res = run_alns(inst, ALNSConfig(budget=budget, seed=1))
    s = res["solution"]
    ok, errs, met = check_solution(inst, s.routes, s.schedules(), s.bank)
    key = inst.name.lower() if inst.family == "lilim" else inst.name.replace("hetIUY", "")
    b = bks.get(key)
    gap = 100 * (res["distance"] - b[1]) / b[1] if b else float("nan")
    print(f"{inst.name:14s} n={inst.n:3d} veh={res['vehicles']:3d} (bks {b[0]}) dist={res['distance']:8.2f} bks={b[1]:8.2f} gap={gap:6.2f}% "
          f"unserved={res['unserved']} iters={res['iterations']} t={res['elapsed']:.1f}s checker={ok} {errs[:1]}")
