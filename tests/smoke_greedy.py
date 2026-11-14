import sys, time
sys.path.insert(0, "src")
from hgalns.io import list_instances, load_instance
from hgalns.solution import Evaluator, Solution
from hgalns.checker import check_solution
from hgalns.feasibility import BIG

def greedy(inst):
    ev = Evaluator(inst); s = Solution(ev, bank=range(inst.n))
    while s.bank:
        best = (BIG, None)
        for r in s.bank:
            for k in s.candidate_vehicles():
                d, i, j = ev.insertion(s.routes[k], k, r)
                if d < best[0]:
                    best = (d, (r, k, i, j))
        if best[1] is None:
            break
        s.insert(*best[1])
    return s

for path in [list_instances('lilim',100)[0], list_instances('lilim',100)[30], list_instances('cordeau')[0], list_instances('malheiros')[0]]:
    inst = load_instance(path); t=time.time(); s = greedy(inst); el=time.time()-t
    ok, errs, met = check_solution(inst, s.routes, s.schedules(), s.bank)
    print(f"{inst.name:16s} n={inst.n:3d} K={inst.K:3d} veh={s.n_vehicles:3d} dist={s.distance:9.2f} bank={len(s.bank)} t={el:5.2f}s checker_ok={ok} {errs[:2]}")
