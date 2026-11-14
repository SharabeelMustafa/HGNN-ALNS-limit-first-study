"""Pick the best tuning configuration per family from a results file.
Score per run (relative to the best run on that instance): 100*(extra vehicles) for
lexicographic instances + % distance above the best distance. Lower is better."""
import json, sys, collections

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip() and not l.startswith('{"_meta"')]
best = {}
for r in rows:
    k = (r["family"], r["instance"])
    key = (r["vehicles"] if "lilim" in r["family"] else 0, r["distance"]) if r["unserved"] == 0 else (1e9, 1e9)
    if k not in best or key < best[k]:
        best[k] = key
score = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    bv, bd = best[(r["family"], r["instance"])]
    fam = "lilim" if "lilim" in r["family"] else "hdarp"
    if r["unserved"] > 0:
        s = 1000.0
    else:
        s = (100.0 * (r["vehicles"] - bv) if fam == "lilim" else 0.0) + 100.0 * (r["distance"] - bd) / bd
    score[fam][r["algo"]].append(s)
out = {}
for fam, d in score.items():
    print(f"== {fam}")
    ranked = sorted(d.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    for a, v in ranked:
        print(f"   {a:30s} mean score {sum(v)/len(v):7.3f}  (n={len(v)})")
    out[fam] = ranked[0][0]
print(json.dumps(out))
