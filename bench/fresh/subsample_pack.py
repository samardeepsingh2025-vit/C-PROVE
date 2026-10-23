"""Cap the number of claims per summary that annotators must label (uniform random, seeded), so the
workload is feasible. Random sampling keeps the per-summary share of unsupported claims unbiased.
Summary-level ratings still cover the WHOLE summary.

  python -m bench.fresh.subsample_pack --cap 10        (run BEFORE make_old_pack; keeps a backup pack_full.json)
"""
import argparse, json, random, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--pack", default="data/meetverify_v2/pack.json")
ap.add_argument("--cap", type=int, default=10)
ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()
p = Path(a.pack)
full = p.with_name("pack_full.json")
if not full.exists():
    shutil.copy(p, full)
pk = json.load(open(full, encoding="utf-8"))
rng = random.Random(a.seed)
n0 = n1 = 0
for s in pk["summaries"]:
    n0 += len(s["claims"])
    s["n_claims_total"] = len(s["claims"])
    if len(s["claims"]) > a.cap:
        s["claims"] = sorted(rng.sample(s["claims"], a.cap), key=lambda c: c["public_id"])
    n1 += len(s["claims"])
json.dump(pk, open(p, "w", encoding="utf-8"), ensure_ascii=False)
print(f"{n0} claims -> {n1} claims to annotate ({len(pk['summaries'])} summaries, cap {a.cap}/summary)")
