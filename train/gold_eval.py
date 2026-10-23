"""Score every non-learned method on the 50-claim independent GOLD set vs the author's labels on the SAME claims (CPU).
  python -m train.gold_eval --gold data/meetverify_v2/agreement/gold_claims.jsonl --features cache/frozen/features.jsonl
Shows (i) AUROC/AUPRC/Brier per method against gold and against author labels on the same claims, (ii) whether the method ranking
changes (label-noise sensitivity). Gold: SUPPORTED=1, everything else 0. n=50 -> wide CIs: report as a sanity check, not a headline."""
import argparse, json
import numpy as np
from cprove import metrics as M
from train.nested_cv import baselines

ap = argparse.ArgumentParser()
ap.add_argument("--gold", default="data/meetverify_v2/agreement/gold_claims.jsonl"); ap.add_argument("--features", default="cache/frozen/features.jsonl")
ap.add_argument("--boots", type=int, default=2000)
a = ap.parse_args()
gold = {}
for l in open(a.gold, encoding="utf-8"):
    r = json.loads(l)
    if r["set"] == "old":
        gold[r["cid"]] = r
rows = [json.loads(l) for l in open(a.features, encoding="utf-8")]
for r in rows:
    r.setdefault("mc", float("nan"))
seen = set(); rows = [r for r in rows if r["cid"] in gold and not (r["cid"] in seen or seen.add(r["cid"]))]
print(f"gold claims matched to features: {len(rows)}/{len(gold)}")
yg = np.array([int(gold[r["cid"]]["label"] == "SUPPORTED") for r in rows])
ya = np.array([int(r.get("author_label") == "SUPPORTED") for r in rows]); g = np.array([r["meeting_id"] for r in rows])
print(f"gold supported {yg.sum()}/{len(yg)}; author supported {ya.sum()}; label flips {(yg != ya).sum()}")
P = baselines(rows)
print(f"{'method':22} AUROC(gold) [95% CI]        AUROC(author)  AUPRC(gold)  Brier(gold)")
res = {}
for k, p in P.items():
    if np.isnan(p).any():
        continue
    ci = M.cluster_bootstrap(M.auroc, yg, p, g, a.boots)
    res[k] = (M.auroc(yg, p), M.auroc(ya, p))
    print(f"{k:22} {res[k][0]:.3f} [{ci[0]:.2f},{ci[1]:.2f}]        {res[k][1]:.3f}          {M.auprc(yg, p):.3f}        {M.brier(yg, p):.3f}")
rk = lambda i: [k for k, _ in sorted(res.items(), key=lambda kv: -kv[1][i])]
print("ranking by gold  :", rk(0)); print("ranking by author:", rk(1))
