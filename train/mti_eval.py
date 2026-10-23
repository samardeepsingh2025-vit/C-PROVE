"""Review prioritisation and summary-level index (CPU, seconds).

  python -m train.mti_eval --features cache/frozen/features.jsonl
(1) Review prioritisation: rank claims by lowest trust; fraction of ALL unsupported claims found when only the top 10/20/30/50%
    are reviewed, per method, with meeting-clustered CIs (the practical use case of a trust score).
(2) Summary level: index = omega-weighted mean of calibrated claim trust (omega: decision/action_item=2, others=1, as in the MTI
    definition; the component scores need the full pipeline and are NOT used here). Compared with simple baselines
    (mean trust, minimum trust, mean MiniCheck) against the human-reviewed share of unsupported claims in the summary
    (Spearman) and for detecting high-error summaries (share unsupported above the median; AUROC).  Summary-level gold here is DERIVED from the
    author-reviewed claim labels - it is NOT an independent summary-factuality judgement, so this is exploratory.
"""
import argparse, json
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from cprove import metrics as M
from train.nested_cv import load_rows, run, baselines

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="cache/frozen/features.jsonl"); ap.add_argument("--out", default="outputs/mti_eval.json")
a = ap.parse_args()
rows = load_rows(a.features)
y = np.array([int(r["author_label"] == "SUPPORTED") for r in rows]); g = np.array([r["meeting_id"] for r in rows])
out, _ = run(rows); preds = {**baselines(rows), "fusion (selected)": out["selected"]}
res = {"review": {}, "summary": {}}
print("(1) fraction of unsupported claims caught when reviewing the lowest-trust X% of claims")
print(f"{'method':26} 10%     20%     30%     50%      (random review would find 10/20/30/50%)")
for k, p in preds.items():
    if np.isnan(p).any():
        continue
    r = M.recall_at_review(y, p); res["review"][k] = r
    cis = {f: M.cluster_bootstrap(lambda yy, pp, f=f: M.recall_at_review(yy, pp, (f,))[f], y, p, g, 300) for f in (0.2,)}
    print(f"{k:26} " + "  ".join(f"{r[f]:.3f}" for f in (0.1, 0.2, 0.3, 0.5)) + f"    [20%: {cis[0.2][0]:.3f}-{cis[0.2][1]:.3f}]")

sid = np.array([r["summary_id"] for r in rows]); omega = np.array([2.0 if r["claim_type"] in ("decision", "action_item") else 1.0 for r in rows])
S = sorted(set(sid)); p = out["selected"]
gold_frac = np.array([1 - y[sid == s].mean() for s in S])
med = float(np.median(gold_frac)); has_err = (gold_frac > med).astype(int)
if has_err.min() == has_err.max():
    has_err = (gold_frac >= med).astype(int)
agg = {"MTI-style (omega-weighted mean trust)": [float((omega[sid == s] * p[sid == s]).sum() / omega[sid == s].sum()) for s in S],
       "mean trust": [float(p[sid == s].mean()) for s in S], "min trust": [float(p[sid == s].min()) for s in S]}
mc = np.array([r.get("mc", np.nan) for r in rows])
if not np.isnan(mc).any():
    agg["mean MiniCheck"] = [float(mc[sid == s].mean()) for s in S]
print(f"\n(2) summary level: {len(S)} summaries; mean share unsupported {gold_frac.mean():.2f}; 'high-error' summaries (share unsupported above the median {med:.2f}) {has_err.sum()}")
for k, v in agg.items():
    v = np.array(v); rho = spearmanr(v, gold_frac)[0]
    auc = M.auroc(1 - has_err, v) if len(set(has_err)) == 2 else float("nan")
    res["summary"][k] = {"spearman_vs_unsupported_share": rho, "auroc_has_error": auc}
    print(f"{k:42} Spearman vs share unsupported {rho:+.3f}   AUROC(high-error summary) {auc:.3f}")
json.dump(res, open(a.out, "w"), indent=1, default=float)
