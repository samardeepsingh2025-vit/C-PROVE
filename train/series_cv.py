"""Leave-one-SERIES-out CV (CPU, ~1 min). Same pipeline as train.nested_cv, but outer AND inner folds group by scenario series
(e.g. TS3010a-d -> TS3010), so no series shares participants/topic across train and test.

  python -m train.series_cv            # needs cache/frozen/features.jsonl and the baseline files used by nested_cv
Writes outputs/series_cv.json and prints the numbers for the paper."""
import argparse, json, re, copy
import numpy as np
from cprove import metrics as M
from train.nested_cv import load_rows, run, baselines

ap = argparse.ArgumentParser(); ap.add_argument("--features", default="cache/frozen/features.jsonl")
ap.add_argument("--boots", type=int, default=1000); a = ap.parse_args()
rows = load_rows(a.features)
series = lambda m: re.sub(r"[a-z]$", "", m)                     # TS3010b -> TS3010
orig = np.array([r["meeting_id"] for r in rows]); ser = np.array([series(m) for m in orig])
y = np.array([int(r["author_label"] == "SUPPORTED") for r in rows])
print(f"{len(rows)} claims, {len(set(orig))} meetings, {len(set(ser))} series")
rows_s = copy.deepcopy(rows)
for r in rows_s: r["meeting_id"] = series(r["meeting_id"])      # folds now = series
fixed = {"fusion +mc": ["ent", "con", "attr", "sem", "mc"]}
out, ch = run(rows_s, fixed=fixed)
base = baselines(rows)
preds = {"minicheck": base["minicheck"], "entailment": base["entailment"], "fusion (selected, series-CV)": out["selected"], "fusion +mc (series-CV)": out["fusion +mc"]}
res = {}
print(f"{'method':32} AUROC [CI, meeting-clustered]   AUPRC-unsup   ECE10")
for k, p in preds.items():
    r = {"auroc": M.auroc(y, p), "auroc_ci": M.cluster_bootstrap(M.auroc, y, p, orig, a.boots), "auprc_unsup": M.auprc(y, p), "ece10": M.ece(y, p, 10)}
    res[k] = r; print(f"{k:32} {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f},{r['auroc_ci'][1]:.3f}]   {r['auprc_unsup']:.3f}   {r['ece10']:.3f}")
ps = {}
for k in ("fusion (selected, series-CV)", "fusion +mc (series-CV)"):
    d, lo, hi, p = M.paired_bootstrap(M.auroc, y, preds[k], preds["minicheck"], orig); ps[k] = p
    print(f"paired AUROC {k} - minicheck: {d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}"); res[k]["vs_minicheck"] = [d, lo, hi, p]
print("Holm-adjusted p:", {k: round(v, 3) for k, v in M.holm(ps).items()})
json.dump({"series": int(len(set(ser))), "results": res, "choices": ch}, open("outputs/series_cv.json", "w"), indent=1, default=float)
