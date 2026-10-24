"""Paired AUROC differences (meeting-clustered bootstrap, Holm over the family) of extra baselines vs MiniCheck and vs the selected fusion.
  python -m train.nested_cv            # re-run AFTER the baseline_*.json files exist, so they enter outputs/nested_cv.preds.npy
  python -m train.baseline_paired --names alignscore qwenjudge gemmajudge
Names = baseline files cache/frozen/baseline_<name>.json.  523 claims, same protocol as Table 2 (baselines have no fitted parameters, so no nested refit)."""
import argparse, json
import numpy as np
from cprove import metrics as M
from train.nested_cv import load_rows

ap = argparse.ArgumentParser()
ap.add_argument("--names", nargs="+", required=True); ap.add_argument("--preds", default="outputs/nested_cv.preds.npy")
ap.add_argument("--features", default="cache/frozen/features.jsonl"); ap.add_argument("--boots", type=int, default=2000)
a = ap.parse_args()
P = np.load(a.preds, allow_pickle=True).item(); cids = P.pop("cids")
rows = {r["cid"]: r for r in load_rows(a.features)}
y = np.array([int(rows[c]["author_label"] == "SUPPORTED") for c in cids]); g = np.array([rows[c]["meeting_id"] for c in cids])
print(f"{len(y)} claims, {len(set(g))} meetings")
ref, fus = np.array(P["minicheck"], float), np.array(P["fusion (selected in inner CV)"], float)
ps = {}
for n in a.names:
    if n not in P: print(f"{n}: not in preds - run train.nested_cv again"); continue
    p = np.array(P[n], float); lo, hi = M.cluster_bootstrap(M.auroc, y, p, g, a.boots)
    d, dlo, dhi, pv = M.paired_bootstrap(M.auroc, y, p, ref, g, n=a.boots); ps[n] = pv
    d2, d2lo, d2hi, pv2 = M.paired_bootstrap(M.auroc, y, p, fus, g, n=a.boots)
    print(f"{n:14} AUROC {M.auroc(y, p):.3f} [{lo:.3f},{hi:.3f}]  AUPRC(unsup) {M.auprc(y, p):.3f}  | vs MiniCheck {d:+.3f} [{dlo:+.3f},{dhi:+.3f}] p={pv:.3f}  | vs fusion {d2:+.3f} [{d2lo:+.3f},{d2hi:+.3f}] p={pv2:.3f}")
print("Holm-adjusted p (vs MiniCheck):", {k: round(v, 3) for k, v in M.holm(ps).items()})
