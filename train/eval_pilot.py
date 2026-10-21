"""Score C-PROVE against your human labels on the pilot claims.
python -m train.eval_pilot --ann data\\meetverify\\annotations_A1.jsonl --cache cache\\items_qwen_dev20.pkl"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from cprove.utils import load_config
from cprove.dimensions import Calibrators
from cprove.hlmrf import HLMRF, Weights
from experiments.common import programs
from train.fit_cprove import recalibrate

ap = argparse.ArgumentParser()
ap.add_argument("--ann", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--config", default=None)
a = ap.parse_args()
cfg = load_config(a.config)
ann = {}
for l in open(a.ann, encoding="utf-8"):
    r = json.loads(l); ann[r["cid"]] = r      # last label wins
items = pickle.load(open(a.cache, "rb"))
recalibrate(items, Calibrators.load(cfg["paths"]["models"] + "/calibrators.pkl"))
h = cfg["hlmrf"]
model = HLMRF(Weights.load(h.get("learned_weights"), h["init_weights"]), h["iters"], h["tol"])
rows = []
for it, pr in zip(items, programs(items)):
    y, _ = model.infer(pr)
    for c in it.claims:
        r = ann.get(c.cid)
        if not r or r["label"] == "UNVERIFIABLE":
            continue
        o = it.obs[c.cid]
        sc = {"fused": float(y[pr.y_index(c.cid)]), "fact": o.raw["fact"]}
        for d in ("attr", "ctx", "sem"):
            sc[d] = o.raw.get(d)
        rows.append((int(r["label"] == "SUPPORTED"), r["error_type"], sc))
y = np.array([r[0] for r in rows])
print(f"labelled claims matched to pipeline: {len(rows)} of {len(ann)}  | unsupported: {int((y == 0).sum())} ({(y == 0).mean():.0%})")
ch = [r for r in ann.values() if "changed" in r]
if ch:
    print(f"human changed the AI draft on {sum(r['changed'] for r in ch)} of {len(ch)} items; label-only disagreement {sum(r['label'] != r['draft_label'] for r in ch)}")
if len(set(y)) == 2:
    print("\nAUROC for detecting unsupported claims (higher = better)")
    for k in ("fused", "fact", "attr", "ctx", "sem"):
        s = [(r[0], r[2][k]) for r in rows if r[2][k] is not None]
        yy = np.array([x[0] for x in s]); ss = np.array([x[1] for x in s])
        if len(set(yy)) == 2:
            print(f"  {k:6} n={len(s):3d}  AUROC {roc_auc_score(1 - yy, 1 - ss):.3f}  AUPRC {average_precision_score(1 - yy, 1 - ss):.3f}")
    print("\nby error type (fused / fact AUROC vs all supported claims):")
    for t in sorted({r[1] for r in rows if r[0] == 0}):
        m = [(r[0], r[2]["fused"], r[2]["fact"]) for r in rows if r[0] == 1 or r[1] == t]
        yy = np.array([x[0] for x in m])
        print(f"  {t:18} errors={int((yy == 0).sum()):3d}  fused {roc_auc_score(1 - yy, 1 - np.array([x[1] for x in m])):.2f}  fact {roc_auc_score(1 - yy, 1 - np.array([x[2] for x in m])):.2f}")
