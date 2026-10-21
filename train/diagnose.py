"""AUROC of each raw dimension score per error type (dev cache). python -m train.diagnose [items_pkl]"""
import pickle, sys, numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
p = sys.argv[1] if len(sys.argv) > 1 else "cache/items_synth_dev.pkl"
items = pickle.load(open(p, "rb"))
from experiments.common import refine_gold, load_records
_rp = sys.argv[2] if len(sys.argv) > 2 else "data/meetverify/synth_dev.jsonl"
import os
if os.path.exists(_rp):
    refine_gold(items, load_records(_rp))
DIMS = ("fact", "attr", "temp", "ctx", "sem")
fused = {}
try:
    from cprove.utils import load_config
    from cprove.dimensions import Calibrators
    from cprove.hlmrf import HLMRF, Weights
    from train.fit_cprove import recalibrate
    from experiments.common import programs
    cfg = load_config(None); h = cfg["hlmrf"]
    recalibrate(items, Calibrators.load(cfg["paths"]["models"] + "/calibrators.pkl"))
    model = HLMRF(Weights.load(h.get("learned_weights"), h["init_weights"]), h["iters"], h["tol"])
    for it, pr in zip(items, programs(items)):
        y, _ = model.infer(pr)
        for c in it.claims:
            fused[c.cid] = float(y[pr.y_index(c.cid)])
except Exception as e:
    print("fused skipped:", repr(e))
DIMS_ALL = DIMS + ("fused",)
rows = []
for it in items:
    for c in it.claims:
        if c.cid in it.gold:
            o = it.obs[c.cid]
            raw = {d: o.raw.get(d) for d in DIMS}
            raw["fused"] = fused.get(c.cid)
            vs = [v for v in raw.values() if v is not None]
            raw["mean"] = float(np.mean(vs)) if vs else None
            raw["min"] = float(min(vs)) if vs else None
            rows.append((it.gold[c.cid], getattr(c, "gold_error", "none") or "none", raw))
print("claims", len(rows), "errors", sum(1 for r in rows if r[0] == 0))
print("coverage (non-None):", {d: round(np.mean([r[2][d] is not None for r in rows]), 2) for d in DIMS})
types = sorted({r[1] for r in rows if r[0] == 0})
print(f"{'dim':6}" + "".join(f"{t[:11]:>13}" for t in ["ALL"] + types))
for d in list(DIMS) + ["mean", "min", "fused"]:
    out = []
    for t in ["ALL"] + types:
        sub = [r for r in rows if r[2][d] is not None and (r[0] == 1 or t == "ALL" or r[1] == t)]
        y = [r[0] for r in sub]; s = [r[2][d] for r in sub]
        out.append(f"{roc_auc_score(1 - np.array(y), 1 - np.array(s)):13.2f}" if len(set(y)) == 2 else f"{'-':>13}")
    print(f"{d:6}" + "".join(out))
print("errors per type:", {t: sum(1 for r in rows if r[0] == 0 and r[1] == t) for t in types})