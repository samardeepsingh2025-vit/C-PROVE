"""Recompute y_graph / y_nograph in the feature table with the CONFIGURED (hand) rule weights only (CPU, ~1 min, no GPU).
  python -m train.recompute_hand      # writes cache/frozen/features.jsonl (backup: features.jsonl.bak)
Use only if repro/check_leakage.py reports that models/rule_weights.json differs from hlmrf.init_weights."""
import glob, json, shutil
from pathlib import Path
from cprove.utils import load_config
from cprove.hlmrf import HLMRF, Weights
from experiments.common import programs
import pickle
cfg = load_config("configs/frozen.yaml"); h = cfg["hlmrf"]
model = HLMRF(Weights(h["init_weights"]), h["iters"], h["tol"])      # NOT Weights.load: no learned file
items = [it for f in sorted(glob.glob(str(Path(cfg["paths"]["cache"]) / "items_*.pkl"))) for it in pickle.load(open(f, "rb"))]
y = {True: {}, False: {}}
for graph in (True, False):
    for it, pr in zip(items, programs(items, graph=graph)):
        yy, _ = model.infer(pr)
        for c in it.claims: y[graph][c.cid] = float(yy[pr.y_index(c.cid)])
fp = Path("cache/frozen/features.jsonl"); shutil.copy(fp, str(fp) + ".bak"); n = 0; out = []
for l in open(fp, encoding="utf-8"):
    if not l.strip(): continue
    r = json.loads(l)
    if r["cid"] in y[True]: r["y_graph"], r["y_nograph"] = y[True][r["cid"]], y[False][r["cid"]]; n += 1
    out.append(json.dumps(r, ensure_ascii=False))
open(fp, "w", encoding="utf-8").write("\n".join(out) + "\n"); print(f"updated {n} rows with hand-weight posteriors")
