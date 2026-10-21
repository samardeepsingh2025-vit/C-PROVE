"""Which parts of the pipeline help on REAL summaries? AUROC on the human-verified pilot labels.
python -m train.ablate_pilot --ann data\\meetverify\\annotations_A1.jsonl --cache cache\\items_qwen_dev20.pkl"""
import argparse, json, pickle
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
    r = json.loads(l); ann[r["cid"]] = r
items = pickle.load(open(a.cache, "rb"))
recalibrate(items, Calibrators.load(cfg["paths"]["models"] + "/calibrators.pkl"))
h = cfg["hlmrf"]
learned = HLMRF(Weights.load(h.get("learned_weights"), h["init_weights"]), h["iters"], h["tol"])
hand = HLMRF(Weights(h["init_weights"]), h["iters"], h["tol"])
VARIANTS = [
    ("full (learned lambda)", learned, dict()),
    ("full (hand lambda)", hand, dict()),
    ("no graph", learned, dict(graph=False)),
    ("drop attr", learned, dict(drop_dims=("attr",))),
    ("drop ctx", learned, dict(drop_dims=("ctx",))),
    ("drop sem", learned, dict(drop_dims=("sem",))),
    ("drop attr+ctx+sem", learned, dict(drop_dims=("attr", "ctx", "sem"))),
    ("drop attr+ctx+sem, no graph", learned, dict(drop_dims=("attr", "ctx", "sem"), graph=False)),
    ("drop ctx+sem", learned, dict(drop_dims=("ctx", "sem"))),
]
y_all = {}
print(f"{'variant':34}{'n':>5}{'AUROC':>8}{'AUPRC':>8}")
def report(name, scores):
    s = [(l, v) for l, v in scores if v is not None]
    y = np.array([x[0] for x in s]); v = np.array([x[1] for x in s])
    print(f"{name:34}{len(s):5d}{roc_auc_score(1 - y, 1 - v):8.3f}{average_precision_score(1 - y, 1 - v):8.3f}")
for name, model, kw in VARIANTS:
    sc = []
    for it, pr in zip(items, programs(items, **kw)):
        y, _ = model.infer(pr)
        for c in it.claims:
            r = ann.get(c.cid)
            if r and r["label"] != "UNVERIFIABLE":
                sc.append((int(r["label"] == "SUPPORTED"), float(y[pr.y_index(c.cid)])))
    report(name, sc)
sc = [(int(ann[c.cid]["label"] == "SUPPORTED"), it.obs[c.cid].raw["fact"]) for it in items for c in it.claims
      if c.cid in ann and ann[c.cid]["label"] != "UNVERIFIABLE"]
report("fact only (raw NLI)", sc)
sc = [(int(ann[c.cid]["label"] == "SUPPORTED"), 1 - it.obs[c.cid].raw["con"]) for it in items for c in it.claims
      if c.cid in ann and ann[c.cid]["label"] != "UNVERIFIABLE"]
report("1 - contradiction only", sc)
sc = [(int(ann[c.cid]["label"] == "SUPPORTED"), it.obs[c.cid].raw["ent"]) for it in items for c in it.claims
      if c.cid in ann and ann[c.cid]["label"] != "UNVERIFIABLE"]
report("entailment only", sc)
