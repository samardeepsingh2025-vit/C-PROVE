"""Final held-out evaluation. Fit fusion on DEV pilot labels, evaluate on TEST labels (never used for fitting).
python -m train.final_eval --ann data\\meetverify\\annotations_A1.jsonl --dev cache\\items_qwen_dev20.pkl --test data\\meetverify\\qwen_test.jsonl"""
import argparse, json, pickle
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from cprove.utils import load_config, seed_everything
from cprove.data import load_corpus
from cprove.pipeline import CProve, Ablation
from cprove.dimensions import Calibrators
from cprove.hlmrf import HLMRF, Weights
from experiments.common import load_records, build_items, programs
from train.fit_cprove import recalibrate

ap = argparse.ArgumentParser()
ap.add_argument("--ann", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--config", default=None)
a = ap.parse_args()
cfg = load_config(a.config); seed_everything(cfg["seed"])
ann = {}
for l in open(a.ann, encoding="utf-8"):
    r = json.loads(l); ann[r["cid"]] = r

def feats(o):
    raw = o.raw; row = [float(raw.get("ent") or 0), float(raw.get("con") or 0)]
    for f in ("attr", "sem"):
        v = raw.get(f); row += [0.0 if v is None else float(v) - 0.5, 0.0 if v is None else 1.0]
    return row

def collect(items, ys=None):
    X, y, base, cids = [], [], [], []
    for k, it in enumerate(items):
        for c in it.claims:
            r = ann.get(c.cid)
            if not r or r["label"] == "UNVERIFIABLE": continue
            X.append(feats(it.obs[c.cid])); y.append(int(r["label"] == "SUPPORTED"))
            base.append(ys[k][c.cid] if ys else np.nan); cids.append(c.cid)
    return np.array(X), np.array(y), np.array(base), cids

dev = pickle.load(open(a.dev, "rb"))
Xd, yd, _, _ = collect(dev)
meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
recs = [r for r in load_records(a.test) if r.get("system") != "reference"]
cp = CProve(cfg, Ablation(hand_weights=True)); cp.dims.cal = Calibrators("none")
items = build_items(cp, meetings, recs, str(Path(cfg["paths"]["cache"]) / ("items_" + Path(a.test).stem + ".pkl")))
recalibrate(items, Calibrators.load(cfg["paths"]["models"] + "/calibrators.pkl"))
h = cfg["hlmrf"]
model = HLMRF(Weights.load(h.get("learned_weights"), h["init_weights"]), h["iters"], h["tol"])
ys = []
for it, pr in zip(items, programs(items)):
    y, _ = model.infer(pr); ys.append({c.cid: float(y[pr.y_index(c.cid)]) for c in it.claims})
Xt, yt, yc, cids = collect(items, ys)
print(f"dev n={len(yd)} (supported {yd.mean():.2f}) | test n={len(yt)} (supported {yt.mean():.2f}) meetings={len({c.split('__')[0] for c in cids})}")

def auroc(p, y=yt): return roc_auc_score(1 - y, 1 - p)
def boot(p, y=yt, n=1000):
    rng = np.random.default_rng(0); s = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1: s.append(auroc(p[i], y[i]))
    return np.percentile(s, [2.5, 97.5])
def ece(p, y, b=10):
    e = 0
    for k in range(b):
        m = (p >= k / b) & (p < (k + 1) / b + (k == b - 1))
        if m.sum(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e
res = {"C-PROVE (hand weights, graph)": yc, "entailment only": Xt[:, 0]}
for name, idx in {"fusion ent+con": [0, 1], "fusion ent+con+attr": [0, 1, 2, 3], "fusion ent+con+attr+sem": [0, 1, 2, 3, 4, 5]}.items():
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(Xd[:, idx], yd)
    res[name] = clf.predict_proba(Xt[:, idx])[:, 1]
print(f"{'method':32} AUROC [95% CI]       AUPRC(unsup)  ECE")
for k, p in res.items():
    lo, hi = boot(p)
    print(f"{k:32} {auroc(p):.3f} [{lo:.3f},{hi:.3f}]  {average_precision_score(1 - yt, 1 - p):.3f}        {ece(p, yt):.3f}")
# selective prediction: accuracy on kept claims when abstaining on least-confident
p = res["fusion ent+con+attr+sem"]
conf = np.abs(p - 0.5)
for cov in (1.0, 0.8, 0.6):
    keep = conf >= np.quantile(conf, 1 - cov)
    print(f"coverage {cov:.1f}: accuracy {((p[keep] >= 0.5) == yt[keep]).mean():.3f}")
