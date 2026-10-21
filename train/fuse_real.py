"""Learn a small fusion on REAL human-verified claims, scored with leave-one-meeting-out CV.
python -m train.fuse_real --ann data\\meetverify\\annotations_A1.jsonl --cache cache\\items_qwen_dev20.pkl"""
import argparse, json, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
ap = argparse.ArgumentParser(); ap.add_argument("--ann", required=True); ap.add_argument("--cache", required=True)
a = ap.parse_args()
ann = {}
for l in open(a.ann, encoding="utf-8"):
    r = json.loads(l); ann[r["cid"]] = r
items = pickle.load(open(a.cache, "rb"))
FEATS = ["ent", "con", "attr", "ctx", "sem"]
X, y, g = [], [], []
for it in items:
    mid = str(it.rec["summary_id"]).split("__")[0]
    for c in it.claims:
        r = ann.get(c.cid)
        if not r or r["label"] == "UNVERIFIABLE": continue
        raw = it.obs[c.cid].raw
        row = []
        for f in FEATS:
            v = raw.get(f)
            row += [0.0 if v is None else float(v) - 0.5, 0.0 if v is None else 1.0] if f in ("attr", "ctx", "sem") else [float(v or 0)]
        X.append(row); y.append(int(r["label"] == "SUPPORTED")); g.append(mid)
X, y, g = np.array(X), np.array(y), np.array(g)
names = ["ent", "con", "attr", "attr?", "ctx", "ctx?", "sem", "sem?"]
cols = {"ent only": [0], "ent+con": [0, 1], "ent+con+sem": [0, 1, 6, 7], "ent+con+attr": [0, 1, 2, 3],
        "ent+con+ctx": [0, 1, 4, 5], "all": list(range(8))}
print(f"n={len(y)} meetings={len(set(g))}")
for name, idx in cols.items():
    for C in (0.1, 1.0):
        p = np.zeros(len(y))
        for m in set(g):
            te = g == m
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000).fit(X[~te][:, idx], y[~te])
            p[te] = clf.predict_proba(X[te][:, idx])[:, 1]
        print(f"{name:14} C={C:<4} CV AUROC {roc_auc_score(1 - y, 1 - p):.3f}")
