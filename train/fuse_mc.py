"""C-PROVE with MiniCheck as an extra evidence feature. Fit on dev, test on held-out test; also leave-one-meeting-out on all 523.
python -m train.fuse_mc --ann data\\meetverify\\annotations_A1.jsonl --dev cache\\items_qwen_dev20.pkl --test cache\\items_qwen_test.pkl"""
import argparse, json, pickle, gc
import numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from cprove.utils import load_config, seed_everything
from cprove.data import load_corpus
from baselines.run_baselines import transcript_text
ap = argparse.ArgumentParser()
ap.add_argument("--ann", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--test", required=True); ap.add_argument("--config", default=None)
a = ap.parse_args(); cfg = load_config(a.config); seed_everything(cfg["seed"])
ann = {}
for l in open(a.ann, encoding="utf-8"):
    r = json.loads(l); ann[r["cid"]] = r
meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
def feats(raw):
    row = [float(raw.get("ent") or 0), float(raw.get("con") or 0)]
    for f in ("attr", "sem"):
        v = raw.get(f); row += [0.0 if v is None else float(v) - 0.5, 0.0 if v is None else 1.0]
    return row
rows = []   # (split, meeting, cid, text, feats, y)
for split, p in (("dev", a.dev), ("test", a.test)):
    for it in pickle.load(open(p, "rb")):
        for c in it.claims:
            r = ann.get(c.cid)
            if r and r["label"] != "UNVERIFIABLE":
                rows.append((split, it.meeting_id, c.cid, c.text, feats(it.obs[c.cid].raw), int(r["label"] == "SUPPORTED")))
from minicheck.minicheck import MiniCheck
mc = MiniCheck(model_name="flan-t5-large", cache_dir="models/hf"); s = np.zeros(len(rows))
for mid in sorted({r[1] for r in rows}):
    idx = [i for i, r in enumerate(rows) if r[1] == mid]; doc = transcript_text(meetings[mid])
    _, p, _, _ = mc.score(docs=[doc] * len(idx), claims=[rows[i][3] for i in idx]); s[idx] = p; torch.cuda.empty_cache()
X = np.hstack([np.array([r[4] for r in rows]), s[:, None]]); y = np.array([r[5] for r in rows]); sp = np.array([r[0] for r in rows]); g = np.array([r[1] for r in rows])
subsets = {"MiniCheck only": [6], "ent+con+attr+sem": [0, 1, 2, 3, 4, 5], "+MiniCheck": [0, 1, 2, 3, 4, 5, 6], "ent+con+MiniCheck": [0, 1, 6], "ent+con+sem+MiniCheck": [0, 1, 4, 5, 6]}
tr, te = sp == "dev", sp == "test"
print(f"dev n={tr.sum()} test n={te.sum()}")
rng = np.random.default_rng(0)
def ci(p, yy):
    b = []
    for _ in range(1000):
        i = rng.integers(0, len(yy), len(yy))
        if len(set(yy[i])) > 1: b.append(roc_auc_score(1 - yy[i], 1 - p[i]))
    return np.percentile(b, [2.5, 97.5])
print(f"{'features':24} TEST AUROC [CI]       AUPRC   | LOMO-all AUROC")
for name, idx in subsets.items():
    if len(idx) == 1: pt = X[te][:, idx[0]]; pl = X[:, idx[0]]
    else:
        pt = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(X[tr][:, idx], y[tr]).predict_proba(X[te][:, idx])[:, 1]
        pl = np.zeros(len(y))
        for m in set(g):
            h = g == m
            pl[h] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(X[~h][:, idx], y[~h]).predict_proba(X[h][:, idx])[:, 1]
    lo, hi = ci(pt, y[te])
    print(f"{name:24} {roc_auc_score(1-y[te],1-pt):.3f} [{lo:.3f},{hi:.3f}]  {average_precision_score(1-y[te],1-pt):.3f}  | {roc_auc_score(1-y,1-pl):.3f}")
