"""Meeting-clustered bootstrap CIs for macro-F1 (all methods, per-fold threshold) and for P/R/F1 of the selected fusion at the inner-CV threshold.
   Put in train/ and run:  python -m train.f1_ci"""
import json, numpy as np
from cprove import metrics as M
from train.nested_cv import best_threshold, load_rows
P = np.load("outputs/nested_cv.preds.npy", allow_pickle=True).item(); cids = P.pop("cids")
rows = {r["cid"]: r for r in load_rows("cache/frozen/features.jsonl")}
y = np.array([int(rows[c]["author_label"] == "SUPPORTED") for c in cids]); g = np.array([rows[c]["meeting_id"] for c in cids])
B = 1000
def ci(key):
    return lambda yy, pp: M.prf(yy, pp, 0.5)[key]
def show(tag, pred):
    out = []
    for key in ("macro_f1", "precision_unsup", "recall_unsup", "f1_unsup"):
        v = M.prf(y, pred, 0.5)[key]; lo, hi = M.cluster_bootstrap(ci(key), y, pred, g, B)
        out.append(f"{key} {v:.3f} [{lo:.3f},{hi:.3f}]")
    print(f"{tag:34}", " | ".join(out))
print("== per-held-out-meeting threshold (as in Table 2) ==")
for k, p in P.items():
    p = np.array(p, float)
    if np.isnan(p).any(): continue
    t = np.zeros(len(p))
    for m in np.unique(g):
        te = g == m; t[te] = best_threshold(y[~te], p[~te])
    show(k, (p >= t).astype(float))
print("== selected fusion at inner-CV threshold ==")
ch = {c["meeting"]: c["threshold"] for c in json.load(open("outputs/nested_cv.json"))["choices"]}
t = np.array([ch[m] for m in g]); p = np.array(P["fusion (selected in inner CV)"], float)
show("fusion (selected), inner thr", (p >= t).astype(float))
print(M.prf(y, (p >= t).astype(float), 0.5)["confusion"])
