"""Out-of-fold predictions of EVERY method (incl. the nested-CV fusion) on the 50 independently adjudicated gold claims,
scored against (a) the adjudicated gold labels and (b) the author labels for the same claims. CPU, seconds.

  python -m train.gold_fusion_eval
Needs: outputs/nested_cv.preds.npy (written by train.nested_cv) and data/meetverify_v2/agreement/gold_claims.jsonl.
Gold: SUPPORTED=1, everything else 0.  No refitting: each claim's prediction comes from a model that never saw its meeting.
Writes outputs/gold_fusion_table.tex and outputs/gold_fusion.json."""
import argparse, json
import numpy as np
from cprove import metrics as M

ap = argparse.ArgumentParser()
ap.add_argument("--preds", default="outputs/nested_cv.preds.npy"); ap.add_argument("--features", default="cache/frozen/features.jsonl")
ap.add_argument("--gold", default="data/meetverify_v2/agreement/gold_claims.jsonl"); ap.add_argument("--boots", type=int, default=2000)
a = ap.parse_args()
P = np.load(a.preds, allow_pickle=True).item(); cids = P.pop("cids"); pos = {c: i for i, c in enumerate(cids)}
gold = {}
for l in open(a.gold, encoding="utf-8"):
    r = json.loads(l)
    if r["set"] == "old" and r["cid"] in pos:
        gold[r["cid"]] = r["label"]
meta = {}
for l in open(a.features, encoding="utf-8"):
    r = json.loads(l); meta.setdefault(r["cid"], r)
ids = sorted(gold); idx = np.array([pos[c] for c in ids])
yg = np.array([int(gold[c] == "SUPPORTED") for c in ids]); ya = np.array([int(meta[c]["author_label"] == "SUPPORTED") for c in ids])
grp = np.array([meta[c]["meeting_id"] for c in ids])
print(f"{len(ids)} gold claims with out-of-fold predictions ({len({*grp})} meetings); gold supported {yg.sum()}, author supported {ya.sum()}, labels differing {(yg != ya).sum()}")
names = [("fusion (selected in inner CV)", "Fusion (selected)"), ("fusion +mc", "Fusion + MiniCheck"), ("entailment", "Entailment"), ("minicheck", "MiniCheck"),
         ("cprove_hand_nograph", "Hand weights, no graph"), ("cprove_hand_graph", "Hand weights, graph"), ("summac", "SummaC-ZS")]
res, rows = {}, []
print(f"{'method':26} {'AUROC gold [CI]':24} {'AUROC author [CI]':24} {'recall@20% gold':>15} {'recall@20% author':>17}")
for key, label in names:
    if key not in P:
        continue
    p = np.array(P[key], float)[idx]
    if np.isnan(p).any():
        continue
    g_auc, a_auc = M.auroc(yg, p), M.auroc(ya, p)
    g_ci, a_ci = M.cluster_bootstrap(M.auroc, yg, p, grp, a.boots), M.cluster_bootstrap(M.auroc, ya, p, grp, a.boots)
    g_r, a_r = M.recall_at_review(yg, p, (0.2,))[0.2], M.recall_at_review(ya, p, (0.2,))[0.2]
    res[key] = dict(auroc_gold=g_auc, ci_gold=g_ci, auroc_author=a_auc, ci_author=a_ci, recall20_gold=g_r, recall20_author=a_r)
    rows.append((label, g_auc, g_ci, a_auc, a_ci, g_r, a_r))
    print(f"{label:26} {g_auc:.3f} [{g_ci[0]:.2f},{g_ci[1]:.2f}]      {a_auc:.3f} [{a_ci[0]:.2f},{a_ci[1]:.2f}]      {g_r:15.3f} {a_r:17.3f}")
r_g = sorted(res, key=lambda k: -res[k]["auroc_gold"]); r_a = sorted(res, key=lambda k: -res[k]["auroc_author"])
print("ranking by gold  :", r_g); print("ranking by author:", r_a)
tex = ["\\begin{table}[htbp]", "\\centering",
       "\\caption{Out-of-fold performance on the 50 independently adjudicated claims, scored against the adjudicated gold labels and against the author's labels for the same claims. AUROC with 95\\% meeting-clustered CIs; recall@20\\% is the share of unsupported claims among the 20\\% lowest-trust claims.}",
       "\\label{tab:goldfusion}", "\\small", "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
       "Method & AUROC (gold) & AUROC (author) & Recall@20\\% (gold) & Recall@20\\% (author) \\\\", "\\midrule"]
for label, g, gc, aa, ac, gr, ar in rows:
    tex.append(f"{label} & {g:.3f} [{gc[0]:.2f}, {gc[1]:.2f}] & {aa:.3f} [{ac[0]:.2f}, {ac[1]:.2f}] & {gr:.2f} & {ar:.2f} \\\\")
tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
import pathlib; pathlib.Path("outputs").mkdir(exist_ok=True)
open("outputs/gold_fusion_table.tex", "w").write("\n".join(tex)); json.dump(res, open("outputs/gold_fusion.json", "w"), indent=1)
print("wrote outputs/gold_fusion_table.tex")
