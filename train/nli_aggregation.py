"""STEP 2 (CPU, ~5-8 min): NLI aggregation ablation under nested leave-one-meeting-out CV.
  python -m train.nli_aggregation [--unverifiable_as_unsupported]
Aggregations of the per-window NLI outputs (cache/frozen/nli_windows.json):
  mix      shipped relevance/decisiveness-weighted mixture (taken from the feature table)
  max      ent = max_w p_ent,  con = max_w p_con
  mean     ent = mean_w p_ent, con = mean_w p_con
  top3     mean of the 3 largest p_ent (and of the 3 largest p_con)
  diff     ent = (max_w (p_ent - p_con) + 1)/2,  con = max_w p_con
  learned  logistic regression on the sorted top-3 p_ent and top-3 p_con (6 inputs), refitted in every fold
Each is evaluated (a) as the only feature ('entailment only': raw AUROC for scalar variants; nested LR for 'learned') and (b) inside the
selected fusion (feature set + C chosen in inner folds, everything else unchanged).  Paired AUROC difference to the shipped 'mix'."""
import argparse, json
import numpy as np
from cprove import metrics as M
import train.nested_cv as N

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="cache/frozen/features.jsonl"); ap.add_argument("--windows", default="cache/frozen/nli_windows.json")
ap.add_argument("--unverifiable_as_unsupported", action="store_true"); ap.add_argument("--boots", type=int, default=1000)
ap.add_argument("--out", default="outputs/nli_aggregation.json")
a = ap.parse_args()
W = json.load(open(a.windows))
base = [r for r in N.load_rows(a.features, unver_as_unsup=a.unverifiable_as_unsupported) if r["cid"] in W]
y = np.array([int(r["author_label"] == "SUPPORTED") for r in base]); g = np.array([r["meeting_id"] for r in base])
print(f"{len(base)} claims, {len(set(g))} meetings")


def topk(v, k=3):
    v = sorted(v, reverse=True)[:k]
    return v + [0.0] * (k - len(v))


def variant(r, kind):
    D = np.array(W[r["cid"]]["dist"]).reshape(-1, 3)
    if len(D) == 0:
        return dict(ent=0.0, con=0.0, e1=0, e2=0, e3=0, c1=0, c2=0, c3=0)
    e, c = D[:, 0], D[:, 2]; d = {}
    d["e1"], d["e2"], d["e3"] = topk(e); d["c1"], d["c2"], d["c3"] = topk(c)
    if kind == "max":
        d["ent"], d["con"] = e.max(), c.max()
    elif kind == "mean":
        d["ent"], d["con"] = e.mean(), c.mean()
    elif kind == "top3":
        d["ent"], d["con"] = float(np.mean(topk(e))), float(np.mean(topk(c)))
    elif kind == "diff":
        d["ent"], d["con"] = (float((e - c).max()) + 1) / 2, c.max()
    else:                       # mix / learned keep shipped ent, con
        d["ent"], d["con"] = r["ent"], r["con"]
    return {k: float(v) for k, v in d.items()}


N.FEATS["topk"] = ["e1", "e2", "e3", "c1", "c2", "c3"]
KINDS = ["mix", "max", "mean", "top3", "diff", "learned"]
res, preds_f = {}, {}
print(f"{'aggregation':10} {'entailment-only AUROC [CI]':30} {'selected fusion AUROC [CI]':30} fusion - mix (paired)")
for kind in KINDS:
    rows = []
    for r in base:
        r2 = dict(r); r2.update(variant(r, kind)); rows.append(r2)
    if kind == "learned":
        only, _ = N.run(rows, cands={"topk": ["topk"]})
        s_only = only["selected"]
        cands = {k: [("topk" if b in ("ent", "con") else b) for b in v] for k, v in N.CANDIDATES.items()}
        cands = {k: sorted(set(v), key=v.index) for k, v in cands.items()}
        cands = {k: v for k, v in cands.items()}
    else:
        s_only = np.array([r["ent"] for r in rows]); cands = None
    out, _ = N.run(rows, cands=cands)
    s_f = out["selected"]; preds_f[kind] = s_f
    ci_o, ci_f = M.cluster_bootstrap(M.auroc, y, s_only, g, a.boots), M.cluster_bootstrap(M.auroc, y, s_f, g, a.boots)
    res[kind] = dict(only=M.auroc(y, s_only), only_ci=ci_o, fusion=M.auroc(y, s_f), fusion_ci=ci_f)
    if kind != "mix":
        d, lo, hi, p = M.paired_bootstrap(M.auroc, y, s_f, preds_f["mix"], g); res[kind].update(diff=d, diff_ci=[lo, hi], diff_p=p)
        dtxt = f"{d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}"
    else:
        dtxt = "(reference)"
    print(f"{kind:10} {res[kind]['only']:.3f} [{ci_o[0]:.3f},{ci_o[1]:.3f}]".ljust(42) + f"{res[kind]['fusion']:.3f} [{ci_f[0]:.3f},{ci_f[1]:.3f}]".ljust(32) + dtxt, flush=True)
import pathlib; pathlib.Path("outputs").mkdir(exist_ok=True); json.dump(res, open(a.out, "w"), indent=1, default=float)
names = {"mix": "Weighted mixture (shipped)", "max": "Maximum over windows", "mean": "Mean over windows", "top3": "Mean of top-3 windows", "diff": "Max of $p_{\\mathrm{ent}}-p_{\\mathrm{con}}$", "learned": "Learned (LR on sorted top-3)"}
t = ["\\begin{table}[htbp]", "\\centering", "\\caption{NLI aggregation over the retrieved windows: nested leave-one-meeting-out AUROC [95\\% meeting-clustered CI] of the entailment feature alone and of the selected fusion, with the paired AUROC difference of the fusion to the shipped mixture.}", "\\label{tab:aggr}", "\\small",
     "\\begin{tabular}{@{}lccc@{}}", "\\toprule", "Aggregation & Entailment feature only & Selected fusion & Fusion $-$ shipped \\\\", "\\midrule"]
for k in KINDS:
    r = res[k]; dd = "--" if k == "mix" else f"{r['diff']:+.3f} [{r['diff_ci'][0]:+.3f}, {r['diff_ci'][1]:+.3f}]"
    t.append(f"{names[k]} & {r['only']:.3f} [{r['only_ci'][0]:.3f}, {r['only_ci'][1]:.3f}] & {r['fusion']:.3f} [{r['fusion_ci'][0]:.3f}, {r['fusion_ci'][1]:.3f}] & {dd} \\\\")
t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open("outputs/table_aggr.tex", "w").write("\n".join(t)); print("wrote outputs/table_aggr.tex")
