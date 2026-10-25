"""Full metric suite from the pooled OUT-OF-FOLD scores (CPU, ~1-3 min, no refit).  Unsupported = positive class for AUPRC / P / R / F1.

  python -m train.full_metrics            # needs outputs/nested_cv.preds.npy (+ outputs/nested_cv.json for inner thresholds)
Writes: outputs/full_metrics.json, outputs/table_main.tex, outputs/table_ece_variants.tex, outputs/reliability.png / .pdf
Threshold for P/R/F1: for each held-out meeting, the macro-F1-optimal threshold computed on the OTHER meetings' out-of-fold scores
(same rule for every method). For the selected fusion the inner-CV-chosen threshold is also reported."""
import argparse, json
import numpy as np
from cprove import metrics as M
from train.nested_cv import best_threshold, load_rows

ap = argparse.ArgumentParser()
ap.add_argument("--preds", default="outputs/nested_cv.preds.npy"); ap.add_argument("--features", default="cache/frozen/features.jsonl")
ap.add_argument("--cv", default="outputs/nested_cv.json"); ap.add_argument("--boots", type=int, default=1000)
a = ap.parse_args()
P = np.load(a.preds, allow_pickle=True).item(); cids = P.pop("cids")
rows = {r["cid"]: r for r in load_rows(a.features)}
y = np.array([int(rows[c]["author_label"] == "SUPPORTED") for c in cids]); g = np.array([rows[c]["meeting_id"] for c in cids])
ORDER = [("summac", "SummaC-ZS"), ("qwenjudge", "Qwen2.5-7B zero-shot judge"), ("llamajudge", "Llama-3.1-8B zero-shot judge"), ("minicheck", "MiniCheck"), ("minicheck (Platt in fold)", "MiniCheck + Platt"), ("entailment", "Entailment only"),
         ("entailment (Platt in fold)", "Entailment + Platt"), ("cprove_hand_graph", "Hand weights, graph"), ("cprove_hand_nograph", "Hand weights, no graph"),
         ("fusion ent+con", "Fusion: ent+con"), ("fusion +attr+sem", "Fusion: +attr+status"),
         ("fusion (selected in inner CV)", "Fusion (selected)"), ("fusion +mc", "Fusion + MiniCheck")]
ORDER = [(k, n) for k, n in ORDER if k in P and not np.isnan(np.array(P[k], float)).any()]
fold_thr = {}
for k, _ in ORDER:
    p = np.array(P[k], float); t = np.zeros(len(p))
    for m in np.unique(g):
        te = g == m; t[te] = best_threshold(y[~te], p[~te])
    fold_thr[k] = t
inner = None
try:
    ch = {c["meeting"]: c["threshold"] for c in json.load(open(a.cv))["choices"]}
    inner = np.array([ch[m] for m in g])
except Exception:
    pass
res, W = {}, a.boots
print(f"{len(y)} claims, {len(set(g))} meetings, unsupported {(1 - y).mean():.2f}")
print(f"{'method':24} {'AUROC [CI]':22} {'AUPRC-unsup [CI]':22} {'Brier [CI]':20} {'ECE10 [CI]':20} {'LogLoss':8} {'MacroF1':8} P_u    R_u    F1_u   F1_s")
for k, name in ORDER:
    p = np.array(P[k], float); r = {}
    for fn, nm in ((M.auroc, "auroc"), (M.auprc, "auprc_unsup"), (M.brier, "brier"), (M.ece, "ece10")):
        r[nm] = fn(y, p); r[nm + "_ci"] = M.cluster_bootstrap(fn, y, p, g, W)
    r["logloss"] = M.logloss(y, p)
    pred_ok = (p >= fold_thr[k]).astype(int)
    r.update(M.prf(y, pred_ok.astype(float), 0.5))
    r["ece_variants"] = {f"{b}/{mode}": M.ece(y, p, b, mode) for b in (5, 10, 15, 20) for mode in ("width", "mass")}
    res[k] = r
    f = lambda nm: f"{r[nm]:.3f} [{r[nm + '_ci'][0]:.3f},{r[nm + '_ci'][1]:.3f}]"
    print(f"{name:24} {f('auroc'):22} {f('auprc_unsup'):22} {f('brier'):20} {f('ece10'):20} {r['logloss']:<8.3f} {r['macro_f1']:<8.3f} {r['precision_unsup']:.3f}  {r['recall_unsup']:.3f}  {r['f1_unsup']:.3f}  {r['f1_sup']:.3f}")
sel = "fusion (selected in inner CV)"
if sel in res:
    c = res[sel]["confusion"]
    print(f"\nconfusion, selected fusion at per-fold threshold (rows = truth, cols = prediction; unsupported = positive):\n  truth supported   : predicted supported {c[0][0]}, predicted unsupported {c[0][1]}\n  truth unsupported : predicted supported {c[1][0]}, predicted unsupported {c[1][1]}")
    if inner is not None:
        pi = M.prf(y, (np.array(P[sel], float) >= inner).astype(float), 0.5); res[sel]["inner_threshold_prf"] = pi
        print(f"  with the inner-CV-chosen threshold: macro-F1 {pi['macro_f1']:.3f}, P_unsup {pi['precision_unsup']:.3f}, R_unsup {pi['recall_unsup']:.3f}, confusion {pi['confusion']}")
print("\nECE variants (equal-width / equal-mass):")
print(f"{'method':24} " + " ".join(f"{b}w   {b}m  " for b in (5, 10, 15, 20)))
for k, name in ORDER:
    v = res[k]["ece_variants"]; print(f"{name:24} " + " ".join(f"{v[f'{b}/width']:.3f} {v[f'{b}/mass']:.3f}" for b in (5, 10, 15, 20)))
import pathlib; pathlib.Path("outputs").mkdir(exist_ok=True)
json.dump(res, open("outputs/full_metrics.json", "w"), indent=1, default=float)
ci = lambda r, nm: f"{r[nm]:.3f} [{r[nm + '_ci'][0]:.3f}, {r[nm + '_ci'][1]:.3f}]"
t = ["\\begin{table}[htbp]", "\\centering",
     "\\caption{Nested leave-one-meeting-out results over 523 claims from 40 meetings. 95\\% meeting-clustered bootstrap CIs (1000 resamples). AUPRC is for the unsupported class (positive class); ECE uses ten equal-width bins; macro-F1 uses, for each held-out meeting, the threshold that maximises macro-F1 on the other meetings' out-of-fold scores. ``+ Platt'' denotes in-fold Platt scaling.}",
     "\\label{tab:main}", "\\scriptsize", "\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
     "Method & AUROC & AUPRC (unsup.) & Brier & ECE & Log loss & Macro-F1 \\\\", "\\midrule"]
for k, name in ORDER:
    r = res[k]; b = "\\textbf{%s}" if k == sel else "%s"
    t.append(f"{b % name} & {ci(r, 'auroc')} & {ci(r, 'auprc_unsup')} & {ci(r, 'brier')} & {ci(r, 'ece10')} & {r['logloss']:.3f} & {r['macro_f1']:.3f} \\\\")
t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open("outputs/table_main.tex", "w").write("\n".join(t))
t = ["\\begin{table}[htbp]", "\\centering", "\\caption{ECE under different binning schemes (width: equal-width bins; mass: equal-count bins) for the out-of-fold scores.}",
     "\\label{tab:ecevar}", "\\small", "\\begin{tabular}{@{}lcccccccc@{}}", "\\toprule",
     "Method & 5w & 5m & 10w & 10m & 15w & 15m & 20w & 20m \\\\", "\\midrule"]
for k, name in ORDER:
    v = res[k]["ece_variants"]; t.append(name + " & " + " & ".join(f"{v[f'{b}/{m}']:.3f}" for b in (5, 10, 15, 20) for m in ("width", "mass")) + " \\\\")
t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open("outputs/table_ece_variants.tex", "w").write("\n".join(t))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42})
panels = [(k, n) for k, n in [("entailment", "Entailment"), ("minicheck", "MiniCheck"), ("fusion (selected in inner CV)", "Fusion (selected)"),
                              ("entailment (Platt in fold)", "Entailment + Platt"), ("minicheck (Platt in fold)", "MiniCheck + Platt"), ("fusion +mc", "Fusion + MiniCheck")] if k in P]
fig, ax = plt.subplots(2, 3, figsize=(6.4, 4.4), sharex=True, sharey=True, constrained_layout=True)
for A, (k, n) in zip(ax.ravel(), panels):
    p = np.array(P[k], float); e = np.unique(np.quantile(p, np.linspace(0, 1, 11))); xs, ys, ns = [], [], []
    for i in range(len(e) - 1):
        m = (p >= e[i]) & ((p < e[i + 1]) if i < len(e) - 2 else (p <= e[i + 1]))
        if m.any():
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(m.sum())
    A.plot([0, 1], [0, 1], "--", color="grey", lw=1); A.plot(xs, ys, "o-", color="#1f5fa8", ms=4)
    A.set_title(f"{n}\nECE {res[k]['ece10']:.3f}", fontsize=7.5, linespacing=1.15); A.set_xlim(0, 1); A.set_ylim(0, 1)
for A in ax[1]:
    A.set_xlabel("mean predicted P(supported)")
for A in ax[:, 0]:
    A.set_ylabel("observed fraction supported")
fig.savefig("outputs/reliability.pdf"); fig.savefig("outputs/reliability.png", dpi=300) 
print("\nwrote outputs/table_main.tex, table_ece_variants.tex, reliability.png/.pdf, full_metrics.json")
