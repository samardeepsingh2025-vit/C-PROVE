"""E8: veto cap sweep on out-of-fold fusion scores (CPU, ~1 min).
  python -m train.veto_sweep
Caps {0.1,0.25,0.4} x contradiction thresholds {0.8,0.9,0.95}, with and without the numeric veto.
Reports AUROC, Brier, ECE, claims changed (score actually lowered), unsupported/supported among changed. Writes outputs/table_vetosweep.tex."""
import argparse, json
import numpy as np
from cprove import metrics as M
from train.nested_cv import load_rows, run

ap = argparse.ArgumentParser(); ap.add_argument("--features", default="cache/frozen/features.jsonl"); a = ap.parse_args()
rows = load_rows(a.features); out, _ = run(rows); p = np.asarray(out["selected"], float)
y = np.array([int(r["author_label"] == "SUPPORTED") for r in rows]); g = np.array([r["meeting_id"] for r in rows])
con = np.array([r["con"] for r in rows]); num = np.array([r["num_unseen"] for r in rows]) > 0
base = dict(auroc=M.auroc(y, p), brier=M.brier(y, p), ece=M.ece(y, p))
print(f"no veto: AUROC {base['auroc']:.3f} Brier {base['brier']:.3f} ECE {base['ece']:.3f}  (n={len(y)})")
res = [dict(name="none", cap=None, thr=None, numv=False, changed=0, unsup=0, sup=0, **base)]
for numv in (False, True):
    for thr in (0.8, 0.9, 0.95):
        for cap in (0.1, 0.25, 0.4):
            trig = (con >= thr) | (num if numv else False)
            q = np.where(trig, np.minimum(p, cap), p); ch = q < p - 1e-12
            r = dict(name="", cap=cap, thr=thr, numv=numv, changed=int(ch.sum()), unsup=int((ch & (y == 0)).sum()), sup=int((ch & (y == 1)).sum()),
                     auroc=M.auroc(y, q), brier=M.brier(y, q), ece=M.ece(y, q)); res.append(r)
            print(f"num={int(numv)} thr {thr} cap {cap}: AUROC {r['auroc']:.3f} Brier {r['brier']:.3f} ECE {r['ece']:.3f} changed {r['changed']} (unsup {r['unsup']}, sup {r['sup']})")
json.dump(res, open("outputs/veto_sweep.json", "w"), indent=1, default=float)
L = [r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule", r"Numeric veto & $p_{\mathrm{con}}\ge$ & Cap & AUROC & Brier & ECE & Changed & Unsup./Sup. \\", r"\midrule",
     f"-- & -- & -- & {base['auroc']:.3f} & {base['brier']:.3f} & {base['ece']:.3f} & 0 & -- \\\\"]
for r in res[1:]:
    L.append(f"{'yes' if r['numv'] else 'no'} & {r['thr']} & {r['cap']} & {r['auroc']:.3f} & {r['brier']:.3f} & {r['ece']:.3f} & {r['changed']} & {r['unsup']}/{r['sup']} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open("outputs/table_vetosweep.tex", "w").write("\n".join(L)); print("wrote outputs/table_vetosweep.tex")
