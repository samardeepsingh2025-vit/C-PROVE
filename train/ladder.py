"""E2 ladder table (CPU, ~1 min): NLI -> +vetoes -> +fusion (ent+con) -> +attribution/status -> +MiniCheck -> +graph, all under nested LOMO.
  python -m train.ladder
Each fusion rung refits feature weights, Platt calibration inside every outer fold. Paired meeting-clustered bootstrap of each rung against the previous one.
Vetoes rung = entailment capped at 0.25 where (unseen number or p_con>=0.9). Graph rung adds the hand-weight HL-MRF posterior WITH the graph as one more feature.
-> outputs/table_ladder.tex"""
import argparse, json
import numpy as np
from cprove import metrics as M
import train.nested_cv as N

ap = argparse.ArgumentParser(); ap.add_argument("--features", default="cache/frozen/features.jsonl"); a = ap.parse_args()
rows = N.load_rows(a.features)
N.FEATS["graph"] = ["y_graph"]
y = np.array([int(r["author_label"] == "SUPPORTED") for r in rows]); g = np.array([r["meeting_id"] for r in rows])
ent = np.array([r["ent"] for r in rows]); veto = (np.array([r["num_unseen"] for r in rows]) > 0) | (np.array([r["con"] for r in rows]) >= 0.9)
fixed = {"fusion ent+con": ["ent", "con"], "+ attribution + status": ["ent", "con", "attr", "sem"],
         "+ MiniCheck": ["ent", "con", "attr", "sem", "mc"], "+ claim graph": ["ent", "con", "attr", "sem", "mc", "graph"]}
out, _ = N.run(rows, cands={"ent": ["ent"]}, fixed=fixed)
rungs = [("1. NLI entailment", ent), ("2. + vetoes (cap 0.25)", np.where(veto, np.minimum(ent, 0.25), ent)), ("3. + learned fusion (ent, con)", out["fusion ent+con"]),
         ("4. + attribution and status", out["+ attribution + status"]), ("5. + MiniCheck", out["+ MiniCheck"]), ("6. + claim graph", out["+ claim graph"])]
L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Configuration & AUROC [95\% CI] & Brier & $\Delta$ AUROC vs previous (paired $p$) \\", r"\midrule"]
prev = None
print(f"{'rung':36}{'AUROC [CI]':>24}{'Brier':>8}{'d vs previous (p)':>24}")
for nm, p in rungs:
    lo, hi = M.cluster_bootstrap(M.auroc, y, p, g, 1000); d = ""
    if prev is not None:
        dd, l2, h2, pv = M.paired_bootstrap(M.auroc, y, p, prev, g, 1000); d = f"{dd:+.3f} [{l2:+.3f}, {h2:+.3f}] ({pv:.2f})"
    print(f"{nm:36}{M.auroc(y, p):>8.3f} [{lo:.3f},{hi:.3f}]{M.brier(y, p):>8.3f}{d:>34}")
    L.append(f"{nm} & {M.auroc(y, p):.3f} [{lo:.3f}, {hi:.3f}] & {M.brier(y, p):.3f} & {d or '--'} \\\\".replace("+", "+"))
    prev = p
L += [r"\bottomrule", r"\end{tabular}"]; open("outputs/table_ladder.tex", "w").write("\n".join(L)); print("wrote outputs/table_ladder.tex")
