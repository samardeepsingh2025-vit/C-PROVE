"""Rule-level validation of the hard vetoes + the 'simultaneous high ent and con' analysis (CPU, seconds).

  python -m train.veto_eval --features cache/frozen/features.jsonl
For each veto (numeric / contradiction at several thresholds): number of claims triggered per set, precision (share of
triggered claims whose human-reviewed label is UNSUPPORTED), recall of all unsupported claims, the false vetoes listed with
claim text and the unseen numbers (outputs/veto_false_positives.csv) for manual inspection, the empirical support rate among
vetoed claims (what the cap value 0.25 should mean if it were a probability), and the effect of applying the veto cap on top of
the nested-CV fusion scores (AUROC / Brier / ECE with and without veto).
"""
import argparse, csv, json
from collections import Counter
import numpy as np
from cprove import metrics as M
from train.nested_cv import load_rows, run

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="cache/frozen/features.jsonl"); ap.add_argument("--out", default="outputs/veto_eval.json")
a = ap.parse_args()
rows = load_rows(a.features)
y = np.array([int(r["author_label"] == "SUPPORTED") for r in rows]); uns = 1 - y
g = np.array([r["meeting_id"] for r in rows]); sets = np.array([r["set"] for r in rows])
num = np.array([r["num_unseen"] for r in rows]) > 0
con = np.array([r["con"] for r in rows]); ent = np.array([r["ent"] for r in rows])
res = {}


def line(name, trig):
    n = int(trig.sum()); tp = int((trig & (uns == 1)).sum())
    prec = tp / n if n else float("nan"); rec = tp / max(1, int(uns.sum()))
    sup_rate = float(y[trig].mean()) if n else float("nan")
    per = {str(s): int((trig & (sets == s)).sum()) for s in sorted(set(sets))}
    res[name] = {"triggered": n, "per_set": per, "precision": prec, "recall_of_unsupported": rec, "support_rate_among_vetoed": sup_rate}
    print(f"{name:28} n={n:<4} per set {per}  precision {prec:.3f}  recall {rec:.3f}  empirical P(supported | vetoed) {sup_rate:.3f}")


print(f"{len(rows)} labelled claims; unsupported {int(uns.sum())}")
line("numeric veto", num)
for t in (0.7, 0.8, 0.9, 0.95):
    line(f"contradiction p_con>={t}", con >= t)
line("numeric OR con>=0.9", num | (con >= 0.9))
fp = [r for r, t in zip(rows, num | (con >= 0.9)) if t and r["author_label"] == "SUPPORTED"]
with open("outputs/veto_false_positives.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["cid", "veto", "claim", "unseen_numbers", "p_con", "p_ent", "author_note"])
    for r in fp:
        w.writerow([r["cid"], "numeric" if r["num_unseen"] else "contradiction", r["text"], r.get("num_unseen_values"), round(r["con"], 3), round(r["ent"], 3), ""])
print(f"false vetoes (supported claims that were vetoed): {len(fp)} -> outputs/veto_false_positives.csv (inspect: paraphrase? unit? arithmetic?)")

both = (ent > 0.5) & (con > 0.5)
print(f"\nsimultaneous high entailment and contradiction (both > 0.5): {int(both.sum())} claims; supported rate {y[both].mean() if both.any() else float('nan'):.3f}; "
      f"how many get the contradiction veto: {int((both & (con >= 0.9)).sum())}")
res["both_high"] = {"n": int(both.sum()), "supported_rate": float(y[both].mean()) if both.any() else None}

# effect of the veto cap on top of the cross-validated fusion
out, _ = run(rows); p = out["selected"]
for t in (0.9,):
    pv = np.where(num | (con >= t), np.minimum(p, 0.25), p)
    for nm, q in (("fusion", p), (f"fusion + veto(num|con>={t}, cap .25)", pv)):
        lo, hi = M.cluster_bootstrap(M.auroc, y, q, g, 500)
        print(f"{nm:38} AUROC {M.auroc(y, q):.3f} [{lo:.3f},{hi:.3f}]  AUPRC {M.auprc(y, q):.3f}  Brier {M.brier(y, q):.3f}  ECE {M.ece(y, q):.3f}")
        res[nm] = {"auroc": M.auroc(y, q), "auprc": M.auprc(y, q), "brier": M.brier(y, q), "ece": M.ece(y, q)}
json.dump(res, open(a.out, "w"), indent=1, default=float)
