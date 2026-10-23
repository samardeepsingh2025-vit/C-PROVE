"""Does graph propagation change verdicts, and are the changes right? (CPU, seconds)

  python -m train.graph_eval --features cache/frozen/features.jsonl
Compares the hand-weight HL-MRF posterior WITH the dependency graph (y_graph) and WITHOUT it (y_nograph) on the same claims:
number of verdict flips (threshold 0.5), beneficial vs harmful flips against the human-reviewed labels, propagation-induced
false positives (supported claims pushed below 0.5 by the graph), AUROC difference with a paired meeting-clustered bootstrap,
and the list of flipped claims (outputs/graph_flips.csv) for inspection.  Edge precision comes from the separate edge annotation
(train.sample_edges + bench.fresh.edge_app + train.edge_precision).
"""
import argparse, csv, json
import numpy as np
from cprove import metrics as M
from train.nested_cv import load_rows

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="cache/frozen/features.jsonl"); ap.add_argument("--out", default="outputs/graph_eval.json")
a = ap.parse_args()
rows = load_rows(a.features)
y = np.array([int(r["author_label"] == "SUPPORTED") for r in rows]); g = np.array([r["meeting_id"] for r in rows])
pg = np.array([r["y_graph"] for r in rows]); pn = np.array([r["y_nograph"] for r in rows])
dg, dn = pg >= 0.5, pn >= 0.5; flip = dg != dn
good = flip & (dg == (y == 1)); bad = flip & (dn == (y == 1))
fp_induced = flip & (~dg) & (y == 1); fn_induced = flip & dg & (y == 0)
print(f"{len(rows)} claims; verdict flips caused by the graph: {int(flip.sum())} ({flip.mean():.1%})")
print(f"  beneficial flips {int(good.sum())}   harmful flips {int(bad.sum())}   (net {int(good.sum() - bad.sum()):+d})")
print(f"  propagation-induced false alarms (supported pushed to unsupported) {int(fp_induced.sum())}; induced misses {int(fn_induced.sum())}")
d, lo, hi, p = M.paired_bootstrap(M.auroc, y, pg, pn, g, 2000)
print(f"AUROC graph {M.auroc(y, pg):.3f} vs no graph {M.auroc(y, pn):.3f}: diff {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3f}")
d2, lo2, hi2, p2 = M.paired_bootstrap(M.auprc, y, pg, pn, g, 2000)
print(f"AUPRC diff {d2:+.4f} [{lo2:+.4f},{hi2:+.4f}] p={p2:.3f}")
with open("outputs/graph_flips.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["cid", "claim", "y_nograph", "y_graph", "label", "flip_correct"])
    for i in np.where(flip)[0]:
        w.writerow([rows[i]["cid"], rows[i]["text"], round(pn[i], 3), round(pg[i], 3), rows[i]["author_label"], bool(good[i])])
json.dump({"flips": int(flip.sum()), "beneficial": int(good.sum()), "harmful": int(bad.sum()), "induced_false_alarms": int(fp_induced.sum()),
           "auroc_diff": [d, lo, hi, p], "auprc_diff": [d2, lo2, hi2, p2]}, open(a.out, "w"), indent=1)
