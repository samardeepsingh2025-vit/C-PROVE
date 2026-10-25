"""MiniCheck-free fusion vs MiniCheck, paired and Holm-corrected, for both label treatments (CPU, a few minutes).
  python -m train.mcfree
Candidate feature sets are restricted to those WITHOUT the MiniCheck feature; everything else (inner CV, Platt, threshold) is unchanged."""
import argparse, json
import numpy as np
from cprove import metrics as M
from train.nested_cv import load_rows, run, baselines, CANDIDATES

ap = argparse.ArgumentParser(); ap.add_argument("--features", default="cache/frozen/features.jsonl"); ap.add_argument("--boots", type=int, default=1000)
a = ap.parse_args()
NOMC = {k: v for k, v in CANDIDATES.items() if "mc" not in v}
print("MiniCheck-free candidate sets:", list(NOMC))
res = {}
for tag, unver in (("excluded (523)", False), ("as unsupported (552)", True)):
    rows = load_rows(a.features, unver_as_unsup=unver)
    y = np.array([int(r["author_label"] == "SUPPORTED") for r in rows]); g = np.array([r["meeting_id"] for r in rows])
    full, _ = run(rows, fixed={"fusion +mc": ["ent", "con", "attr", "sem", "mc"]})
    nomc, ch = run(rows, cands=NOMC)
    b = baselines(rows)
    P = {"fusion (selected)": full["selected"], "fusion +mc": full["fusion +mc"], "hand weights, graph": b["cprove_hand_graph"],
         "entailment": b["entailment"], "fusion, MiniCheck-free": nomc["selected"]}
    ref = b["minicheck"]; raw, out = {}, {}
    print(f"\n=== {tag}: {len(y)} claims; minicheck AUROC {M.auroc(y, ref):.3f}")
    for k, p in P.items():
        d, lo, hi, pv = M.paired_bootstrap(M.auroc, y, p, ref, g); raw[k] = pv
        out[k] = {"auroc": M.auroc(y, p), "auroc_ci": M.cluster_bootstrap(M.auroc, y, p, g, a.boots), "diff": d, "ci": [lo, hi], "raw_p": pv}
    four = {k: raw[k] for k in list(raw)[:4]}; h4 = M.holm(four); h5 = M.holm(raw)
    for k in P:
        o = out[k]; o["holm_4"] = h4.get(k); o["holm_5"] = h5[k]
        print(f"{k:26} AUROC {o['auroc']:.3f} [{o['auroc_ci'][0]:.3f},{o['auroc_ci'][1]:.3f}]  diff {o['diff']:+.3f} [{o['ci'][0]:+.3f},{o['ci'][1]:+.3f}]  raw p {o['raw_p']:.3f}  Holm(4) {o['holm_4'] if o['holm_4'] is None else round(o['holm_4'],3)}  Holm(5) {o['holm_5']:.3f}")
    from collections import Counter
    print("MiniCheck-free sets chosen:", dict(Counter(c["features"] for c in ch))); res[tag] = out
json.dump(res, open("outputs/mcfree.json", "w"), indent=1, default=float)
