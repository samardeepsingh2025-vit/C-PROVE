"""Inter-annotator agreement + adjudication -> gold file.

  python -m bench.annotate.agreement --files data/meetverify/annotations_A1.jsonl data/meetverify/annotations_A2.jsonl
Outputs Cohen's kappa (2 annotators) or Fleiss' kappa (3+), per-label confusion, and
writes gold.jsonl using majority vote (ties -> 'adjudicate' list for a third pass).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

LABS = ["SUPPORTED", "UNSUPPORTED", "UNVERIFIABLE"]


def cohen(a, b):
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(a, b))


def fleiss(table):
    """table: N x K counts."""
    t = np.asarray(table, float)
    n = t.sum(1)[0]
    p = t.sum(0) / t.sum()
    P = ((t * t).sum(1) - n) / (n * (n - 1))
    Pbar, Pe = P.mean(), (p * p).sum()
    return float((Pbar - Pe) / (1 - Pe)) if Pe < 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--out", default="data/meetverify/gold.jsonl")
    a = ap.parse_args()
    by = defaultdict(dict)
    meta = {}
    for f in a.files:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            by[r["cid"]][r["annotator"]] = r
            meta[r["cid"]] = r
    ann = sorted({x for d in by.values() for x in d})
    shared = [c for c, d in by.items() if len(d) >= 2]
    print(f"{len(by)} claims, {len(shared)} multiply annotated, annotators={ann}")
    if len(ann) == 2 and shared:
        x = [by[c][ann[0]]["label"] for c in shared if ann[0] in by[c] and ann[1] in by[c]]
        y = [by[c][ann[1]]["label"] for c in shared if ann[0] in by[c] and ann[1] in by[c]]
        print("Cohen kappa (label):", round(cohen(x, y), 3))
        xb = [l == "SUPPORTED" for l in x]; yb = [l == "SUPPORTED" for l in y]
        print("Cohen kappa (binary):", round(cohen(xb, yb), 3))
    elif shared:
        tab = [[sum(r["label"] == l for r in by[c].values()) for l in LABS] for c in shared
               if len(by[c]) == len(ann)]
        if tab:
            print("Fleiss kappa:", round(fleiss(tab), 3))
    adjud, n = [], 0
    with open(a.out, "w", encoding="utf-8") as g:
        for c, d in by.items():
            cnt = Counter(r["label"] for r in d.values()).most_common()
            if len(cnt) > 1 and cnt[0][1] == cnt[1][1]:
                adjud.append(c)
                continue
            lab = cnt[0][0]
            errs = Counter(r["error_type"] for r in d.values() if r["label"] == lab).most_common(1)
            g.write(json.dumps({"cid": c, "summary_id": meta[c]["summary_id"], "label": lab,
                                "gold": 1 if lab == "SUPPORTED" else 0, "error_type": errs[0][0] if errs else "none",
                                "evidence_uids": sorted({u for r in d.values() for u in r["evidence_uids"]}),
                                "edges": [r["edges"] for r in d.values() if r.get("edges")]}) + "\n")
            n += 1
    print(f"gold: {n} claims -> {a.out}; {len(adjud)} ties need adjudication")
    if adjud:
        open(a.out.replace(".jsonl", "_adjudicate.txt"), "w").write("\n".join(adjud))


if __name__ == "__main__":
    main()
