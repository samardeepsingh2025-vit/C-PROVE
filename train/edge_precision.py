"""Edge precision per relation type from the human edge annotations (CPU).
  python -m train.edge_precision --annotators B1 [B2] [--dir data/meetverify_v2/edges]
Precision = HOLDS / (HOLDS + ONLY + DOES) over decided edges (UNCLEAR excluded) with Wilson 95% CIs; also the share of
'ONLY SIMILAR WORDS' (lexical similarity / summary order, not a real dependency). With two annotators: Cohen's kappa (binary HOLDS vs not)."""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--annotators", nargs="+", required=True); ap.add_argument("--dir", default="data/meetverify_v2/edges")
a = ap.parse_args(); d = Path(a.dir)
ann = {n: {json.loads(l)["edge_id"]: json.loads(l) for l in open(d / f"edges_{n}.jsonl", encoding="utf-8")} for n in a.annotators}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


ids = set.intersection(*[set(v) for v in ann.values()]) if len(ann) > 1 else set(next(iter(ann.values())))
first = ann[a.annotators[0]]
if len(ann) > 1:
    b = ann[a.annotators[1]]; x = [first[i]["verdict"] == "HOLDS" for i in ids]; y = [b[i]["verdict"] == "HOLDS" for i in ids]
    po = sum(p == q for p, q in zip(x, y)) / len(ids); px, py = sum(x) / len(ids), sum(y) / len(ids); pe = px * py + (1 - px) * (1 - py)
    print(f"{len(ids)} edges double-annotated; agreement {po:.3f}, Cohen's kappa (HOLDS vs not) {(po - pe) / (1 - pe) if pe < 1 else float('nan'):.3f}")
by = defaultdict(lambda: defaultdict(int))
for i in ids:
    by[first[i]["type"]][first[i]["verdict"]] += 1
print(f"{'relation':14} n(decided)  HOLDS  ONLY-SIMILAR  NOT   UNCLEAR   precision [95% CI]")
tot = defaultdict(int)
for t, c in sorted(by.items()):
    n = c["HOLDS"] + c["ONLY"] + c["DOES"]; lo, hi = wilson(c["HOLDS"], n)
    print(f"{t:14} {n:<10} {c['HOLDS']:<6} {c['ONLY']:<13} {c['DOES']:<5} {c['UNCLEAR']:<8}  {c['HOLDS'] / n if n else float('nan'):.2f} [{lo:.2f},{hi:.2f}]")
import os
os.makedirs("outputs", exist_ok=True)
json.dump({t: (c["HOLDS"] / max(1, c["HOLDS"] + c["ONLY"] + c["DOES"])) for t, c in by.items()}, open("outputs/edge_precision.json", "w"), indent=1)
