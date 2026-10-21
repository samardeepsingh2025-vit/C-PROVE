"""Where do attr / ctx / sem go wrong on REAL claims? python -m train.dimerrors sem --ann ... --cache ..."""
import argparse, json, pickle
ap = argparse.ArgumentParser()
ap.add_argument("dim"); ap.add_argument("--ann", required=True); ap.add_argument("--cache", required=True); ap.add_argument("--k", type=int, default=7)
a = ap.parse_args()
ann = {}
for l in open(a.ann, encoding="utf-8"):
    r = json.loads(l); ann[r["cid"]] = r
items = pickle.load(open(a.cache, "rb"))
rows = []
for it in items:
    for c in it.claims:
        r = ann.get(c.cid)
        v = it.obs[c.cid].raw.get(a.dim)
        if r and r["label"] != "UNVERIFIABLE" and v is not None:
            rows.append((v, r["label"], r["error_type"], c))
print(f"{a.dim}: {len(rows)} scored claims; {sum(r[1]=='UNSUPPORTED' for r in rows)} unsupported")
def show(r):
    f = r[3].frame
    print(f"  {r[0]:.2f} [{r[1][:5]} {r[2]}] {r[3].text} | status={f.status} agent={f.agent} recip={f.recipient}")
print("FALSE ALARMS (human: SUPPORTED, dimension score lowest):")
for r in sorted([r for r in rows if r[1] == "SUPPORTED"], key=lambda r: r[0])[: a.k]: show(r)
print("MISSES (human: UNSUPPORTED, dimension score highest):")
for r in sorted([r for r in rows if r[1] == "UNSUPPORTED"], key=lambda r: -r[0])[: a.k]: show(r)
import numpy as np
for lab in ("SUPPORTED", "UNSUPPORTED"):
    v = [r[0] for r in rows if r[1] == lab]
    print(f"mean {a.dim} for {lab}: {np.mean(v):.2f} (n={len(v)})")
