"""Injected errors the cheap score thinks are fine (and true claims it flags). python -m train.misses [items_pkl]"""
import pickle, sys
p = sys.argv[1] if len(sys.argv) > 1 else "cache/items_synth_dev.pkl"
items = pickle.load(open(p, "rb"))
from experiments.common import refine_gold, load_records
_rp = sys.argv[2] if len(sys.argv) > 2 else "data/meetverify/synth_dev.jsonl"
import os
if os.path.exists(_rp):
    refine_gold(items, load_records(_rp))
rows = [(it.obs[c.cid].raw["fact"], it.obs[c.cid].raw["ent"], it.gold[c.cid], getattr(c, "gold_error", "none"), c.text, it.rec["summary_id"])
        for it in items for c in it.claims if c.cid in it.gold]
err = sorted([r for r in rows if r[2] == 0], key=lambda r: -r[0])
print("=== ERRORS scored most faithful (misses) ===")
for r in err[:15]:
    print(f"fact={r[0]:.2f} ent={r[1]:.2f} [{r[3]}] {r[4]}")
ok = sorted([r for r in rows if r[2] == 1], key=lambda r: r[0])
print("\n=== TRUE claims scored least faithful (false alarms) ===")
for r in ok[:10]:
    print(f"fact={r[0]:.2f} ent={r[1]:.2f} {r[4]}")
import numpy as np
print("\nmedian fact: errors", round(float(np.median([r[0] for r in err])), 2), " true", round(float(np.median([r[0] for r in ok])), 2))
print("share of TRUE claims with ent<0.2:", round(float(np.mean([r[1] < 0.2 for r in ok])), 2))