"""Why does a dimension fail? python -m train.dimprobe attr attribution_swap   (dims: attr ctx sem)"""
import pickle, sys, random, os
from experiments.common import refine_gold, load_records
dim, et = sys.argv[1], sys.argv[2]
items = pickle.load(open((sys.argv[3] if len(sys.argv) > 3 else "cache/items_synth_dev.pkl"), "rb"))
if os.path.exists("data/meetverify/synth_dev.jsonl"):
    refine_gold(items, load_records("data/meetverify/synth_dev.jsonl"))
def show(c, o, tag):
    f = c.frame
    print(f"{tag} {dim}={o.raw.get(dim)} | {c.text} | agent={f.agent} recip={f.recipient} status={f.status} qty={f.quantity} time={f.time}")
err, ok = [], []
for it in items:
    for c in it.claims:
        if c.cid in it.gold and it.obs[c.cid].raw.get(dim) is not None:
            (err if it.gold[c.cid] == 0 and c.gold_error == et else ok if it.gold[c.cid] == 1 else []).append((c, it.obs[c.cid]))
random.Random(0).shuffle(err); random.Random(1).shuffle(ok)
print(f"errors with a {dim} score: {len(err)}")
for c, o in err[:8]: show(c, o, "ERR")
print()
for c, o in ok[:8]: show(c, o, "OK ")