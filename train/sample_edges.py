"""Sample claim-dependency edges for HUMAN validation (blind: no weights, no source).

  python -m train.sample_edges --items cache/items_dev.pkl cache/items_test.pkl --per_type 50
Writes data/meetverify_v2/edges/edge_pack.json (+ private key).  Annotate with:  python -m bench.fresh.edge_app --annotator B1
Then: python -m train.edge_precision --annotators B1 B2
"""
import argparse, json, pickle, random
from collections import defaultdict, Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--items", nargs="+", default=["cache/frozen/items_dev.pkl", "cache/frozen/items_test.pkl"])
ap.add_argument("--per_type", type=int, default=50); ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--out", default="data/meetverify_v2/edges")
a = ap.parse_args()
rng = random.Random(a.seed)
pool = defaultdict(list); counts = Counter()
for f in a.items:
    for it in pickle.load(open(f, "rb")):
        txt = {c.cid: c.text for c in it.claims}; sent = {c.cid: c.sent_idx for c in it.claims}
        for e in it.edges:
            if e.src in txt and e.dst in txt:
                counts[e.etype] += 1
                pool[e.etype].append({"meeting_id": it.meeting_id, "summary_id": it.rec["summary_id"], "type": e.etype, "a": txt[e.src], "b": txt[e.dst],
                                      "_src": e.src, "_dst": e.dst, "_weight": float(e.weight), "_source": e.source,
                                      "_adjacent": abs(sent[e.src] - sent[e.dst]) <= 1})
print("edge counts per type:", dict(counts))
pack, key = [], {}
for t, v in pool.items():
    rng.shuffle(v)
    for e in v[: a.per_type]:
        pack.append({k: x for k, x in e.items() if not k.startswith("_")})
rng.shuffle(pack)
for i, e in enumerate(pack):
    e["edge_id"] = f"E{i + 1:04d}"
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
json.dump(pack, open(out / "edge_pack.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
json.dump({"counts": dict(counts), "per_type": a.per_type}, open(out / "edge_stats.json", "w"))
print(f"{len(pack)} edges to annotate -> {out}/edge_pack.json")
