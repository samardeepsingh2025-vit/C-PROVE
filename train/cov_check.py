"""Gold coverage on the SAME 118 claims with the SAME definition (per-claim, macro-averaged) for every Table-9 row.
   python -m train.cov_check"""
import json, numpy as np
from collections import defaultdict
from cprove.utils import load_config
from cprove.data import load_corpus
from cprove.schema import Claim, Frame
from cprove.retrieval_eval import MultiRetriever, MODES, rank_metrics
cfg = load_config(None); meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
items = {}
for f in ["data/meetverify/items_pilot.jsonl", "data/meetverify/items_test.jsonl"]:
    for l in open(f, encoding="utf-8"):
        r = json.loads(l); items[r["cid"]] = r
gold = {}
for l in open("data/meetverify/annotations_A1.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["cid"] in items and r["label"] != "UNVERIFIABLE" and r.get("evidence_uids"): gold[r["cid"]] = r["evidence_uids"]
rc = cfg["retrieval"]; ret = MultiRetriever(rc["encoder"], rc["dense_weight"], rc["sparse_weight"], rc["top_k"], rc["context_window"], rc["slot_boost"])
mk = lambda c: Claim(cid=c, text=items[c]["text"], frame=Frame(**{k: v for k, v in (items[c].get("frame") or {}).items() if k in Frame.__dataclass_fields__}))
by = defaultdict(list)
for c in gold: by[items[c]["meeting_id"]].append(c)
rows = [(m, 5) for m in MODES] + [("hybrid", 10)]
print(f"{len(gold)} claims; columns: WR@k raw rank +-2 | coverage of shipped evidence (retrieve_mode, w=2) | mean #lines | micro coverage")
for mode, k in rows:
    wr, cv, nl, hit, tot = [], [], [], 0, 0
    for mid, cids in by.items():
        m = meetings[mid]; pos = {u.uid: i for i, u in enumerate(m.utterances)}; cl = [mk(c) for c in cids]
        rk = ret.rank(m, cl, mode); ev = ret.retrieve_mode(m, cl, mode, k, 2)
        for c in cids:
            g = {pos[u] for u in gold[c] if u in pos}
            if not g: continue
            got = {pos[u] for u in ev[c].uids if u in pos}
            wr.append(rank_metrics(rk[c], g)[f"wrecall@{k}"]); cv.append(len(g & got) / len(g)); nl.append(len(got)); hit += len(g & got); tot += len(g)
    print(f"{mode:8} k={k:<2} n={len(cv)}  WR={np.mean(wr):.3f}  cov={np.mean(cv):.3f}  lines={np.mean(nl):.1f}  micro={hit/tot:.3f}")
