"""Evidence-retrieval evaluation: lexical / BM25 / dense / hybrid (shipped) / RRF against gold evidence.

  python -m train.eval_retrieval --ann data/meetverify/annotations_A1.jsonl [--gold data/meetverify_v2/agreement/gold_claims.jsonl]
Two gold sources are reported separately:
  author       evidence lines confirmed by the author for all labelled claims (author-reviewed, weaker)
  independent  evidence lines chosen by independent annotators (gold_claims.jsonl, small but independent)
Also reports a multi-turn subset (gold lines >5 utterances apart: revised decisions / action items).
"""
import argparse, json
import numpy as np
from collections import defaultdict
from cprove.utils import load_config
from cprove.data import load_corpus
from cprove.schema import Claim, Frame
from cprove.retrieval_eval import MultiRetriever, MODES, rank_metrics, is_multi_turn

ap = argparse.ArgumentParser()
ap.add_argument("--ann", default="data/meetverify/annotations_A1.jsonl")
ap.add_argument("--items", nargs="+", default=["data/meetverify/items_pilot.jsonl", "data/meetverify/items_test.jsonl"])
ap.add_argument("--gold", default=None); ap.add_argument("--out", default="outputs/retrieval_eval.json")
ap.add_argument("--no_dense", action="store_true"); ap.add_argument("--config", default=None)
a = ap.parse_args()
cfg = load_config(a.config)
meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
items = {}
for f in a.items:
    for l in open(f, encoding="utf-8"):
        r = json.loads(l); items[r["cid"]] = r
sets = {"author": {}, "independent": {}}
for l in open(a.ann, encoding="utf-8"):
    r = json.loads(l)
    if r["label"] != "UNVERIFIABLE" and r.get("evidence_uids") and r["cid"] in items:
        sets["author"][r["cid"]] = r["evidence_uids"]
if a.gold:
    for l in open(a.gold, encoding="utf-8"):
        r = json.loads(l)
        if r["set"] == "old" and r["gold_evidence_uids"] and r["cid"] in items:
            sets["independent"][r["cid"]] = r["gold_evidence_uids"]
r_cfg = cfg["retrieval"]
ret = MultiRetriever(r_cfg["encoder"], r_cfg["dense_weight"], r_cfg["sparse_weight"], r_cfg["top_k"], r_cfg["context_window"],
                     r_cfg["slot_boost"], use_dense=not a.no_dense)
res = {}
for sname, gold in sets.items():
    if not gold:
        continue
    by_m = defaultdict(list)
    for cid in gold:
        by_m[items[cid]["meeting_id"]].append(cid)
    for mode in MODES:
        rows, rows_mt = [], []
        for mid, cids in by_m.items():
            m = meetings[mid]; pos = {u.uid: i for i, u in enumerate(m.utterances)}
            claims = [Claim(cid=c, text=items[c]["text"], frame=Frame(**{k: v for k, v in (items[c].get("frame") or {}).items() if k in Frame.__dataclass_fields__})) for c in cids]
            ranks = ret.rank(m, claims, mode)
            for c in cids:
                g = {pos[u] for u in gold[c] if u in pos}
                if not g:
                    continue
                mt = rank_metrics(ranks[c], g); rows.append(mt)
                if is_multi_turn(g):
                    rows_mt.append(mt)
        agg = lambda R: {k: round(float(np.mean([r[k] for r in R])), 3) for k in R[0]} | {"n": len(R)} if R else None
        res[f"{sname}/{mode}"] = {"all": agg(rows), "multi_turn": agg(rows_mt)}
print(f"{'set/mode':22} n   R@1   R@3   R@5   R@10  WR@5  MRR   nDCG@5 | multi-turn n  R@5  WR@5  MRR")
for k, v in res.items():
    A, B = v["all"], v["multi_turn"]
    print(f"{k:22} {A['n']:<3} {A['recall@1']:.3f} {A['recall@3']:.3f} {A['recall@5']:.3f} {A['recall@10']:.3f} {A['wrecall@5']:.3f} {A['mrr']:.3f} {A['ndcg@5']:.3f} | "
          + (f"{B['n']:<3} {B['recall@5']:.3f} {B['wrecall@5']:.3f} {B['mrr']:.3f}" if B else "-"))
import pathlib; pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
