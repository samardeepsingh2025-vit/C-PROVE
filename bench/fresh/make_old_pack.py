"""Add a stratified, BLIND re-annotation sample of the EXISTING labelled claims to pack.json
(so that annotators cannot tell old from new claims). Used to measure agreement with the author-reviewed labels.

  python -m bench.fresh.make_old_pack --n 150 --pack data/meetverify_v2/pack.json
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path
from cprove.utils import load_config
from cprove.data import load_corpus
from cprove.text import split_sentences


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--pack", default="data/meetverify_v2/pack.json")
    ap.add_argument("--ann", default="data/meetverify/annotations_A1.jsonl")
    ap.add_argument("--items", nargs="+", default=["data/meetverify/items_pilot.jsonl", "data/meetverify/items_test.jsonl"])
    ap.add_argument("--summaries", nargs="+", default=["data/meetverify/summaries_dev.jsonl", "data/meetverify/summaries_test.jsonl"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_meetings", type=int, default=None, help="draw claims from only this many meetings (less transcript reading)")
    ap.add_argument("--quota", default=None, help="e.g. SUPPORTED=22,UNSUPPORTED=22,UNVERIFIABLE=6 (deliberate class balance so kappa is informative)")
    ap.add_argument("--new_pack", action="store_true", help="start an empty pack (agreement-only study) instead of appending")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    ann = {}
    for l in open(a.ann, encoding="utf-8"):
        r = json.loads(l); ann[r["cid"]] = r["label"]
    items = {}
    for f in a.items:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); items[r["cid"]] = r
    sums = {}
    for f in a.summaries:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); sums[r["summary_id"]] = r
    allowed = None
    if a.max_meetings:
        mids = sorted({items[c]["meeting_id"] for c in ann if c in items})
        rng.shuffle(mids); allowed = set(mids[: a.max_meetings])
    pool = defaultdict(list)
    for cid, lab in ann.items():
        if cid in items and (allowed is None or items[cid]["meeting_id"] in allowed):
            pool[lab].append(cid)
    chosen = []
    if a.quota:
        for part in a.quota.split(","):
            lab, q = part.split("="); v = pool.get(lab, []); rng.shuffle(v); chosen += v[: int(q)]
    else:   # stratify by label proportionally (keep all rare classes represented)
        tot = sum(len(v) for v in pool.values())
        for lab, v in pool.items():
            rng.shuffle(v); chosen += v[: max(5, round(a.n * len(v) / tot))]
        chosen = chosen[: a.n]
    pk = {"transcripts": {}, "summaries": []} if a.new_pack else json.load(open(a.pack, encoding="utf-8"))
    meetings = {m.meeting_id: m for m in load_corpus(load_config())[0]}
    key_path = Path(a.pack).parent / "key_private.json"
    key = {} if a.new_pack else json.load(open(key_path))
    by_sum = defaultdict(list)
    for cid in chosen:
        by_sum[items[cid]["summary_id"]].append(cid)
    n0 = len(pk["summaries"])
    for k, (sid_old, cids) in enumerate(by_sum.items()):
        mid = items[cids[0]]["meeting_id"]
        if mid not in pk["transcripts"]:
            m = meetings[mid]
            pk["transcripts"][mid] = [{"n": i + 1, "uid": u.uid, "speaker": u.speaker, "text": u.text} for i, u in enumerate(m.utterances)]
        sid = f"S{n0 + k + 1:03d}"
        sents = split_sentences(sums[sid_old]["summary"]) if sid_old in sums else []
        pk["summaries"].append({"sid": sid, "meeting_id": mid, "sentences": sents, "rate_summary": False,
                                "claims": [{"cid": c, "text": items[c]["text"], "sent_idx": None, "public_id": f"{sid}.{j + 1}"}
                                           for j, c in enumerate(cids)]})
        key[sid] = {"summary_id": sid_old, "meeting_id": mid, "system": "OLD", "old_cids": cids}
    rng.shuffle(pk["summaries"])
    Path(a.pack).parent.mkdir(parents=True, exist_ok=True)
    json.dump(pk, open(a.pack, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(key, open(key_path, "w"), indent=1)
    print(f"added {len(chosen)} old claims in {len(by_sum)} summaries / {len({items[c]['meeting_id'] for c in chosen})} meetings; "
          f"label mix {dict((k, len([c for c in chosen if ann[c]==k])) for k in pool)}")


if __name__ == "__main__":
    main()