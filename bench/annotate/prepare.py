"""Build annotation items: decompose each system summary into claims (Stage 1),
retrieve evidence (Stage 3), and sample a stratified set (~1,500 claims) for MeetVerify-Human.

  python -m bench.annotate.prepare --summaries data/meetverify/summaries_test.jsonl --n 1500
Stratification: equal claims per system; within system, all decision/action-item claims
first (they are rarer and matter more for MTI), then random facts. 15% of items are
duplicated across annotators for agreement (Cohen/Fleiss kappa target >= 0.6).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from cprove.utils import load_config
from cprove.data import load_corpus
from cprove.pipeline import CProve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out", default="data/meetverify/items.jsonl")
    ap.add_argument("--overlap", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cfg = load_config()
    rng = random.Random(a.seed)
    meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
    cp = CProve(cfg)
    per_sys = defaultdict(list)
    for line in open(a.summaries, encoding="utf-8"):
        s = json.loads(line)
        if s.get("system") == "reference":
            continue
        m = meetings[s["meeting_id"]]
        claims = cp.decompose(m, s["summary"], s["summary_id"])
        evid = cp.retriever.retrieve(m, claims)
        sents = [c.text for c in claims]
        for c in claims:
            e = evid[c.cid]
            lo = max(0, min(int(u.split("_u")[-1]) for u in e.uids) - 8)
            hi = min(len(m.utterances), max(int(u.split("_u")[-1]) for u in e.uids) + 8)
            per_sys[s["system"]].append({
                "cid": c.cid, "summary_id": s["summary_id"], "system": s["system"], "meeting_id": m.meeting_id,
                "text": c.text, "frame": asdict(c.frame), "claim_type": c.claim_type,
                "summary_context": sents[max(0, sents.index(c.text) - 2): sents.index(c.text) + 3],
                "evidence": e.text,
                "full_window": "\n".join(f"[{u.uid}] {u.speaker}: {u.text}" for u in m.utterances[lo:hi])})
    k = a.n // max(1, len(per_sys))
    items = []
    for sysn, rows in per_sys.items():
        pri = [r for r in rows if r["claim_type"] in ("decision", "action_item")]
        rest = [r for r in rows if r not in pri]
        rng.shuffle(pri); rng.shuffle(rest)
        items += (pri + rest)[:k]
    rng.shuffle(items)
    for it in items:
        it["double"] = rng.random() < a.overlap
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} items -> {a.out}  ({sum(i['double'] for i in items)} double-annotated)")


if __name__ == "__main__":
    main()
