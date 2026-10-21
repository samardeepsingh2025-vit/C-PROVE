"""Silver labels for the two small classifiers, distilled from Qwen2.5-7B (or from AMI
dialogue acts when no LLM is available).

  status : utterance window (+-2) -> one of 8 statuses          -> data/silver/status.jsonl
  edges  : pair of claims from the same summary -> edge type    -> data/silver/edges.jsonl

  python -m train.label_silver --task status --n 6000
  python -m train.label_silver --task edges  --summaries data/meetverify/summaries_train.jsonl --n 4000

Why distil? The 7B judge costs ~1.5 s/call on an RTX 5060; a DeBERTa-v3-base classifier
runs ~400 windows/s, so the cheap atoms stay cheap and the LLM is used only where VoI says.
Report silver-vs-gold agreement on the MeetVerify dev split in the paper.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from cprove.utils import load_config, seed_everything
from cprove.llm import make_llm
from cprove.data import load_corpus, meeting_split
from cprove.prompts import STATUS_ENUM, EDGE_SCHEMA, EDGE_SYSTEM, EDGE_USER
from cprove.text import guess_status

STATUS_SCHEMA = {"type": "object", "properties": {"status": {"type": "string", "enum": STATUS_ENUM},
                                                   "topic": {"type": "string"}}, "required": ["status", "topic"]}
STATUS_SYSTEM = """Read a short meeting excerpt. Classify what the CENTRAL (marked >>) utterance does to its topic:
proposed (suggests an option), agreed (accepts someone's option), decided (finalises), rejected (drops an option),
assigned (gives/takes a task), deferred (postpones), revised (changes an earlier position), stated (plain information).
Use the surrounding lines only as context. Output JSON."""

# AMI dialogue acts -> status prior (used when no LLM is available)
DA2STATUS = {"suggest": "proposed", "offer": "assigned", "be-positive": "agreed", "be-negative": "rejected",
             "assess": "agreed", "inform": "stated", "elicit-offer-or-suggestion": "stated"}


def status_task(cfg, n, out):
    meetings, _ = load_corpus(cfg)
    split = meeting_split([m.meeting_id for m in meetings], cfg["data"]["split_ratio"], cfg["data"]["split_seed"])
    train = [m for m in meetings if m.meeting_id in set(split["train"])]
    llm = make_llm(cfg["llm"])
    rng = random.Random(cfg["seed"])
    pool = [(m, j) for m in train for j, u in enumerate(m.utterances) if len(u.text.split()) >= 4]
    rng.shuffle(pool)
    # oversample likely non-'stated' windows so classes are not 80% 'stated'
    pool.sort(key=lambda x: guess_status(x[0].utterances[x[1]].text) == "stated")
    pool = pool[: int(n * 0.6)] + rng.sample(pool[int(n * 0.6):], min(len(pool) - int(n * 0.6), int(n * 0.4)))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    k = 0
    with open(out, "w", encoding="utf-8") as f:
        for m, j in pool:
            win = m.window(j, 2)
            text = "\n".join((">> " if u.idx == j else "   ") + f"{u.speaker}: {u.text}" for u in win)
            lab = None
            if llm.available:
                r = llm.json(STATUS_SYSTEM, text, STATUS_SCHEMA, max_tokens=60)
                lab = r and r.get("status")
            if lab is None:
                da = m.utterances[j].dialogue_act
                lab = guess_status(m.utterances[j].text)
                if lab == "stated" and da in DA2STATUS:
                    lab = DA2STATUS[da]
            f.write(json.dumps({"text": text, "label": lab,   # same >>-marked format the LLM saw and dim_sem uses
                                "meeting_id": m.meeting_id, "src": "llm" if llm.available else "rules"}) + "\n")
            k += 1
            if k % 200 == 0:
                print(k)
    print("->", out)


def edges_task(cfg, summaries, n, out):
    from cprove.pipeline import CProve
    from cprove.graph import candidate_pairs, rule_edges, EdgeClassifier
    cp = CProve(cfg)
    meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
    llm = cp.llm
    rows = []
    for line in open(summaries, encoding="utf-8"):
        s = json.loads(line)
        claims = cp.decompose(meetings[s["meeting_id"]], s["summary"], s["summary_id"])
        for a, b in candidate_pairs(claims, cfg["graph"]["window_sents"]):
            lab, conf = "none", 1.0
            if llm.available:
                r = llm.json(EDGE_SYSTEM, EDGE_USER.format(a=a.text, b=b.text), EDGE_SCHEMA, max_tokens=40)
                if r:
                    lab, conf = r["relation"], float(r.get("confidence", 0.7))
            else:
                es = [e for e in rule_edges(a, b) if e.etype != "same_agent"]
                if es:
                    lab, conf = es[0].etype, es[0].weight
            rows.append({"a": EdgeClassifier.fmt(a), "b": EdgeClassifier.fmt(b), "label": lab, "conf": conf,
                         "summary_id": s["summary_id"]})
            if len(rows) >= n:
                break
        if len(rows) >= n:
            break
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "->", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["status", "edges"], required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--summaries", default="data/meetverify/summaries_train.jsonl")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg["seed"])
    if a.task == "status":
        status_task(cfg, a.n, a.out or "data/silver/status.jsonl")
    else:
        edges_task(cfg, a.summaries, a.n, a.out or "data/silver/edges.jsonl")


if __name__ == "__main__":
    main()
