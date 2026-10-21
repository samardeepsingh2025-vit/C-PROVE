"""Stage 0 for MeetVerify-Human: generate abstractive summaries with three local
open LLMs (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B; all GGUF Q4_K_M, one at a time on 8 GB).

Long AMI transcripts (~6-12k tokens) are summarised map-reduce style: chunks of
~3k tokens -> partial notes -> final summary, so n_ctx=8192 is enough.

  python -m bench.generate --split test --out data/meetverify/summaries.jsonl
Also writes the AMI reference abstracts (gold-faithful by construction) for MeetVerify-Synth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cprove.utils import load_config, seed_everything, SLOT
from cprove.llm import make_llm
from cprove.data import load_corpus, meeting_split
from cprove.prompts import SUMMARY_SYSTEM, SUMMARY_USER

CHUNK_WORDS = 2200


def transcript_chunks(m, words=CHUNK_WORDS):
    buf, n, out = [], 0, []
    for u in m.utterances:
        line = f"{u.speaker}: {u.text}"
        buf.append(line)
        n += len(line.split())
        if n >= words:
            out.append("\n".join(buf)); buf, n = [], 0
    if buf:
        out.append("\n".join(buf))
    return out


def summarise(llm, m):
    parts = m.meta.get("participants") or m.speakers
    chunks = transcript_chunks(m)
    if len(chunks) == 1:
        return llm.text(SUMMARY_SYSTEM, SUMMARY_USER.format(participants=", ".join(parts), transcript=chunks[0]), 700)
    notes = []
    for k, ch in enumerate(chunks):
        notes.append(llm.text("Write concise bullet notes of this meeting segment: topics, proposals (who), decisions (who), "
                              "action items (owner, deadline), changes of plan. Keep speaker roles.",
                              f"Participants: {', '.join(parts)}\nSegment {k + 1}/{len(chunks)}:\n{ch}", 500) or "")
    return llm.text(SUMMARY_SYSTEM, SUMMARY_USER.format(participants=", ".join(parts),
                    transcript="(segment notes in order)\n" + "\n\n".join(notes)), 700)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--max_meetings", type=int, default=None)
    ap.add_argument("--refs_only", action="store_true", help="only write AMI reference abstracts (input for MeetVerify-Synth)")
    a = ap.parse_args()
    cfg = load_config()
    seed_everything(cfg["seed"])
    meetings, refs = load_corpus(cfg)
    split = meeting_split([m.meeting_id for m in meetings], cfg["data"]["split_ratio"], cfg["data"]["split_seed"])
    keep = set(split[a.split]) if a.split != "all" else {m.meeting_id for m in meetings}
    ms = [m for m in meetings if m.meeting_id in keep][: a.max_meetings]
    out = Path(a.out or Path(cfg["paths"]["bench"]) / f"summaries_{a.split}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for m in ms:   # reference abstracts
            r = refs.get(m.meeting_id, {})
            text = " ".join(r.get("abstract", []) + r.get("decisions", []) + r.get("actions", []))
            if text:
                f.write(json.dumps({"summary_id": f"{m.meeting_id}__ref", "meeting_id": m.meeting_id, "system": "reference",
                                    "summary": text}) + "\n")
        for sysd in ([] if a.refs_only else cfg["summarisers"]):
            if a.systems and sysd["name"] not in a.systems:
                continue
            llm = make_llm(cfg["llm"], sysd["gguf_path"])
            if not llm.available:
                print(f"[skip] {sysd['name']}: model not available")
                continue
            for m in ms:
                s = summarise(llm, m)
                if s:
                    f.write(json.dumps({"summary_id": f"{m.meeting_id}__{sysd['name']}", "meeting_id": m.meeting_id,
                                        "system": sysd["name"], "summary": s.strip()}, ensure_ascii=False) + "\n")
                    f.flush()
                    print("ok", m.meeting_id, sysd["name"])
            SLOT.release()     # unload before the next system
    print("->", out)


if __name__ == "__main__":
    main()
