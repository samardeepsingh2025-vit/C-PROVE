"""Generate summaries (2 summarisers) for the fresh meetings, decompose into claims, and build the BLIND
annotation pack. Annotators never see model scores, drafts, system names or retrieval output.

  python -m bench.fresh.make_pack --meetings data/meetverify_v2/meetings.json --out data/meetverify_v2
      [--systems qwen2.5:7b-instruct-q4_K_M llama3.1:8b]      (Ollama model names; must be pulled)

Outputs in --out:
  summaries.jsonl     private (system names)             - never give to annotators
  claims.jsonl        private (frames, claim types)      - model side
  pack.json           PUBLIC: transcripts + anonymised summaries + claims (what annotators see)
  key_private.json    summary_id -> system / meeting
"""
from __future__ import annotations
import argparse, json, random
from dataclasses import asdict
from pathlib import Path
from cprove.utils import load_config, seed_everything
from cprove.llm import make_llm
from cprove.data import load_corpus
from cprove.text import split_sentences
from cprove.frames import FrameExtractor
from bench.generate import summarise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", default="data/meetverify_v2/meetings.json")
    ap.add_argument("--out", default="data/meetverify_v2")
    ap.add_argument("--systems", nargs="+", default=["qwen2.5:7b-instruct-q4_K_M", "llama3.1:8b"])
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    cfg = load_config(); seed_everything(cfg["seed"])
    ids = json.load(open(a.meetings))["meetings"]
    meetings = {m.meeting_id: m for m in load_corpus(cfg)[0] if m.meeting_id in ids}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # 1. summaries -------------------------------------------------------------------------------
    sums = []
    for sysname in a.systems:
        c = dict(cfg["llm"]); c["backend"] = "ollama"; c["ollama_model"] = sysname
        llm = make_llm(c)
        for mid in ids:
            m = meetings[mid]
            s = summarise(llm, m)
            if not s:
                print(f"[skip] {sysname} {mid}: no output"); continue
            sums.append({"summary_id": f"{mid}__{sysname.split(':')[0].replace('.', '').replace('-', '')}", "meeting_id": mid,
                         "system": sysname, "summary": s.strip()})
            print(f"  summarised {mid} with {sysname}")
    with open(out / "summaries.jsonl", "w", encoding="utf-8") as f:
        for s in sums:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 2. claims (the same decomposer as the main experiments) --------------------------------------
    # same decomposer + LLM backend as the main experiments, but WITHOUT loading the NLI/encoder models
    fx = FrameExtractor(make_llm(cfg["llm"]), cfg["frames"]["max_claims_per_sentence"], cfg["frames"]["use_llm"])
    print(f"decomposing {len(sums)} summaries (LLM backend: {cfg['llm'].get('backend')}) ...", flush=True)
    claims_all, pack_sums, key = [], [], {}
    for k, s in enumerate(sums):
        m = meetings[s["meeting_id"]]
        parts = m.meta.get("participants") or m.speakers
        cl = fx.decompose_summary(s["summary"], s["summary_id"], parts, m.meta.get("date"), False)
        print(f"  [{k + 1}/{len(sums)}] {s['summary_id']}: {len(cl)} claims", flush=True)
        sents = split_sentences(s["summary"])
        for c in cl:
            d = c.to_dict(); d["system"] = s["system"]; d["meeting_id"] = m.meeting_id
            claims_all.append(d)
        pack_sums.append({"sid": None, "meeting_id": m.meeting_id, "sentences": sents,
                          "claims": [{"cid": c.cid, "text": c.text, "sent_idx": c.sent_idx} for c in cl],
                          "_summary_id": s["summary_id"]})
    with open(out / "claims.jsonl", "w", encoding="utf-8") as f:
        for d in claims_all:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # 3. anonymise + shuffle: annotators must not know the system or the original ordering -------
    rng = random.Random(a.seed); rng.shuffle(pack_sums)
    for i, p in enumerate(pack_sums):
        sid = f"S{i + 1:03d}"; key[sid] = {"summary_id": p.pop("_summary_id"), "meeting_id": p["meeting_id"]}
        key[sid]["system"] = next(s["system"] for s in sums if s["summary_id"] == key[sid]["summary_id"])
        p["sid"] = sid
        for j, c in enumerate(p["claims"]):
            c["public_id"] = f"{sid}.{j + 1}"
    transcripts = {mid: [{"n": i + 1, "uid": u.uid, "speaker": u.speaker, "text": u.text}
                         for i, u in enumerate(meetings[mid].utterances)] for mid in ids}
    json.dump({"transcripts": transcripts, "summaries": pack_sums}, open(out / "pack.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(key, open(out / "key_private.json", "w"), indent=1)
    n_claims = sum(len(p["claims"]) for p in pack_sums)
    print(f"{len(pack_sums)} summaries, {n_claims} claims -> {out}/pack.json (give ONLY pack.json to annotators)")


if __name__ == "__main__":
    main()
