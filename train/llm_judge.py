"""E2: zero-shot LLM judge baseline on the SAME labelled claims (GPU via Ollama, ~20-40 min for 523 claims; resumable).
  python -m train.llm_judge                       # Qwen2.5-7B (the model in configs/frozen.yaml)
  python -m train.llm_judge --model llama3.1:8b --name llamajudge
Prompt: claim + the transcript lines retrieved/grounded for it (+-2 lines of context; if none, the 12 lines with highest word overlap).
The judge answers JSON {"verdict": "SUPPORTED"|"NOT_SUPPORTED", "confidence": 0-100}; score = P(supported) = conf/100 if SUPPORTED else 1-conf/100.
Verbalised confidence, not token log-probabilities. No labels, no other C-PROVE scores are given to the judge. Writes cache/frozen/baseline_<name>.json,
which train.nested_cv / train.full_metrics pick up automatically as a baseline row."""
import argparse, json, re
from pathlib import Path
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="cache/frozen/features.jsonl"); ap.add_argument("--meetings", default="data/ami_meetings.jsonl")
ap.add_argument("--model", default="qwen2.5:7b-instruct-q4_K_M"); ap.add_argument("--name", default="qwenjudge")
ap.add_argument("--url", default="http://localhost:11434"); ap.add_argument("--window", type=int, default=2)
ap.add_argument("--out_dir", default="cache/frozen")
a = ap.parse_args()
rows, seen = [], set()
for l in open(a.features, encoding="utf-8"):
    if not l.strip(): continue
    r = json.loads(l)
    if r["cid"] in seen or r.get("author_label") not in ("SUPPORTED", "UNSUPPORTED"): continue
    seen.add(r["cid"]); rows.append(r)
trans = {}
for l in open(a.meetings, encoding="utf-8"):
    r = json.loads(l); trans[r["meeting_id"]] = [f"[{i}] {u['speaker']}: {u['text']}" for i, u in enumerate(r["utterances"])]
W = lambda t: set(re.findall(r"[a-z0-9.]+", t.lower()))
SYS = ("You check claims from a meeting summary against the meeting transcript. Answer SUPPORTED only if the transcript lines clearly show the claim is true "
       "(who said or decided it, numbers, negation, and whether it was decided rather than only discussed). Otherwise answer NOT_SUPPORTED. "
       'Reply with JSON only: {"verdict": "SUPPORTED" or "NOT_SUPPORTED", "confidence": integer 0-100 = how sure you are of your verdict}.')
out_f = Path(a.out_dir) / f"baseline_{a.name}.json"; part = Path(a.out_dir) / f"baseline_{a.name}.partial.json"
scores = json.load(open(part)) if part.exists() else {}
bad = 0
for k, r in enumerate(rows):
    if r["cid"] in scores: continue
    lines = trans[r["meeting_id"]]; g = [int(x) for x in (r.get("grounding") or []) if str(x).lstrip("-").isdigit()]
    if not g:
        cw = W(r["text"]); g = sorted(range(len(lines)), key=lambda i: -len(cw & W(lines[i])))[:12]
    keep = sorted({j for i in g for j in range(i - a.window, i + a.window + 1) if 0 <= j < len(lines)})
    ctx = "\n".join(lines[j] for j in keep)
    body = {"model": a.model, "stream": False, "format": "json", "options": {"temperature": 0, "num_ctx": 8192},
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": f"Transcript lines:\n{ctx}\n\nClaim: {r['text']}\n\nJSON answer:"}]}
    try:
        txt = requests.post(a.url + "/api/chat", json=body, timeout=600).json()["message"]["content"]
        d = json.loads(txt); c = max(0, min(100, float(d.get("confidence", 50)))) / 100; v = str(d.get("verdict", "")).upper()
        scores[r["cid"]] = c if v == "SUPPORTED" else (1 - c if v.startswith("NOT") else 0.5)
    except Exception as e:
        bad += 1; scores[r["cid"]] = 0.5
    if (k + 1) % 25 == 0:
        json.dump(scores, open(part, "w")); print(f"  {k + 1}/{len(rows)} (parse failures so far: {bad})", flush=True)
json.dump(scores, open(out_f, "w")); part.exists() and part.unlink()
print(f"{len(scores)} claims scored, {bad} parse/format failures (scored 0.5) -> {out_f}")
