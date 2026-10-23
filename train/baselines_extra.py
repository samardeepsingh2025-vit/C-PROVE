"""Missing baselines on the SAME claims (reads the feature table; GPU).  Scores are P(supported)-like values in [0,1].

  python -m train.baselines_extra --which summac           # SummaC-ZS re-implemented on the ORIGINAL ViTC checkpoint
  python -m train.baselines_extra --which alignscore       # official AlignScore-large (needs the package + checkpoint, see below)

SummaC: the released `summac` package fails with current `transformers` (AlbertTokenizer.batch_encode_plus /
`truncation_strategy`). We therefore re-implement SummaC-ZS exactly as published (sentence granularity, score =
max over source sentences of [P(entail) - P(contradict)] with the VitC-MNLI model tals/albert-xlarge-vitaminc-mnli)
by calling the checkpoint directly. Source sentences = transcript lines "Speaker: text".
AlignScore: pip install git+https://github.com/yuh-zha/AlignScore.git ; python -m spacy download en_core_web_sm ;
  curl -L -o models/AlignScore-large.ckpt https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt
Writes cache/frozen/baseline_<name>.json  {cid: score}.
"""
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--which", choices=["summac", "alignscore"], required=True)
ap.add_argument("--features", default="cache/frozen/features.jsonl")
ap.add_argument("--meetings", default="data/ami_meetings.jsonl")
ap.add_argument("--out_dir", default="cache/frozen")
ap.add_argument("--ckpt", default="models/AlignScore-large.ckpt")
ap.add_argument("--batch", type=int, default=256)
a = ap.parse_args()
rows, _seen = [], set()
for l in open(a.features, encoding="utf-8"):
    if not l.strip():
        continue
    r = json.loads(l)
    if r["cid"] in _seen or r.get("author_label") not in ("SUPPORTED", "UNSUPPORTED"):
        continue            # score each LABELLED claim once
    _seen.add(r["cid"]); rows.append(r)
print(len(rows), "labelled claims to score", flush=True)
trans = {}
for l in open(a.meetings, encoding="utf-8"):
    r = json.loads(l); trans[r["meeting_id"]] = [f"{u['speaker']}: {u['text']}" for u in r["utterances"]]
by_m = defaultdict(list)
for r in rows:
    by_m[r["meeting_id"]].append(r)
scores = {}

if a.which == "summac":
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    name = "tals/albert-xlarge-vitaminc-mnli"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).cuda().half().eval()
    print("label map:", model.config.id2label)
    id2 = {int(k): v.lower() for k, v in model.config.id2label.items()}
    ent_i = [k for k, v in id2.items() if v.startswith("support") or v.startswith("entail")][0]
    con_i = [k for k, v in id2.items() if v.startswith("refute") or v.startswith("contra")][0]
    print("entail idx", ent_i, "contradict idx", con_i, flush=True)
    part = Path(a.out_dir) / "baseline_summac.partial.json"
    if part.exists():
        scores.update(json.load(open(part)))
    for k, (mid, rs) in enumerate(by_m.items()):
        lines = [t for t in trans[mid] if len(t.split()) >= 4]
        for r in rs:
            if r["cid"] in scores:
                continue
            best = -2.0
            for s in range(0, len(lines), a.batch):
                chunk = lines[s:s + a.batch]
                enc = tok(chunk, [r["text"]] * len(chunk), truncation=True, max_length=128, padding=True, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    p = torch.softmax(model(**enc).logits.float(), -1).cpu().numpy()
                best = max(best, float((p[:, ent_i] - p[:, con_i]).max()))
            scores[r["cid"]] = (best + 1) / 2
        Path(a.out_dir).mkdir(parents=True, exist_ok=True); json.dump(scores, open(part, "w"))
        print(f"  summac {k + 1}/{len(by_m)} {mid} ({len(rs)} claims, {len(lines)} lines)", flush=True)
else:
    from alignscore import AlignScore
    scorer = AlignScore(model="roberta-large", batch_size=16, device="cuda", ckpt_path=a.ckpt, evaluation_mode="nli_sp")
    for k, (mid, rs) in enumerate(by_m.items()):
        doc = "\n".join(trans[mid])
        sc = scorer.score(contexts=[doc] * len(rs), claims=[r["text"] for r in rs])
        for r, v in zip(rs, sc):
            scores[r["cid"]] = float(v)
        print(f"  alignscore {k + 1}/{len(by_m)} {mid}", flush=True)
Path(a.out_dir).mkdir(parents=True, exist_ok=True)
json.dump(scores, open(Path(a.out_dir) / f"baseline_{a.which}.json", "w"))
print(f"{len(scores)} claims scored")
