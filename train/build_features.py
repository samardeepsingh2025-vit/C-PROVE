"""Build the per-claim FEATURE TABLE with the frozen config (the only heavy GPU step of the revision).
Everything downstream (nested CV, ablations, calibration, bootstrap, abstention, MTI, robustness) reads this table
and runs on CPU in seconds.

  python -m train.build_features --config configs/frozen.yaml ^
      --sets dev=data/meetverify/qwen_dev20.jsonl test=data/meetverify/qwen_test.jsonl ^
      --ann data/meetverify/annotations_A1.jsonl --out cache/frozen/features.jsonl
Per claim: pure ent/con (numeric veto NOT folded in), attr/sem/temp/ctx (+missing flags), MiniCheck score against the full
transcript, numeric report, hand-weight HL-MRF posterior with and without the graph, retrieved evidence uids, author label.
"""
from __future__ import annotations
import argparse, gc, json, pickle
from pathlib import Path
import numpy as np
from cprove.utils import load_config, seed_everything
from cprove.data import load_corpus
from cprove.pipeline import CProve, Ablation
from cprove.dimensions import Calibrators, numeric_check
from cprove.hlmrf import HLMRF, Weights
from experiments.common import load_records, build_items, programs
from baselines.run_baselines import transcript_text

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/frozen.yaml")
ap.add_argument("--sets", nargs="+", required=True, help="name=records.jsonl")
ap.add_argument("--ann", nargs="*", default=[], help="annotation jsonl(s) with cid,label (author labels)")
ap.add_argument("--out", default="cache/frozen/features.jsonl")
ap.add_argument("--no_minicheck", action="store_true")
a = ap.parse_args()
cfg = load_config(a.config); seed_everything(cfg["seed"])
ann = {}
for f in a.ann:
    for l in open(f, encoding="utf-8"):
        r = json.loads(l); ann[r["cid"]] = r
meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
cp = CProve(cfg, Ablation(hand_weights=True)); cp.dims.cal = Calibrators("none")
h = cfg["hlmrf"]
model = HLMRF(Weights(h["init_weights"]), h["iters"], h["tol"])
cache_dir = Path(cfg["paths"]["cache"]); cache_dir.mkdir(parents=True, exist_ok=True)

rows = []
for spec in a.sets:
    name, path = spec.split("=", 1)
    recs = [r for r in load_records(path) if r.get("system") != "reference"]
    items = build_items(cp, meetings, recs, str(cache_dir / f"items_{name}.pkl"))
    y_g, y_ng = {}, {}
    for graph, store in ((True, y_g), (False, y_ng)):
        for it, pr in zip(items, programs(items, graph=graph)):
            y, _ = model.infer(pr)
            for c in it.claims:
                store[c.cid] = float(y[pr.y_index(c.cid)])
    for it in items:
        m = meetings[it.meeting_id]
        for c in it.claims:
            o = it.obs[c.cid]; raw = o.raw; nc = numeric_check(m, c)
            lab = ann.get(c.cid, {})
            ev = getattr(o, "evidence", None)
            row = {"set": name, "cid": c.cid, "meeting_id": it.meeting_id, "summary_id": c.summary_id, "text": c.text,
                   "claim_type": c.claim_type, "sent_idx": c.sent_idx, "frame": c.frame.__dict__,
                   "ent": float(raw.get("ent") or 0), "con": float(raw.get("con") or 0), "fact": float(raw.get("fact") or 0),
                   "num_unseen": float(bool(nc["unseen"])), "num_values": nc["claim"], "num_unseen_values": nc["unseen"],
                   "y_graph": y_g[c.cid], "y_nograph": y_ng[c.cid], "author_label": lab.get("label"),
                   "author_error": lab.get("error_type"), "author_evidence": lab.get("evidence_uids", [])}
            for d in ("attr", "temp", "ctx", "sem"):
                v = raw.get(d); row[d] = 0.0 if v is None else float(v) - 0.5; row[d + "_flag"] = 0.0 if v is None else 1.0
            row["grounding"] = [g if isinstance(g, (int, str)) else str(g) for g in (raw.get("grounding") or [])][:12]
            rows.append(row)
    print(f"[{name}] {len(items)} summaries, {sum(r['set'] == name for r in rows)} claims", flush=True)

# free the pipeline before MiniCheck (8 GB GPU)
import torch
del cp, model, items
gc.collect(); cp = None
try:
    from cprove.utils import SLOT; SLOT.release()
except Exception:
    pass
torch.cuda.empty_cache()

if not a.no_minicheck:
    from minicheck.minicheck import MiniCheck
    mc = MiniCheck(model_name="flan-t5-large", cache_dir="models/hf")
    mc_path = Path("cache/frozen/mc_cache.json"); mc_path.parent.mkdir(parents=True, exist_ok=True)
    mc_cache = json.load(open(mc_path)) if mc_path.exists() else {}
    BS = 2   # small batches: long meetings overflow 8 GB if all claims are scored at once
    for mid in sorted({r["meeting_id"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["meeting_id"] == mid]; doc = transcript_text(meetings[mid])
        todo = [i for i in idx if f"{mid}||{rows[i]['text']}" not in mc_cache]
        for b in range(0, len(todo), BS):
            ch = todo[b:b + BS]
            _, p, _, _ = mc.score(docs=[doc] * len(ch), claims=[rows[i]["text"] for i in ch])
            for i, v in zip(ch, p):
                mc_cache[f"{mid}||{rows[i]['text']}"] = float(v)
            torch.cuda.empty_cache()
        json.dump(mc_cache, open(mc_path, "w"))
        for i in idx:
            rows[i]["mc"] = mc_cache[f"{mid}||{rows[i]['text']}"]
        print(f"  minicheck {mid} ({len(todo)} new)", flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"{len(rows)} claims -> {a.out}")
