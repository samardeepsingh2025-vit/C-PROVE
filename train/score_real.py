"""Score real LLM summaries (no gold) and print the claims C-PROVE trusts least, with transcript evidence,
so you can check by eye whether they are really wrong.
python -m train.score_real --summaries data\\meetverify\\qwen_dev20.jsonl --top 25"""
import argparse
from pathlib import Path
from cprove.utils import load_config, seed_everything
from cprove.data import load_corpus
from cprove.pipeline import CProve, Ablation
from cprove.dimensions import Calibrators
from cprove.hlmrf import HLMRF, Weights
from experiments.common import load_records, build_items, programs
from train.fit_cprove import recalibrate

ap = argparse.ArgumentParser()
ap.add_argument("--summaries", required=True); ap.add_argument("--top", type=int, default=25)
ap.add_argument("--config", default=None); ap.add_argument("--system", default=None)
a = ap.parse_args()
cfg = load_config(a.config); seed_everything(cfg["seed"])
meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
recs = [r for r in load_records(a.summaries) if r.get("system") != "reference" and (not a.system or r.get("system") == a.system)]
cp = CProve(cfg, Ablation(hand_weights=True)); cp.dims.cal = Calibrators("none")
items = build_items(cp, meetings, recs, str(Path(cfg["paths"]["cache"]) / ("items_" + Path(a.summaries).stem + ".pkl")))
recalibrate(items, Calibrators.load(cfg["paths"]["models"] + "/calibrators.pkl"))
h = cfg["hlmrf"]
model = HLMRF(Weights.load(h.get("learned_weights"), h["init_weights"]), h["iters"], h["tol"])
rows = []
for it, pr in zip(items, programs(items)):
    y, _ = model.infer(pr)
    m = meetings[it.meeting_id]
    ev = cp.retriever.retrieve(m, it.claims)
    for c in it.claims:
        rows.append((float(y[pr.y_index(c.cid)]), it.rec["summary_id"], c, it.obs[c.cid], ev[c.cid].text))
rows.sort(key=lambda r: r[0])
print(f"{len(rows)} claims from {len(items)} summaries; mean trust {sum(r[0] for r in rows)/len(rows):.2f}")
for p, sid, c, o, evt in rows[: a.top]:
    print(f"\n[{p:.2f}] {sid} | {c.text}\n   ent={o.ent:.2f} con={o.con:.2f}\n   evidence: {evt[:500]}")