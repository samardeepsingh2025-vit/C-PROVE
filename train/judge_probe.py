"""Does the real LLM judge beat the cheap fact score? Judges every claim of N dev summaries.
python -m train.judge_probe --dev data\\meetverify\\synth_dev.jsonl --n 25"""
import argparse, json, pickle, random
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from cprove.utils import load_config, seed_everything
from cprove.data import load_corpus
from cprove.pipeline import CProve, Ablation
from cprove.judge import LLMJudge

ap = argparse.ArgumentParser()
ap.add_argument("--dev", required=True); ap.add_argument("--n", type=int, default=25)
ap.add_argument("--config", default=None)
a = ap.parse_args()
cfg = load_config(a.config); seed_everything(cfg["seed"])
meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
cp = CProve(cfg, Ablation(hand_weights=True))
print("llm available:", cp.llm.available)
items = pickle.load(open(Path(cfg["paths"]["cache"]) / ("items_" + Path(a.dev).stem + ".pkl"), "rb"))
random.Random(0).shuffle(items)
bad = [i for i in items if 0 in i.gold.values()][: a.n // 2]
good = [i for i in items if 0 not in i.gold.values()][: a.n - len(bad)]
y, fact, jp, typ = [], [], [], []
for k, it in enumerate(bad + good):
    m = meetings[it.meeting_id]
    j = LLMJudge(cp.llm, it.claims, cp.retriever.retrieve(m, it.claims), cp._neighbour_fn(it.claims, it.edges))
    for c in it.claims:
        if c.cid not in it.gold:
            continue
        p = j(c.cid)
        if p is None:
            continue
        y.append(it.gold[c.cid]); jp.append(p); fact.append(it.obs[c.cid].raw["fact"]); typ.append(getattr(c, "gold_error", "none"))
    print(f"summary {k + 1}/{a.n} claims so far {len(y)}", flush=True)
y, fact, jp = map(np.array, (y, fact, jp))
print("claims", len(y), "errors", int((y == 0).sum()))
if len(set(y)) == 2:
    print("AUROC  fact:", round(roc_auc_score(1 - y, 1 - fact), 3), " judge:", round(roc_auc_score(1 - y, 1 - jp), 3),
          " avg:", round(roc_auc_score(1 - y, 1 - (fact + jp) / 2), 3))
    for t in sorted(set(typ) - {"none"}):
        mk = np.array([(tt == t) or yy == 1 for tt, yy in zip(typ, y)])
        if len(set(y[mk])) == 2:
            print(f"  {t:18} n_err={sum(1 for tt, yy in zip(typ, y) if tt == t and yy == 0):3d} fact {roc_auc_score(1 - y[mk], 1 - fact[mk]):.2f} judge {roc_auc_score(1 - y[mk], 1 - jp[mk]):.2f}")