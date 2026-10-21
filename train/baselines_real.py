"""Claim-level baselines on the human-verified test claims (same 379 claims as final_eval).
pip install summac ; pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"
python -m train.baselines_real --ann data\\meetverify\\annotations_A1.jsonl --test data\\meetverify\\qwen_test.jsonl --systems minicheck summac_zs"""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from cprove.utils import load_config, seed_everything
from cprove.data import load_corpus
from cprove.pipeline import CProve, Ablation
from cprove.dimensions import Calibrators
from experiments.common import load_records, build_items
from baselines.run_baselines import transcript_text

ap = argparse.ArgumentParser()
ap.add_argument("--ann", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--systems", nargs="+", default=["minicheck", "summac_zs"]); ap.add_argument("--config", default=None)
a = ap.parse_args()
cfg = load_config(a.config); seed_everything(cfg["seed"])
ann = {}
for l in open(a.ann, encoding="utf-8"):
    r = json.loads(l); ann[r["cid"]] = r
meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
recs = [r for r in load_records(a.test) if r.get("system") != "reference"]
cp = CProve(cfg, Ablation(hand_weights=True)); cp.dims.cal = Calibrators("none")
items = build_items(cp, meetings, recs, str(Path(cfg["paths"]["cache"]) / ("items_" + Path(a.test).stem + ".pkl")))
rows = [(it.meeting_id, c.cid, c.text, int(ann[c.cid]["label"] == "SUPPORTED")) for it in items for c in it.claims
        if c.cid in ann and ann[c.cid]["label"] != "UNVERIFIABLE"]
y = np.array([r[3] for r in rows]); print(f"n={len(y)}")
import gc, torch
from cprove.utils import SLOT
del cp, items; gc.collect(); SLOT.release(); torch.cuda.empty_cache()

def run(name):
    out = np.zeros(len(rows))
    if name == "minicheck":
        from minicheck.minicheck import MiniCheck
        mc = MiniCheck(model_name="flan-t5-large", cache_dir="models/hf")
        for mid in sorted({r[0] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r[0] == mid]; doc = transcript_text(meetings[mid])
            _, p, _, _ = mc.score(docs=[doc] * len(idx), claims=[rows[i][2] for i in idx]); out[idx] = p
            torch.cuda.empty_cache()
        del mc
    elif name == "summac_zs":
        from summac.model_summac import SummaCZS
        from transformers import PreTrainedTokenizerBase as _T
        if not hasattr(_T, "batch_encode_plus"):
            _T.batch_encode_plus = lambda self, *a, **k: self(*a, **k)
        from cprove.utils import device
        m = SummaCZS(granularity="sentence", model_name="vitc", device=device())
        for mid in sorted({r[0] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r[0] == mid]; doc = transcript_text(meetings[mid])
            for i in idx: out[i] = (m.score([doc], [rows[i][2]])["scores"][0] + 1) / 2
            torch.cuda.empty_cache()
        del m
    elif name == "alignscore":
        from alignscore import AlignScore
        from cprove.utils import device
        s = AlignScore(model="roberta-large", batch_size=16, device=device(), ckpt_path="models/AlignScore-large.ckpt", evaluation_mode="nli_sp")
        for mid in sorted({r[0] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r[0] == mid]; doc = transcript_text(meetings[mid])
            out[idx] = s.score(contexts=[doc] * len(idx), claims=[rows[i][2] for i in idx])
    return out

for name in a.systems:
    try:
        p = run(name)
        rng = np.random.default_rng(0); bs = []
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            if len(set(y[i])) > 1: bs.append(roc_auc_score(1 - y[i], 1 - p[i]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{name:12} AUROC {roc_auc_score(1 - y, 1 - p):.3f} [{lo:.3f},{hi:.3f}]  AUPRC(unsup) {average_precision_score(1 - y, 1 - p):.3f}")
        gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        gc.collect(); torch.cuda.empty_cache()
        print(f"{name}: FAILED {type(e).__name__}: {e}")
