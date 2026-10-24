"""Frozen external evaluation on TofuEval-MeetingBank (human sentence-level factual-consistency labels).

  git clone https://github.com/amazon-science/tofueval ..\\tofueval      (once)
  python -m train.tofueval_frozen --tofu ..\\tofueval --max_docs 0        # 0 = every test document; else a seeded random sample
Protocol (fixed before running):
  * config = configs/frozen.yaml (the tagged v2-frozen configuration), no change, no tuning on TofuEval.
  * UNIT: every human-labelled summary SENTENCE is one claim for EVERY method (no decomposition, no frames), so all methods see identical inputs.
  * Fusion = the AMI model restricted to the features that exist without speakers/frames: entailment, contradiction, MiniCheck
    ("ent+con+mc"); C chosen by meeting-grouped inner CV, Platt calibration, all fitted on the 523 AMI claims only (train.nested_cv.fit_select).
    Attribution/status/temporal/context features are undefined here, so the headline AMI model (which uses them) is NOT what is tested.
  * Baselines: entailment only (same retrieval/NLI as the fusion's input), MiniCheck (Flan-T5-Large, sentence vs full transcript).
  * Documents: all test-split MeetingBank documents with labels, or --max_docs sampled with a seeded RNG (--seed 42); ids are printed and saved.
  * Statistics: AUROC / AUPRC (inconsistent = positive) with document-clustered bootstrap CIs (1000), paired differences to MiniCheck, Holm.
Outputs: outputs/tofueval_frozen.json, outputs/tofueval_frozen_table.tex"""
import argparse, gc, json, re
from pathlib import Path
import numpy as np, pandas as pd, torch
from cprove import metrics as M
from cprove.utils import load_config, seed_everything, SLOT
from cprove.schema import Meeting, Utterance, Claim
from cprove.pipeline import CProve, Ablation
from cprove.dimensions import Calibrators
from train.nested_cv import load_rows, fit_select

ap = argparse.ArgumentParser()
ap.add_argument("--tofu", required=True); ap.add_argument("--config", default="configs/frozen.yaml")
ap.add_argument("--features", default="cache/frozen/features.jsonl")
ap.add_argument("--max_docs", type=int, default=0); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--boots", type=int, default=1000)
ap.add_argument("--out", default="outputs/tofueval_frozen.json")
a = ap.parse_args()
cfg = load_config(a.config); seed_everything(cfg["seed"])

# ---- fusion fitted on AMI only
rows = load_rows(a.features)
pred, choice = fit_select(rows, {"ent+con+mc": ["ent", "con", "mc"]})
print(f"AMI fit: {len(rows)} claims, choice {choice}")

# ---- data
tofu = Path(a.tofu); split = json.load(open(tofu / "document_ids_dev_test_split.json")); test_ids = set(map(str, split["test"]["meetingbank"]))
files = sorted((tofu / "factual_consistency").glob("meetingbank*test*.csv")) or sorted((tofu / "factual_consistency").glob("meetingbank*.csv"))
fc = pd.concat([pd.read_csv(f) for f in files]); fc = fc[fc["doc_id"].astype(str).isin(test_ids)]
from datasets import load_dataset
docs = {str(r["meeting_id"]): r["source"] for r in load_dataset("lytang/MeetingBank-transcript", split="test")}
fc = fc[fc["doc_id"].astype(str).isin(docs)]
ids = sorted(fc["doc_id"].astype(str).unique())
if a.max_docs and a.max_docs < len(ids):
    ids = sorted(np.random.default_rng(a.seed).choice(ids, a.max_docs, replace=False).tolist())
fc = fc[fc["doc_id"].astype(str).isin(ids)].reset_index(drop=True)
print(f"{len(ids)} documents, {len(fc)} summary sentences, {(fc.sent_label == 'no').mean():.2f} inconsistent\ndoc ids: {ids}")


def meeting(did):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", docs[did]) if s.strip()]
    return Meeting(did, [Utterance(f"{did}_u{i:04d}", "Speaker", s, idx=i) for i, s in enumerate(sents)], {})

# ---- stage 1: retrieval + NLI (frozen pipeline), one claim = one summary sentence
cp = CProve(cfg, Ablation(hand_weights=True)); cp.dims.cal = Calibrators("none")
ent, con, y, g = [], [], [], []
for did in ids:
    m = meeting(did); sub = fc[fc["doc_id"].astype(str) == did]
    claims = [Claim(cid=f"{did}__{i}", text=str(r.summ_sent), summary_id=did, sent_idx=int(r.sent_idx)) for i, r in enumerate(sub.itertuples())]
    _, _, obs, _ = cp.observe(m, claims)
    for c, r in zip(claims, sub.itertuples()):
        ent.append(float(obs[c.cid].raw.get("ent") or 0)); con.append(float(obs[c.cid].raw.get("con") or 0)); y.append(int(r.sent_label == "yes")); g.append(did)
    print("  nli done", did, flush=True)
del cp; gc.collect(); SLOT.release(); torch.cuda.empty_cache()
# ---- stage 2: MiniCheck, sentence vs full transcript
from minicheck.minicheck import MiniCheck
mc = MiniCheck(model_name="flan-t5-large", cache_dir="models/hf"); mcs = []
for did in ids:
    sub = fc[fc["doc_id"].astype(str) == did]
    _, pr, _, _ = mc.score(docs=[docs[did]] * len(sub), claims=[str(s) for s in sub["summ_sent"]]); mcs += list(map(float, pr)); torch.cuda.empty_cache()
y, g = np.array(y), np.array(g); ent, con, mcs = np.array(ent), np.array(con), np.array(mcs)
fus = pred([{"ent": e, "con": c, "mc": m_} for e, c, m_ in zip(ent, con, mcs)])
S = {"Entailment only": ent, "MiniCheck": mcs, "Fusion (AMI-trained: ent+con+MiniCheck)": np.asarray(fus, float)}

# ---- statistics (supported = 1; AUPRC for the inconsistent class)
res = {"n_sentences": int(len(y)), "n_docs": len(ids), "doc_ids": ids, "inconsistent_share": float(1 - y.mean()), "ami_choice": choice, "methods": {}}
print(f"\nn={len(y)} sentences, {len(ids)} documents\n{'method':44} AUROC [95% doc-clustered CI]   AUPRC(inconsistent) [CI]")
for k, p in S.items():
    lo, hi = M.cluster_bootstrap(M.auroc, y, p, g, a.boots); f_auprc = lambda yy, pp: M.auprc(yy, pp)
    ulo, uhi = M.cluster_bootstrap(M.auprc, y, p, g, a.boots)
    res["methods"][k] = {"auroc": M.auroc(y, p), "auroc_ci": [lo, hi], "auprc_unsup": M.auprc(y, p), "auprc_ci": [ulo, uhi]}
    print(f"{k:44} {M.auroc(y, p):.3f} [{lo:.3f},{hi:.3f}]   {M.auprc(y, p):.3f} [{ulo:.3f},{uhi:.3f}]")
ps = {}
for k in ("Fusion (AMI-trained: ent+con+MiniCheck)", "Entailment only"):
    d, lo, hi, p = M.paired_bootstrap(M.auroc, y, S[k], S["MiniCheck"], g); ps[k] = p; res["methods"][k]["vs_minicheck"] = {"d": d, "ci": [lo, hi], "p": p}
    print(f"paired AUROC {k} - MiniCheck: {d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}")
res["holm"] = {k: round(v, 3) for k, v in M.holm(ps).items()}; print("Holm-adjusted p:", res["holm"])
Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=1, default=float)
np.save(Path(a.out).with_suffix(".scores.npy"), {"y": y.tolist(), "doc": g.tolist(), **{k: v.tolist() for k, v in S.items()}}, allow_pickle=True)
