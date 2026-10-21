"""External validation on TofuEval-MeetingBank (human sentence-level factual-consistency labels). Fusion is fitted on AMI only.
git clone https://github.com/amazon-science/tofueval  (into D:\\VIT\\2nd year\\tofueval)  ;  pip install datasets pandas
python -m train.tofueval --tofu ..\\tofueval --ann data\\meetverify\\annotations_A1.jsonl --ami_caches cache\\items_qwen_dev20.pkl cache\\items_qwen_test.pkl --max_docs 40"""
import argparse, json, pickle, re, gc
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from cprove.utils import load_config, seed_everything, SLOT
from cprove.schema import Meeting, Utterance, Claim
from cprove.pipeline import CProve, Ablation
from cprove.dimensions import Calibrators
from baselines.run_baselines import transcript_text

ap = argparse.ArgumentParser()
ap.add_argument("--tofu", required=True); ap.add_argument("--ann", required=True)
ap.add_argument("--ami_caches", nargs="+", required=True); ap.add_argument("--max_docs", type=int, default=40)
ap.add_argument("--config", default=None)
a = ap.parse_args()
cfg = load_config(a.config); seed_everything(cfg["seed"])

def feats(raw):
    row = [float(raw.get("ent") or 0), float(raw.get("con") or 0)]
    for f in ("attr", "sem"):
        v = raw.get(f); row += [0.0 if v is None else float(v) - 0.5, 0.0 if v is None else 1.0]
    return row

# ---- fusion fitted on AMI only (all human-verified claims)
ann = {}
for l in open(a.ann, encoding="utf-8"):
    r = json.loads(l); ann[r["cid"]] = r
Xa, ya = [], []
for p in a.ami_caches:
    for it in pickle.load(open(p, "rb")):
        for c in it.claims:
            r = ann.get(c.cid)
            if r and r["label"] != "UNVERIFIABLE":
                Xa.append(feats(it.obs[c.cid].raw)); ya.append(int(r["label"] == "SUPPORTED"))
Xa, ya = np.array(Xa), np.array(ya); print(f"AMI fit n={len(ya)}")
# attr is undefined without speakers, so the TofuEval fusion uses ent, con, sem
cols = [0, 1, 4, 5]
clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(Xa[:, cols], ya)

# ---- TofuEval data
tofu = Path(a.tofu); split = json.load(open(tofu / "document_ids_dev_test_split.json"))
test_ids = set(map(str, split["test"]["meetingbank"]))
fc = pd.concat([pd.read_csv(f) for f in sorted((tofu / "factual_consistency").glob("meetingbank*test*.csv"))] or
               [pd.read_csv(f) for f in sorted((tofu / "factual_consistency").glob("meetingbank*.csv"))])
fc = fc[fc["doc_id"].astype(str).isin(test_ids)]
from datasets import load_dataset
ds = load_dataset("lytang/MeetingBank-transcript", split="test")
docs = {str(r["meeting_id"]): r["source"] for r in ds}
fc = fc[fc["doc_id"].astype(str).isin(docs)]
keep = sorted(fc["doc_id"].astype(str).unique())[: a.max_docs]
fc = fc[fc["doc_id"].astype(str).isin(keep)].reset_index(drop=True)
print(f"{len(keep)} docs, {len(fc)} summary sentences, {(fc.sent_label=='no').mean():.2f} inconsistent")

def meeting(did):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", docs[did]) if s.strip()]
    return Meeting(did, [Utterance(f"{did}_u{i:04d}", "Speaker", s, idx=i) for i, s in enumerate(sents)], {})

cp = CProve(cfg, Ablation(hand_weights=True)); cp.dims.cal = Calibrators("none")
from train.fit_cprove import recalibrate
X, y = [], []
for did in keep:
    m = meeting(did); sub = fc[fc["doc_id"].astype(str) == did]
    claims = [Claim(cid=f"{did}__{i}", text=str(r.summ_sent), summary_id=did, sent_idx=int(r.sent_idx)) for i, r in enumerate(sub.itertuples())]
    _, _, obs, _ = cp.observe(m, claims)
    for c, r in zip(claims, sub.itertuples()):
        X.append(feats(obs[c.cid].raw)); y.append(int(r.sent_label == "yes"))
    print("  done", did)
X, y = np.array(X), np.array(y)
del cp; gc.collect(); SLOT.release(); torch.cuda.empty_cache()

res = {"entailment only": X[:, 0], "C-PROVE fusion (AMI-trained)": clf.predict_proba(X[:, cols])[:, 1]}
from minicheck.minicheck import MiniCheck
mc = MiniCheck(model_name="flan-t5-large", cache_dir="models/hf"); p = np.zeros(len(y)); k = 0
for did in keep:
    n = int((fc["doc_id"].astype(str) == did).sum()); sub = fc[fc["doc_id"].astype(str) == did]
    _, pr, _, _ = mc.score(docs=[docs[did]] * n, claims=[str(s) for s in sub["summ_sent"]]); p[k:k + n] = pr; k += n
    torch.cuda.empty_cache()
res["MiniCheck"] = p
rng = np.random.default_rng(0)
print(f"\nn={len(y)}  method  AUROC [95% CI]  AUPRC(inconsistent)")
for name, s in res.items():
    bs = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1: bs.append(roc_auc_score(1 - y[i], 1 - s[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{name:30} {roc_auc_score(1 - y, 1 - s):.3f} [{lo:.3f},{hi:.3f}]  {average_precision_score(1 - y, 1 - s):.3f}")
