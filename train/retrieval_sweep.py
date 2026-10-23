"""Downstream effect of retrieval choices (GPU, ~15-30 min): AUROC of the entailment feature when the NLI model reads
evidence from each retriever / depth k / window size, plus an ORACLE-evidence run (gold lines), plus a failure split:
for SUPPORTED claims that the NLI scores as unsupported (p_ent < 0.5), is the gold evidence inside the retrieved windows
(NLI/claim problem) or not (retrieval miss)?   Labels: author-reviewed labels (all dev+test claims).

  python -m train.retrieval_sweep --ann data/meetverify/annotations_A1.jsonl
"""
import argparse, json
from collections import defaultdict
import numpy as np
from cprove import metrics as M
from cprove.utils import load_config, seed_everything
from cprove.data import load_corpus
from cprove.schema import Claim, Frame
from cprove.nli import make_nli, score_claims
from cprove.retrieval_eval import MultiRetriever, MODES

ap = argparse.ArgumentParser()
ap.add_argument("--ann", default="data/meetverify/annotations_A1.jsonl")
ap.add_argument("--items", nargs="+", default=["data/meetverify/items_pilot.jsonl", "data/meetverify/items_test.jsonl"])
ap.add_argument("--out", default="outputs/retrieval_sweep.json"); ap.add_argument("--config", default=None)
ap.add_argument("--only_oracle", action="store_true")
a = ap.parse_args()
cfg = load_config(a.config); seed_everything(cfg["seed"])
meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
items, ann = {}, {}
for f in a.items:
    for l in open(f, encoding="utf-8"):
        r = json.loads(l); items[r["cid"]] = r
for l in open(a.ann, encoding="utf-8"):
    r = json.loads(l)
    if r["label"] in ("SUPPORTED", "UNSUPPORTED") and r["cid"] in items:
        ann[r["cid"]] = r
by_m = defaultdict(list)
for cid in ann:
    by_m[items[cid]["meeting_id"]].append(cid)
rc = cfg["retrieval"]
ret = MultiRetriever(rc["encoder"], rc["dense_weight"], rc["sparse_weight"], rc["top_k"], rc["context_window"], rc["slot_boost"])
nli = make_nli(cfg["nli"])
mk = lambda c: Claim(cid=c, text=items[c]["text"], frame=Frame(**{k: v for k, v in (items[c].get("frame") or {}).items() if k in Frame.__dataclass_fields__}))


def run(mode, k, win):
    ent, cov = {}, {}
    for mid, cids in by_m.items():
        m = meetings[mid]; pos = {u.uid: i for i, u in enumerate(m.utterances)}; cl = [mk(c) for c in cids]
        if mode == "oracle":
            base = ret.retrieve_mode(m, cl, "hybrid", 5, win)     # claims WITHOUT annotated evidence keep normal retrieval
            ev = {}
            for c in cl:
                gi = [pos[u] for u in ann[c.cid].get("evidence_uids", []) if u in pos]
                ev[c.cid] = ret.retrieve_oracle(m, c, gi, win) if gi else base[c.cid]
        else:
            ev = ret.retrieve_mode(m, cl, mode, k, win)
        sc = score_claims(nli, m, cl, ev, win)
        for c in cl:
            ent[c.cid] = sc[c.cid]["ent"]
            g = {pos[u] for u in ann[c.cid].get("evidence_uids", []) if u in pos}
            cov[c.cid] = (len(g & {pos[u] for u in ev[c.cid].uids if u in pos}) / len(g)) if g else float("nan")
    return ent, cov


cids = sorted(ann); y = np.array([int(ann[c]["label"] == "SUPPORTED") for c in cids]); g = np.array([items[c]["meeting_id"] for c in cids])
grid = [("oracle", 5, 2)] if a.only_oracle else [(mode, k, 2) for mode in MODES for k in (1, 3, 5, 10)] + [("hybrid", 5, w) for w in (0, 1, 3)] + [("oracle", 5, 2)]
res = {}
print(f"{len(cids)} claims, {len(by_m)} meetings")
print(f"{'mode':8} k  win  AUROC [95% CI]       gold-covered  | false alarms: retrieval-miss / NLI-error")
for mode, k, w in grid:
    ent, cov = run(mode, k, w)
    p = np.array([ent[c] for c in cids]); lo, hi = M.cluster_bootstrap(M.auroc, y, p, g, 300)
    fa = [c for c, yy in zip(cids, y) if yy == 1 and ent[c] < 0.5]
    miss = sum(1 for c in fa if cov[c] < 1.0 and not np.isnan(cov[c])); err = sum(1 for c in fa if cov[c] >= 1.0)
    cv = np.nanmean([cov[c] for c in cids])
    res[f"{mode}/k{k}/w{w}"] = {"auroc": M.auroc(y, p), "ci": [lo, hi], "gold_coverage": float(cv), "false_alarms": len(fa), "retrieval_miss": miss, "nli_error": err}
    print(f"{mode:8} {k:<2} {w:<3}  {M.auroc(y, p):.3f} [{lo:.3f},{hi:.3f}]   {cv:.3f}        | {len(fa)} total: {miss} miss / {err} NLI")
import pathlib; pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
