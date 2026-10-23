"""STEP 1 of the NLI-aggregation ablation (GPU, ~5 min). Stores the per-window NLI distributions [ent, neu, con] and the retrieval
relevance of every annotated claim under the SHIPPED retrieval setting (hybrid, k=5, window +-2), so aggregation variants can be
compared on CPU without re-running the NLI model.

  python -m train.nli_windows
Writes cache/frozen/nli_windows.json {cid: {"dist": [[e,n,c],...], "rel": [...]}} and prints a sanity check against the shipped
'ent'/'con' features (should correlate ~1.0)."""
import argparse, json
from collections import defaultdict
import numpy as np
from cprove.utils import load_config, seed_everything
from cprove.data import load_corpus
from cprove.schema import Claim, Frame
from cprove.nli import make_nli, windows_for, _mix
from cprove.retrieval_eval import MultiRetriever

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="cache/frozen/features.jsonl"); ap.add_argument("--config", default=None)
ap.add_argument("--out", default="cache/frozen/nli_windows.json"); ap.add_argument("--k", type=int, default=5); ap.add_argument("--win", type=int, default=2)
a = ap.parse_args()
cfg = load_config(a.config); seed_everything(cfg["seed"])
meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
F = {}
for l in open(a.features, encoding="utf-8"):
    r = json.loads(l)
    if r.get("author_label") in ("SUPPORTED", "UNSUPPORTED", "UNVERIFIABLE"):
        F.setdefault(r["cid"], r)
by_m = defaultdict(list)
for c, r in F.items():
    by_m[r["meeting_id"]].append(c)
rc = cfg["retrieval"]
ret = MultiRetriever(rc["encoder"], rc["dense_weight"], rc["sparse_weight"], rc["top_k"], rc["context_window"], rc["slot_boost"])
nli = make_nli(cfg["nli"])
mk = lambda c: Claim(cid=c, text=F[c]["text"], frame=Frame(**{k: v for k, v in (F[c].get("frame") or {}).items() if k in Frame.__dataclass_fields__}),
                     claim_type=F[c].get("claim_type", "fact"))
out, chk = {}, []
for n, (mid, cids) in enumerate(by_m.items()):
    m = meetings[mid]; cl = [mk(c) for c in cids]
    ev = ret.retrieve_mode(m, cl, "hybrid", a.k, a.win)
    prem, hyp, own, rel = [], [], [], []
    for c in cl:
        for txt, r in windows_for(m, ev[c.cid], a.win):
            prem.append(txt); hyp.append(c.text); own.append(c.cid); rel.append(r)
    D = nli.score_pairs(prem, hyp) if prem else np.zeros((0, 3)); own = np.array(own); rel = np.array(rel)
    for c in cl:
        i = np.where(own == c.cid)[0]
        out[c.cid] = {"dist": D[i].tolist(), "rel": rel[i].tolist()}
        if len(i):
            mix = _mix(D[i], rel[i]); chk.append((mix[0], F[c.cid]["ent"], mix[2], F[c.cid]["con"]))
    print(f"  {n + 1}/{len(by_m)} {mid} ({len(cl)} claims)", flush=True)
json.dump(out, open(a.out, "w"))
if chk:
    A = np.array(chk)
    print(f"{len(out)} claims; sanity vs shipped features: corr(ent) {np.corrcoef(A[:, 0], A[:, 1])[0, 1]:.3f}, corr(con) {np.corrcoef(A[:, 2], A[:, 3])[0, 1]:.3f}, mean |d ent| {np.abs(A[:, 0] - A[:, 1]).mean():.3f}")
