"""Retrieval-miss vs model error, for MISSED unsupported claims and for FALSE ALARMS of the selected fusion at its inner-CV threshold.
  python -m train.miss_split
Uses out-of-fold fusion scores + the per-fold inner threshold (outputs/nested_cv.json), the shipped retriever (hybrid, k=5, window 2) re-run to see
which transcript lines the verifier actually read, and the author-confirmed evidence lines (only available for a subset of claims).
Categories (automatic, NO new human labelling):
  A  annotated evidence, ALL of it retrieved        -> the evidence was read; the error is in NLI/fusion
  B  annotated evidence, PART retrieved             -> partial retrieval miss
  C  annotated evidence, NONE retrieved             -> retrieval miss: relevant lines exist in the transcript but were not read
  D  no annotated evidence, MiniCheck (full transcript) >= 0.5 for a false alarm / < 0.5 for a miss -> PROXY: the full transcript points the other way
     than the retrieval-based decision, so relevant evidence is likely outside the retrieved windows
  E  no annotated evidence, MiniCheck agrees with the (wrong) decision -> no sign of unretrieved evidence (e.g. fabricated content)
D/E are proxies from a model, not human labels; A-C use the author's evidence lines (author-reviewed, not independently annotated)."""
import argparse, json
from collections import Counter, defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="cache/frozen/features.jsonl"); ap.add_argument("--preds", default="outputs/nested_cv.preds.npy")
ap.add_argument("--cv", default="outputs/nested_cv.json"); ap.add_argument("--config", default="configs/frozen.yaml")
a = ap.parse_args()
from cprove.utils import load_config
from cprove.data import load_corpus
from cprove.schema import Claim, Frame
from cprove.retrieval_eval import MultiRetriever
from train.nested_cv import load_rows

P = np.load(a.preds, allow_pickle=True).item(); cids = P.pop("cids"); s = np.array(P["fusion (selected in inner CV)"], float)
rows = {r["cid"]: r for r in load_rows(a.features)}
ch = {c["meeting"]: c["threshold"] for c in json.load(open(a.cv))["choices"]}
cfg = load_config(a.config); meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}; rc = cfg["retrieval"]
ret = MultiRetriever(rc["encoder"], rc["dense_weight"], rc["sparse_weight"], rc["top_k"], rc["context_window"], rc["slot_boost"])
y = np.array([int(rows[c]["author_label"] == "SUPPORTED") for c in cids]); thr = np.array([ch[rows[c]["meeting_id"]] for c in cids])
pred = (s >= thr).astype(int)
miss = [c for c, yy, pp in zip(cids, y, pred) if yy == 0 and pp == 1]; fa = [c for c, yy, pp in zip(cids, y, pred) if yy == 1 and pp == 0]
print(f"missed unsupported {len(miss)}, false alarms {len(fa)} (expected 66 and 57 at the inner threshold)")
cov = {}
by = defaultdict(list)
for c in miss + fa: by[rows[c]["meeting_id"]].append(c)
for mid, cs in by.items():
    m = meetings[mid]; pos = {u.uid: i for i, u in enumerate(m.utterances)}
    cl = [Claim(cid=c, text=rows[c]["text"], frame=Frame(**{k: v for k, v in (rows[c].get("frame") or {}).items() if k in Frame.__dataclass_fields__})) for c in cs]
    ev = ret.retrieve_mode(m, cl, "hybrid", rc["top_k"], rc["context_window"])
    for c in cs:
        g = {pos[u] for u in rows[c].get("author_evidence") or [] if u in pos}
        got = {pos[u] for u in ev[c].uids if u in pos}
        cov[c] = (len(g & got) / len(g)) if g else None


def cat(c, is_fa):
    v = cov[c]
    if v is not None: return "A all annotated evidence retrieved" if v >= 1 else "B annotated evidence partly retrieved" if v > 0 else "C annotated evidence not retrieved"
    mc = rows[c]["mc"]
    return ("D no annotation; MiniCheck on full transcript disagrees (evidence likely unretrieved)" if ((mc >= 0.5) if is_fa else (mc < 0.5))
            else "E no annotation; MiniCheck agrees (no sign of unretrieved evidence)")


for name, ids, is_fa in (("MISSED unsupported claims", miss, False), ("FALSE ALARMS (supported claims scored low)", fa, True)):
    k = Counter(cat(c, is_fa) for c in ids); print(f"\n{name}: n={len(ids)}")
    for key in sorted(k): print(f"  {key:84} {k[key]:3}  ({100 * k[key] / len(ids):.0f}%)")
    if not is_fa:
        et = defaultdict(Counter)
        for c in ids: et[rows[c].get("author_error")][cat(c, False)[0]] += 1
        print("  by author error type (A-E):", {e: dict(v) for e, v in et.items()})
    ann = [c for c in ids if cov[c] is not None]
    print(f"  with annotated evidence: {len(ann)} of {len(ids)}; retrieval miss (B or C) among them: {sum(cov[c] < 1 for c in ann)}")
