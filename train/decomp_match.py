"""E7 part 4: second decomposer (Llama-3.1-8B) vs the shipped one (Qwen2.5-7B) on the SAME 40 summaries.  CPU + the BGE encoder, ~2 min.
Prerequisite: python -m train.build_features --config configs/frozen_llama.yaml --sets <same sets> --out cache/frozen/features_llama.jsonl
  python -m train.decomp_match
Claims are matched one-to-one within each summary by cosine similarity of sentence embeddings (default > 0.9; 0.85 reported as sensitivity).
A Llama claim inherits the author label of the matched Qwen claim. Unmatched Llama claims are NOT labelled: they are written to
outputs/unmatched_llama.csv for a manual check and are excluded from the AUROC. The fusion (feature set ent+con+attr+sem+mc, C=1) is refit on the
other 39 meetings and applied to both versions of each matched claim, so the comparison is paired."""
import argparse, csv, json
from collections import defaultdict
import numpy as np
from cprove import metrics as M
import train.nested_cv as N

ap = argparse.ArgumentParser()
ap.add_argument("--qwen", default="cache/frozen/features.jsonl"); ap.add_argument("--llama", default="cache/frozen/features_llama.jsonl")
ap.add_argument("--thr", type=float, default=0.9); ap.add_argument("--config", default=None); ap.add_argument("--boots", type=int, default=1000)
a = ap.parse_args()
Q = [r for r in N.load_rows(a.qwen, labelled_only=False) if r.get("author_label") in ("SUPPORTED", "UNSUPPORTED")]
Lm = N.load_rows(a.llama, labelled_only=False)
allq = defaultdict(list); allL = defaultdict(list)
for r in N.load_rows(a.qwen, labelled_only=False):
    allq[r["summary_id"]].append(r)
for r in Lm:
    allL[r["summary_id"]].append(r)
summaries = sorted(set(allq) & set(allL))
from cprove.utils import load_config
from sentence_transformers import SentenceTransformer
cfg = load_config(a.config); enc = SentenceTransformer(cfg["retrieval"]["encoder"])
emb = lambda rs: enc.encode([r["text"] for r in rs], normalize_embeddings=True, batch_size=64) if rs else np.zeros((0, 1))


def match(thr):
    pairs, un_l = [], []
    for sid in summaries:
        q = [r for r in allq[sid] if r.get("author_label") in ("SUPPORTED", "UNSUPPORTED")]; l = allL[sid]
        if not q or not l:
            un_l += l; continue
        S = emb(q) @ emb(l).T; used_q, used_l = set(), set()
        for flat in np.argsort(-S, axis=None):
            i, j = divmod(int(flat), S.shape[1])
            if S[i, j] < thr:
                break
            if i in used_q or j in used_l:
                continue
            used_q.add(i); used_l.add(j); pairs.append((q[i], l[j], float(S[i, j])))
        un_l += [l[j] for j in range(len(l)) if j not in used_l]
    return pairs, un_l


nq = len([r for r in Q]); nl = len(Lm)
print(f"summaries in both: {len(summaries)}; Qwen claims (labelled) {nq}; Llama claims {nl}; mean claims/summary Qwen {np.mean([len(allq[s]) for s in summaries]):.1f}, Llama {np.mean([len(allL[s]) for s in summaries]):.1f}")
blocks = {"set": ["ent", "con", "attr", "sem", "mc"]}
for thr in (a.thr, 0.85):
    pairs, unl = match(thr)
    print(f"\n== cosine > {thr}: matched pairs {len(pairs)} = {100 * len(pairs) / nl:.0f}% of Llama claims, {100 * len(pairs) / nq:.0f}% of labelled Qwen claims; unmatched Llama claims {len(unl)}")
    if thr == a.thr:
        with open("outputs/unmatched_llama.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["summary_id", "cid", "text", "your_label(S/U/V)"])
            for r in unl:
                w.writerow([r["summary_id"], r["cid"], r["text"], ""])
    if len(pairs) < 20:
        continue
    yq = np.array([int(p[0]["author_label"] == "SUPPORTED") for p in pairs]); g = np.array([p[0]["meeting_id"] for p in pairs])
    pq, pl = np.zeros(len(pairs)), np.zeros(len(pairs))
    for m in np.unique(g):
        te = np.where(g == m)[0]; tr = [r for r in Q if r["meeting_id"] != m]
        pred, _ = N.fit_select(tr, blocks)
        pq[te] = pred([pairs[i][0] for i in te]); pl[te] = pred([pairs[i][1] for i in te])
    for nm, p in (("Qwen2.5-7B claims", pq), ("Llama-3.1-8B claims", pl)):
        print(f"  fusion on matched set, {nm}: AUROC {M.auroc(yq, p):.3f} [{M.cluster_bootstrap(M.auroc, yq, p, g, a.boots)[0]:.3f},{M.cluster_bootstrap(M.auroc, yq, p, g, a.boots)[1]:.3f}]")
    d, lo, hi, p = M.paired_bootstrap(M.auroc, yq, pl, pq, g)
    print(f"  paired AUROC Llama - Qwen: {d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}; label agreement of text-matched pairs is by construction (labels inherited)")
print("\nunmatched Llama claims written to outputs/unmatched_llama.csv (fill the last column for the manual check, if you have time)")
