"""E7 part 3: sentence-level baseline vs claim-level verification, on the SAME summary sentences.
A sentence is SUPPORTED iff every labelled claim extracted from it is supported (author labels; unverifiable claims excluded).
  python -m train.sentence_baseline --stage build     # GPU ~10 min: entailment/contradiction (shipped retrieval+NLI) and MiniCheck on the SENTENCE text
  python -m train.sentence_baseline --stage eval      # CPU ~2 min: nested leave-one-meeting-out
Methods: entailment, MiniCheck, sentence-level fusion (ent+con+mc, refit in each fold), and the claim-level system aggregated to the sentence
(minimum / mean of the out-of-fold fusion trust of its claims, from outputs/nested_cv.preds.npy)."""
import argparse, glob, json, os
from collections import defaultdict
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["build", "eval"], required=True); ap.add_argument("--features", default="cache/frozen/features.jsonl")
ap.add_argument("--summaries", nargs="+", default=glob.glob("data/meetverify/qwen_*.jsonl")); ap.add_argument("--out", default="cache/frozen/sentence_features.jsonl")
ap.add_argument("--preds", default="outputs/nested_cv.preds.npy"); ap.add_argument("--config", default=None); ap.add_argument("--boots", type=int, default=1000)
a = ap.parse_args()
L = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
F = {}
for r in L(a.features):
    if r.get("author_label") in ("SUPPORTED", "UNSUPPORTED"):
        F.setdefault(r["cid"], r)
groups = defaultdict(list)
for r in F.values():
    groups[(r["summary_id"], r["sent_idx"])].append(r)

if a.stage == "build":
    import torch, gc
    from cprove.text import split_sentences
    from cprove.utils import load_config, seed_everything
    from cprove.data import load_corpus
    from cprove.schema import Claim
    from cprove.nli import make_nli, windows_for, _mix
    from cprove.retrieval_eval import MultiRetriever
    from baselines.run_baselines import transcript_text
    cfg = load_config(a.config); seed_everything(cfg["seed"]); meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
    summ = {}
    for p in a.summaries:
        for r in L(p):
            if r.get("summary"):
                summ[r["summary_id"]] = r["summary"]
    sents = []
    for (sid, si), cl in sorted(groups.items()):
        ss = split_sentences(summ[sid]) if sid in summ else []
        if si < len(ss):
            sents.append(dict(sid=f"{sid}#{si}", summary_id=sid, sent_idx=si, meeting_id=cl[0]["meeting_id"], text=ss[si], n_claims=len(cl),
                              label=int(all(c["author_label"] == "SUPPORTED" for c in cl)), claim_cids=[c["cid"] for c in cl]))
    print(f"{len(sents)} sentences from {len(groups)} claim groups ({sum(s['label'] for s in sents)} supported)", flush=True)
    rc = cfg["retrieval"]; ret = MultiRetriever(rc["encoder"], rc["dense_weight"], rc["sparse_weight"], rc["top_k"], rc["context_window"], rc["slot_boost"]); nli = make_nli(cfg["nli"])
    byM = defaultdict(list)
    for s in sents:
        byM[s["meeting_id"]].append(s)
    for n, (mid, ss) in enumerate(byM.items()):
        m = meetings[mid]; cl = [Claim(cid=s["sid"], text=s["text"]) for s in ss]
        ev = ret.retrieve_mode(m, cl, "hybrid", 5, 2)
        prem, hyp, own, rel = [], [], [], []
        for c in cl:
            for txt, r in windows_for(m, ev[c.cid], 2):
                prem.append(txt); hyp.append(c.text); own.append(c.cid); rel.append(r)
        D = nli.score_pairs(prem, hyp); own = np.array(own); rel = np.array(rel)
        for s in ss:
            i = np.where(own == s["sid"])[0]; mix = _mix(D[i], rel[i]) if len(i) else np.array([0, 1, 0])
            s["ent"], s["con"] = float(mix[0]), float(mix[2])
        print(f"  nli {n + 1}/{len(byM)} {mid}", flush=True)
    del ret, nli; gc.collect(); torch.cuda.empty_cache()
    from minicheck.minicheck import MiniCheck
    mc = MiniCheck(model_name="flan-t5-large", cache_dir="models/hf")
    for n, (mid, ss) in enumerate(byM.items()):
        doc = transcript_text(meetings[mid])
        for b in range(0, len(ss), 2):
            ch = ss[b:b + 2]; _, p, _, _ = mc.score(docs=[doc] * len(ch), claims=[x["text"] for x in ch])
            for x, v in zip(ch, p):
                x["mc"] = float(v)
            torch.cuda.empty_cache()
        print(f"  minicheck {n + 1}/{len(byM)} {mid}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for s in sents:
            f.write(json.dumps(s) + "\n")
    print(f"{len(sents)} sentences -> {a.out}")
else:
    from cprove import metrics as M
    import train.nested_cv as N
    S = L(a.out); P = np.load(a.preds, allow_pickle=True).item(); cids = P.pop("cids"); fus = dict(zip(cids, P["fusion (selected in inner CV)"]))
    S = [s for s in S if all(c in fus for c in s["claim_cids"])]
    rows = [dict(cid=s["sid"], meeting_id=s["meeting_id"], author_label="SUPPORTED" if s["label"] else "UNSUPPORTED", ent=s["ent"], con=s["con"], mc=s["mc"],
                 **{k: 0.0 for k in ("attr", "attr_flag", "sem", "sem_flag", "temp", "temp_flag", "ctx", "ctx_flag", "num_unseen", "con_high")}) for s in S]
    y = np.array([r["author_label"] == "SUPPORTED" for r in rows]).astype(int); g = np.array([r["meeting_id"] for r in rows])
    ncl = np.array([s["n_claims"] for s in S])
    print(f"{len(S)} sentences, {len(set(g))} meetings, {y.mean():.2f} supported; mean {ncl.mean():.2f} claims per sentence; single-claim sentences {100 * (ncl == 1).mean():.0f}%")
    out, _ = N.run(rows, cands={"ent+con+mc": ["ent", "con", "mc"], "ent+con": ["ent", "con"], "mc": ["mc"]})
    preds = {"Entailment (sentence)": np.array([s["ent"] for s in S]), "MiniCheck (sentence)": np.array([s["mc"] for s in S]), "Fusion ent+con+mc (sentence, refit in folds)": out["selected"],
             "Claim-level fusion, min over claims": np.array([min(fus[c] for c in s["claim_cids"]) for s in S]), "Claim-level fusion, mean over claims": np.array([np.mean([fus[c] for c in s["claim_cids"]]) for s in S])}
    res = {}
    print(f"{'method':46} AUROC [95% CI]          AUPRC-unsup  Brier")
    for k, p in preds.items():
        ci = M.cluster_bootstrap(M.auroc, y, p, g, a.boots); res[k] = dict(auroc=M.auroc(y, p), ci=ci, auprc=M.auprc(y, p), brier=M.brier(y, np.clip(p, 0, 1)))
        print(f"{k:46} {res[k]['auroc']:.3f} [{ci[0]:.3f},{ci[1]:.3f}]   {res[k]['auprc']:.3f}        {res[k]['brier']:.3f}")
    ref = "Claim-level fusion, min over claims"
    for k in list(preds)[:3]:
        d, lo, hi, p = M.paired_bootstrap(M.auroc, y, preds[ref], preds[k], g); res[k].update(diff_vs_claim=-d)
        print(f"paired AUROC: claim-level (min) - {k}: {d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}")
    Path("outputs").mkdir(exist_ok=True); json.dump(res, open("outputs/sentence_baseline.json", "w"), indent=1, default=float)
    t = ["\\begin{table}[htbp]", "\\centering", "\\caption{Sentence-level versus claim-level verification on the same summary sentences. A sentence is supported only if all its labelled claims are. The claim-level rows aggregate the out-of-fold fusion trust of the sentence's claims; sentence-level fusion is refitted in every fold.}", "\\label{tab:sentlevel}", "\\small",
         "\\begin{tabular}{@{}lccc@{}}", "\\toprule", "Method & AUROC [95\\% CI] & AUPRC (unsup.) & Brier \\\\", "\\midrule"]
    for k, r in res.items():
        t.append(f"{k} & {r['auroc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] & {r['auprc']:.3f} & {r['brier']:.3f} \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]; open("outputs/table_sentlevel.tex", "w").write("\n".join(t)); print("wrote outputs/table_sentlevel.tex")
