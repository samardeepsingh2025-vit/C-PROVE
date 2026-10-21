"""Baselines on exactly the same items (same sentences, same gold) as C-PROVE.

  python -m baselines.run_baselines --bench data/meetverify/synth_test.jsonl --systems nli_max summac_zs alignscore minicheck factscore llm_judge notebook_v5
Each system writes outputs/preds/<bench>__<system>.jsonl with one row per summary sentence:
  {summary_id, meeting_id, sent_idx, score (P faithful), gold, error_type}

Install the optional ones (each needs ~1-2 GB VRAM; they are run one at a time):
  pip install summac                                    # SummaC-ZS / Conv (Laban et al. 2022)
  pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git@main"   # MiniCheck (Tang et al. 2024)
  git clone https://github.com/yuh-zha/AlignScore && pip install -e AlignScore   # + AlignScore-large.ckpt in models/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cprove.utils import load_config, SLOT, device
from cprove.data import load_corpus
from cprove.pipeline import CProve, Ablation
from cprove.nli import score_claims, windows_for
from cprove.judge import LLMJudge
from experiments.common import load_records, build_items, units, unit_scores


def transcript_text(m, max_words=None):
    lines = [f"{u.speaker}: {u.text}" for u in m.utterances]
    return "\n".join(lines)


def unit_rows(it, scores, system):
    return [{"summary_id": it.rec["summary_id"], "meeting_id": it.meeting_id, "unit": uid, "score": float(s),
             "gold": g, "error_type": e, "system": system}
            for (uid, _, g, e), s in zip(units(it), scores)]


def unit_texts(it):
    return [t for _, t, _, _ in units(it)]


# ------------------------------------------------------------------ systems
def nli_max(cp, items, meetings):
    """Notebook-style: claim score = max entailment over retrieved windows (no contradiction, no graph)."""
    out = []
    for it in items:
        m = meetings[it.meeting_id]
        ev = cp.retriever.retrieve(m, it.claims)
        sc = score_claims(cp.nli, m, it.claims, ev)
        post = {c.cid: {"y": max(d[0] for d in sc[c.cid]["per_window"]) if sc[c.cid]["per_window"] else 0.0} for c in it.claims}
        out += unit_rows(it, unit_scores(it, post), "nli_max")
    return out


def factscore(cp, items, meetings):
    """FActScore-style: atomic claims + retrieval + NLI; a claim is supported if ent > 0.5 and ent > con;
    sentence score = fraction of its supported claims (claim-level benches use the claim value)."""
    out = []
    for it in items:
        sup = {c.cid: float(it.obs[c.cid].ent > 0.5 and it.obs[c.cid].ent > it.obs[c.cid].con) for c in it.claims}
        if it.sent_gold is not None:
            n = len(it.sent_gold)
            per = [[] for _ in range(n)]
            for c in it.claims:
                if c.sent_idx < n:
                    per[c.sent_idx].append(sup[c.cid])
            sc = [float(np.mean(v)) if v else 1.0 for v in per]
        else:
            sc = unit_scores(it, {k: {"y": v} for k, v in sup.items()})
        out += unit_rows(it, sc, "factscore")
    return out


def summac(items, meetings, conv=False):
    from summac.model_summac import SummaCZS, SummaCConv
    model = SummaCConv(models=["vitc"], bins="percentile", granularity="sentence", nli_labels="e", device=device(),
                       start_file="default", agg="mean") if conv else \
        SummaCZS(granularity="sentence", model_name="vitc", device=device())
    out = []
    for it in items:
        doc = transcript_text(meetings[it.meeting_id])
        sents = unit_texts(it)
        sc = [model.score([doc], [s])["scores"][0] for s in sents]
        # SummaC-ZS is in [-1, 1]
        sc = [(s + 1) / 2 if not conv else s for s in sc]
        out += unit_rows(it, sc, "summac_conv" if conv else "summac_zs")
    del model
    SLOT.release()
    return out


def alignscore(items, meetings, ckpt="models/AlignScore-large.ckpt"):
    from alignscore import AlignScore
    scorer = AlignScore(model="roberta-large", batch_size=16, device=device(), ckpt_path=ckpt, evaluation_mode="nli_sp")
    out = []
    for it in items:
        doc = transcript_text(meetings[it.meeting_id])
        sents = unit_texts(it)
        out += unit_rows(it, scorer.score(contexts=[doc] * len(sents), claims=sents), "alignscore")
    del scorer
    SLOT.release()
    return out


def minicheck(items, meetings, name="flan-t5-large"):
    from minicheck.minicheck import MiniCheck
    mc = MiniCheck(model_name=name, cache_dir="models/hf")
    out = []
    for it in items:
        doc = transcript_text(meetings[it.meeting_id])
        sents = unit_texts(it)
        _, probs, _, _ = mc.score(docs=[doc] * len(sents), claims=sents)
        out += unit_rows(it, probs, "minicheck_" + name.split("-")[0])
    del mc
    SLOT.release()
    return out


def llm_judge(cp, items, meetings):
    """Qwen2.5-7B judge on EVERY claim, no graph, no rules (upper-cost baseline)."""
    out = []
    for it in items:
        m = meetings[it.meeting_id]
        ev = cp.retriever.retrieve(m, it.claims)
        j = LLMJudge(cp.llm, it.claims, ev)
        post = {c.cid: {"y": (j(c.cid) if cp.llm.available else 0.5) or 0.5} for c in it.claims}
        out += unit_rows(it, unit_scores(it, post), "llm_judge")
    return out


def notebook_v5(cp, items, meetings, w=(0.30, 0.15, 0.15, 0.15, 0.15, 0.10), alpha=0.3):
    """Port of the original notebook (cells 8-17): uncalibrated 6D weighted fusion (missing
    dimensions imputed with 0.5), then one round of neighbour-averaging propagation over an
    untyped similarity graph. Kept as the 'prior version' baseline."""
    from cprove.graph import build_cdg
    out = []
    for it in items:
        s = {}
        for c in it.claims:
            o = it.obs[c.cid]
            r = o.raw
            dims = [r["fact"], r.get("attr"), r.get("temp"), r.get("ctx"), r.get("sem"), 1 - o.con]
            dims = [0.5 if d is None else d for d in dims]
            s[c.cid] = float(np.dot(w, dims))
        E = build_cdg(it.claims, mode="similarity")
        nb = {c.cid: [] for c in it.claims}
        for e in E:
            nb[e.src].append(e.dst); nb[e.dst].append(e.src)
        post = {cid: {"y": (1 - alpha) * v + alpha * (np.mean([s[j] for j in nb[cid]]) if nb[cid] else v)} for cid, v in s.items()}
        out += unit_rows(it, unit_scores(it, post), "notebook_v5")
    return out


SYSTEMS = ["nli_max", "factscore", "summac_zs", "summac_conv", "alignscore", "minicheck", "llm_judge", "notebook_v5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--systems", nargs="+", default=["nli_max", "factscore", "notebook_v5"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
    cp = CProve(cfg, Ablation(hand_weights=True))
    from cprove.dimensions import Calibrators
    cp.dims.cal = Calibrators("none")
    recs = load_records(a.bench)
    items = build_items(cp, meetings, recs, str(Path(cfg["paths"]["cache"]) / f"items_{Path(a.bench).stem}_raw.pkl"))
    od = Path(cfg["paths"]["outputs"]) / "preds"
    od.mkdir(parents=True, exist_ok=True)
    for s in a.systems:
        print("==", s)
        fn = {"nli_max": lambda: nli_max(cp, items, meetings), "factscore": lambda: factscore(cp, items, meetings),
              "summac_zs": lambda: summac(items, meetings), "summac_conv": lambda: summac(items, meetings, True),
              "alignscore": lambda: alignscore(items, meetings), "minicheck": lambda: minicheck(items, meetings),
              "minicheck_deberta": lambda: minicheck(items, meetings, "deberta-v3-large"),
              "llm_judge": lambda: llm_judge(cp, items, meetings), "notebook_v5": lambda: notebook_v5(cp, items, meetings)}[s]
        try:
            rows = fn()
        except ImportError as e:
            print(f"   skipped ({e}); see install notes at the top of this file")
            continue
        with open(od / f"{Path(a.bench).stem}__{s}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print("   ", len(rows), "rows")


if __name__ == "__main__":
    main()
