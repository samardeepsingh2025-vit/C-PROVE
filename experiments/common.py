"""Shared experiment helpers: turn benchmark records into cached C-PROVE programs,
evaluate posteriors, and map claim scores to sentence level.

Benchmark record formats
  synth / tofueval : {summary_id, meeting_id, sentences[], gold[] (per sentence), error_type, sent_idx[]}
  human            : {summary_id, meeting_id, summary, claims:[{cid, gold, error_type}]}  (claim-level gold)
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cprove.pipeline import CProve
from cprove.rules import ground


@dataclass
class Item:
    rec: dict
    claims: list
    edges: list
    obs: dict
    grounding: dict
    order_ok: dict
    gold: dict          # cid -> 0/1
    sent_gold: list     # per sentence (synth/tofueval) or None
    meeting_id: str


def load_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


_STOP = {"the", "a", "an", "of", "to", "and", "in", "on", "for", "is", "was", "that", "it"}


def _words(t: str) -> set:
    import re
    return set(re.findall(r"[a-z0-9.]+", (t or "").lower())) - _STOP


def refine_gold(items: list, records: list[dict]) -> list:
    """Claim-level gold from sentence-level gold: in an injected sentence only the claims that carry the
    corrupted tokens (words not in the clean sentence) are errors; other claims of that sentence stay faithful.
    Falls back to the whole sentence when no claim carries a changed token. Idempotent."""
    import re
    clean = {r["summary_id"]: r for r in records if r["summary_id"].endswith("_clean")}
    for it in items:
        r = it.rec
        if "sentences" not in r or not r.get("sent_idx") or r.get("error_type", "none") == "none":
            continue
        cl = clean.get(re.sub(rf"_{r['error_type']}_\d+$", "", r["summary_id"]) + "_clean")
        if not cl:
            continue
        for i in r["sent_idx"]:
            if i >= len(cl["sentences"]) or i >= len(r["sentences"]):
                continue
            new_toks = _words(r["sentences"][i]) - _words(cl["sentences"][i])
            cs = [c for c in it.claims if c.sent_idx == i and c.cid in it.gold]
            hit = [c for c in cs if _words(c.text) & new_toks]
            if not hit or len(hit) == len(cs):
                continue
            for c in cs:
                if c not in hit:
                    it.gold[c.cid] = 1
                    c.gold_label = "SUPPORTED"
                    c.gold_error = "none"
    return items


def build_items(cp: CProve, meetings: dict, records: list[dict], cache_path: str | None = None, verbose=True) -> list[Item]:
    """Stages 1-4 for every record (the expensive, model-heavy part), cached to disk.
    Stage 5/6 (grounding + inference) is rebuilt cheaply per experiment/ablation."""
    if cache_path and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            return refine_gold(pickle.load(f), records)
    import time
    items, timing = [], []
    for k, r in enumerate(records):
        m = meetings[r["meeting_id"]]
        text = r.get("summary") or " ".join(r["sentences"])
        t0 = time.perf_counter()
        claims = cp.decompose(m, text, r["summary_id"])
        t1 = time.perf_counter()
        gold = {}
        if "sentences" in r:
            # sentence-level gold -> claims inherit their sentence's label
            for c in claims:
                if c.sent_idx < len(r["gold"]):
                    gold[c.cid] = int(r["gold"][c.sent_idx])
                    c.gold_label = "SUPPORTED" if gold[c.cid] else "UNSUPPORTED"
                    c.gold_error = r.get("error_type", "none") if c.sent_idx in r.get("sent_idx", []) else "none"
        else:
            by = {g["cid"]: g for g in r.get("claims", [])}
            for c in claims:
                if c.cid in by:
                    gold[c.cid] = int(by[c.cid]["gold"])
                    c.gold_error = by[c.cid].get("error_type", "none")
        edges = cp.edges_for(claims, r["summary_id"])
        t2 = time.perf_counter()
        _, _, obs, grounding = cp.observe(m, claims)
        t3 = time.perf_counter()
        timing.append({"frames_s": t1 - t0, "graph_s": t2 - t1, "evidence_nli_dims_s": t3 - t2, "claims": len(claims)})
        from cprove.rules import order_consistency
        ok = order_consistency(claims, edges, grounding)
        items.append(Item(r, claims, edges, obs, grounding, ok, gold, r.get("gold") if "sentences" in r else None,
                          r["meeting_id"]))
        if verbose and (k + 1) % 25 == 0:
            print(f"  built {k + 1}/{len(records)}")
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(items, f)
        with open(Path(cache_path).parent / "stage_timing.jsonl", "a") as f:
            for t in timing:
                f.write(json.dumps(t) + "\n")
    return refine_gold(items, records)


def programs(items: list[Item], disabled=(), graph=True, drop_dims=()):
    progs = []
    for it in items:
        obs = it.obs
        if drop_dims:
            import copy
            obs = {k: copy.copy(v) for k, v in obs.items()}
            for o in obs.values():
                for d in drop_dims:
                    setattr(o, d, None)
        progs.append(ground(it.claims, obs, it.edges if graph else [], it.order_ok, disabled))
    return progs


def sentence_scores(item: Item, post: dict) -> list[float]:
    """Sentence faithfulness = min over its claims (a sentence is wrong if any claim is)."""
    n = len(item.sent_gold) if item.sent_gold is not None else 1 + max(c.sent_idx for c in item.claims)
    out = [1.0] * n
    for c in item.claims:
        if c.sent_idx < n:
            out[c.sent_idx] = min(out[c.sent_idx], post[c.cid]["y"])
    return out


def units(item: Item):
    """Evaluation units: summary sentences (synth / TofuEval, sentence-level gold) or
    atomic claims (MeetVerify-Human, claim-level gold). Returns [(uid, text, gold, error_type)]."""
    if item.sent_gold is not None:
        err = set(item.rec.get("sent_idx", []))
        return [(f"{item.rec['summary_id']}#s{i}", s, int(g), item.rec.get("error_type", "none") if i in err else "none")
                for i, (s, g) in enumerate(zip(item.rec["sentences"], item.sent_gold))]
    return [(c.cid, c.text, item.gold[c.cid], c.gold_error) for c in item.claims if c.cid in item.gold]


def unit_scores(item: Item, post: dict) -> list[float]:
    if item.sent_gold is not None:
        return sentence_scores(item, post)
    return [post[c.cid]["y"] for c in item.claims if c.cid in item.gold]


def metrics(y_true, y_score, thr=0.5) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    out = {"n": int(len(y_true)), "pos_rate": float(y_true.mean())}
    if len(set(y_true)) == 2:
        # "unsupported" is the positive class for detection
        out["auroc"] = float(roc_auc_score(1 - y_true, 1 - y_score))
        out["auprc_err"] = float(average_precision_score(1 - y_true, 1 - y_score))
    pred = (y_score >= thr).astype(int)
    out["bacc"] = float(balanced_accuracy_score(y_true, pred)) if len(set(y_true)) == 2 else float("nan")
    out["f1_err"] = float(f1_score(1 - y_true, 1 - pred, zero_division=0))
    return out


def best_threshold(y_true, y_score):
    from sklearn.metrics import balanced_accuracy_score
    grid = np.linspace(0.05, 0.95, 91)
    accs = [balanced_accuracy_score(y_true, (np.asarray(y_score) >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(accs))])


def ece(y_true, p, bins=10):
    y_true, p = np.asarray(y_true), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b if b < 1 else p <= b)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y_true[m].mean())
    return float(e)