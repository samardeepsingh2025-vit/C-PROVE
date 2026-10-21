"""Stage 3b — Entailment / contradiction scoring of (evidence window, claim).

Fix over the notebook: the notebook took max(entail) and max(contra) over windows
independently, which produced incoherent pairs such as ent=0.993 & con=0.998 for
the same claim. Here each window yields a proper 3-way distribution and windows
are mixed with relevance weights, so ent + con <= 1 always holds.

Scorers (selected in config):
  deberta     MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (~1.7 GB fp16)
  minicheck   lytang/MiniCheck-Flan-T5-Large  (fact-only, 0..1 support)
  lexical     model-free fallback (CPU smoke tests)
"""
from __future__ import annotations

import numpy as np

from .schema import Claim, Meeting, Evidence
from .text import content_tokens, quantities, polarity, norm
from .utils import SLOT, device, JsonCache, stable_hash


def windows_for(m: Meeting, ev: Evidence, k: int = 2) -> list[tuple[str, float]]:
    tops = getattr(ev, "raw_top", None)
    if tops is None:   # reconstruct from uids
        pos = {u.uid: i for i, u in enumerate(m.utterances)}
        tops = [pos[u] for u in ev.uids if u in pos][:5]
    sc = dict(zip(ev.uids, ev.scores))
    out = []
    for r, t in enumerate(tops):
        w = m.window(t, k)
        out.append(("\n".join(f"{u.speaker}: {u.text}" for u in w), float(sc.get(m.utterances[t].uid, 0.0))))
    return out


def _mix(dists: np.ndarray, rel: np.ndarray, temp: float = 5.0) -> np.ndarray:
    """dists: (W,3) [ent, neu, con]; weight windows by relevance and decisiveness."""
    decisive = np.maximum(dists[:, 0], dists[:, 2])
    logits = temp * (rel - rel.max()) + 3.0 * decisive
    w = np.exp(logits - logits.max())
    w /= w.sum()
    return (w[:, None] * dists).sum(0)


class LexicalNLI:
    """Model-free fallback: compares the claim with the best-matching premise line.
    Only for CPU smoke tests — the paper uses DeBERTa-v3-large NLI / MiniCheck."""
    name = "lexical"

    def score_pairs(self, premises: list[str], hyps: list[str]) -> np.ndarray:
        out = []
        for p, h in zip(premises, hyps):
            H = content_tokens(h)
            lines = [l for l in p.split("\n") if l.strip()] or [p]
            covs = [len(H & content_tokens(l)) / max(1, len(H)) for l in lines]
            order = np.argsort(covs)[::-1]
            best = lines[order[0]]
            cov = len(H & content_tokens(p)) / max(1, len(H))
            qh = set(quantities(h))
            qp = set().union(*[set(quantities(lines[k])) for k in order[:2]])
            qty_conflict = bool(qh) and bool(qp) and not (qh & set(quantities(p)))
            neg_conflict = polarity(h) != polarity(best) and covs[order[0]] >= 0.5
            con = 0.75 if (qty_conflict or neg_conflict) else 0.05
            ent = max(0.0, cov ** 1.5 - (0.6 if con > 0.5 else 0.0))
            neu = max(0.0, 1.0 - ent - con)
            s = ent + neu + con
            out.append([ent / s, neu / s, con / s])
        return np.array(out)


class HFNLI:
    name = "deberta"

    def __init__(self, model_name: str, batch_size=8, max_length=512, fp16=True, cache_file="cache/nli_cache.jsonl"):
        self.model_name, self.bs, self.ml, self.fp16 = model_name, batch_size, max_length, fp16
        self.cache = JsonCache(cache_file)

    def _load(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tok = AutoTokenizer.from_pretrained(self.model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, torch_dtype=torch.float16 if (self.fp16 and device() == "cuda") else torch.float32)
        mdl.to(device()).eval()
        lab = {v.lower(): k for k, v in mdl.config.id2label.items()}
        order = [lab.get("entailment"), lab.get("neutral", lab.get("not_entailment")), lab.get("contradiction")]
        return tok, mdl, order

    def score_pairs(self, premises, hyps):
        import torch
        keys = [stable_hash(self.model_name, p, h) for p, h in zip(premises, hyps)]
        res = [self.cache.get(k) for k in keys]
        todo = [i for i, r in enumerate(res) if r is None]
        if todo:
            tok, mdl, order = SLOT.get("nli:" + self.model_name, self._load)
            for b in range(0, len(todo), self.bs):
                idx = todo[b:b + self.bs]
                enc = tok([premises[i] for i in idx], [hyps[i] for i in idx], truncation="only_first",
                          max_length=self.ml, padding=True, return_tensors="pt").to(device())
                with torch.no_grad():
                    p = torch.softmax(mdl(**enc).logits.float(), -1).cpu().numpy()
                for i, row in zip(idx, p):
                    if order[2] is None:   # 2-class model
                        d = [float(row[order[0]]), float(row[order[1]]), 0.0]
                    else:
                        d = [float(row[o]) for o in order]
                    res[i] = d
                    self.cache.put(keys[i], d)
        return np.array(res)


class MiniCheck:
    """Wrapper for MiniCheck (pip install "minicheck @ git+https://github.com/Liyan06/MiniCheck.git").
    Returns [p_support, 1-p_support, 0] so it plugs into the same mixture."""
    name = "minicheck"

    def __init__(self, model_name="flan-t5-large", cache_file="cache/minicheck_cache.jsonl"):
        self.model_name = model_name
        self.cache = JsonCache(cache_file)

    def score_pairs(self, premises, hyps):
        keys = [stable_hash("mc", self.model_name, p, h) for p, h in zip(premises, hyps)]
        res = [self.cache.get(k) for k in keys]
        todo = [i for i, r in enumerate(res) if r is None]
        if todo:
            from minicheck.minicheck import MiniCheck as MC
            mc = SLOT.get("minicheck", lambda: MC(model_name=self.model_name, cache_dir="models/hf"))
            _, probs, _, _ = mc.score(docs=[premises[i] for i in todo], claims=[hyps[i] for i in todo])
            for i, p in zip(todo, probs):
                res[i] = [float(p), float(1 - p), 0.0]
                self.cache.put(keys[i], res[i])
        return np.array(res)


def make_nli(cfg: dict):
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
        if cfg.get("model") and cfg.get("model") != "lexical":
            return HFNLI(cfg["model"], cfg.get("batch_size", 8), cfg.get("max_length", 512), cfg.get("fp16", True))
    except ImportError:
        pass
    return LexicalNLI()


def score_claims(nli, m: Meeting, claims: list[Claim], evid: dict[str, Evidence], k: int = 2) -> dict[str, dict]:
    """Returns cid -> {ent, neu, con, best_window, per_window}."""
    prem, hyp, owner, rel = [], [], [], []
    for c in claims:
        for txt, r in windows_for(m, evid[c.cid], k):
            prem.append(txt); hyp.append(c.text); owner.append(c.cid); rel.append(r)
    if not prem:
        return {}
    D = nli.score_pairs(prem, hyp)
    owner = np.array(owner); rel = np.array(rel)
    out = {}
    for c in claims:
        idx = np.where(owner == c.cid)[0]
        if len(idx) == 0:
            out[c.cid] = dict(ent=0.0, neu=1.0, con=0.0, best_window=None, per_window=[])
            continue
        mix = _mix(D[idx], rel[idx])
        best = idx[int(np.argmax(D[idx, 0] - D[idx, 2]))]
        out[c.cid] = dict(ent=float(mix[0]), neu=float(mix[1]), con=float(mix[2]),
                          best_window=prem[best], per_window=D[idx].tolist())
    return out
