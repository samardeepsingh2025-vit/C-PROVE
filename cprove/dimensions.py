"""Stage 4 — Six confidence dimensions (observations for the HL-MRF).

Each dimension answers ONE question about a claim, with its own evidence signal,
so they are not five re-weightings of the same NLI score (a reviewer complaint):

  fact  Is the content entailed?                 NLI/MiniCheck mixture (ent, con)
  attr  Is it attributed to the right person?    speaker of grounding utterance + acceptance acts
  temp  Is the time/deadline right?              normalised time slots vs evidence times
  ctx   Is it the latest state (not revised)?    object timeline after the grounding point
  sem   Is the status right (proposed/decided)?  status classifier on evidence vs claim status
  rsn   Is it consistent with related claims?    latent only — inferred by graph rules (no observation)

A dimension that does not apply (no agent slot, no time slot, plain 'stated'
status) returns None and creates no atom: it is never imputed with 0.5.
Raw scores are calibrated per dimension (isotonic by default) on the dev split.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from .schema import Claim, Meeting, Evidence, Observation
from .text import (content_tokens, normalise_time, extract_time, time_compatible, status_cues,
                   guess_status, STATUS_COMPAT, quantities, TIME_RX)

ACCEPT = ("yeah", "yes", "okay", "ok", "right", "sure", "agreed", "agree", "good", "fine", "great", "definitely", "exactly", "mm-hmm", "uh-huh")


# ----------------------------------------------------------------- status model
class StatusModel:
    """Predicts the status expressed by an utterance window. Uses the fine-tuned
    DeBERTa-v3-base (train/train_status.py) when available, else lexical cues."""
    LABELS = ["proposed", "agreed", "decided", "rejected", "assigned", "deferred", "revised", "stated"]

    def __init__(self, path: Optional[str] = None):
        self.mdl = None
        self.lr = None
        lr_path = Path(str(path) + "_lr.pkl") if path else None
        if lr_path is not None and lr_path.exists():      # logistic regression on BGE embeddings (train/status_lr.py)
            try:
                with open(lr_path, "rb") as f:
                    self.lr = pickle.load(f)
                from sentence_transformers import SentenceTransformer
                from .utils import device
                self._enc = SentenceTransformer(self.lr["encoder"], device=device())
            except Exception as e:  # pragma: no cover
                print("[cprove] LR status classifier unavailable:", e)
                self.lr = None
        if self.lr is None and path and Path(path).exists():
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                from .utils import device
                self.tok = AutoTokenizer.from_pretrained(path)
                self.mdl = AutoModelForSequenceClassification.from_pretrained(path).to(device()).eval()
                self.torch, self.dev = torch, device()
            except Exception as e:  # pragma: no cover
                print("[cprove] status classifier unavailable:", e)

    def predict(self, texts: list[str]) -> np.ndarray:
        if self.lr is not None:
            from train.status_lr import featurize
            emb = lambda xs: self._enc.encode(xs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            P = self.lr["clf"].predict_proba(featurize(emb, texts))
            out = np.zeros((len(texts), len(self.LABELS)))
            for j, lab in enumerate(self.lr["labels"]):
                out[:, self.LABELS.index(lab)] = P[:, j]
            return out
        if self.mdl is not None:
            enc = self.tok(texts, truncation=True, max_length=192, padding=True, return_tensors="pt").to(self.dev)
            with self.torch.no_grad():
                return self.torch.softmax(self.mdl(**enc).logits, -1).cpu().numpy()
        out = np.zeros((len(texts), len(self.LABELS)))
        # fallback: status cues (negation-aware) — replace with the fine-tuned classifier for the paper
        for i, t in enumerate(texts):
            cues = status_cues(t)
            v = np.array([cues.get(l, 0.0) for l in self.LABELS]) + 1e-3
            v[-1] += 0.3   # 'stated' prior
            out[i] = v / v.sum()
        return out


# ----------------------------------------------------------------- calibration
class Calibrators:
    """One monotone map per dimension, fit on dev (raw score -> P(claim faithful))."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.maps = {}

    def fit(self, raw: dict[str, list[float]], y: dict[str, list[int]]):
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        for d in raw:
            x, t = np.asarray(raw[d], float), np.asarray(y[d], int)
            if len(x) < 20 or len(set(t)) < 2:
                continue
            if self.method == "isotonic":
                m = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99).fit(x, t)
            else:   # temperature/Platt on logit
                m = LogisticRegression().fit(_logit(x)[:, None], t)
            self.maps[d] = m
        return self

    def __call__(self, d: str, v: Optional[float]) -> Optional[float]:
        if v is None or d not in self.maps or self.method == "none":
            return v
        m = self.maps[d]
        if hasattr(m, "predict_proba"):
            return float(m.predict_proba(_logit(np.array([v]))[:, None])[0, 1])
        return float(m.predict([v])[0])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path, method="isotonic"):
        if path and Path(path).exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return Calibrators(method)


def _logit(x):
    x = np.clip(x, 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


# ----------------------------------------------------------------- dimension functions
def _grounding_idx(m: Meeting, c: Claim, ev: Evidence) -> list[int]:
    """Utterances in the evidence that best ground the claim content (by token overlap)."""
    pos = {u.uid: i for i, u in enumerate(m.utterances)}
    toks = content_tokens(c.frame.obj or c.text) | content_tokens(c.text)
    scored = []
    for uid in ev.uids:
        j = pos[uid]
        ov = len(toks & content_tokens(m.utterances[j].text)) / max(1, len(toks))
        scored.append((ov, j))
    scored.sort(reverse=True)
    best = scored[0][0] if scored else 0
    return [j for ov, j in scored if ov >= max(0.15, 0.6 * best)][:3]


def dim_attr(m: Meeting, c: Claim, ev: Evidence, G: list[int]) -> Optional[float]:
    f = c.frame
    who = f.recipient if f.status == "assigned" and f.recipient else f.agent
    if not who:
        return None
    if who == "group":
        # a collective has no individual owner to check: inapplicable (group acceptance is judged by `sem`)
        return None
    if who not in m.speakers:
        # name not a speaker id (e.g. summary uses role names not in meta) — check mentions
        hit = any(who.lower() in m.utterances[j].text.lower() for j in G)
        return 0.7 if hit else 0.35
    if not G:
        return 0.3
    said = [m.utterances[j].speaker == who for j in G]
    if f.status == "assigned":
        # owner either says "I will ..." or is addressed / named in the grounding window
        for j in G:
            for u in m.window(j, 2):
                t = u.text.lower()
                if u.speaker == who and any(k in t for k in ("i'll", "i will", "i can", "i'm going to", "okay")):
                    return 0.95
                if who.lower() in t:
                    return 0.8
    if said[0]:
        return 0.95
    if any(said):
        return 0.75
    # grounding utterance by someone else -> likely attribution swap
    return 0.1


def dim_temp(m: Meeting, c: Claim, ev: Evidence, G: list[int], ref_date=None) -> Optional[float]:
    tn = c.frame.time_norm
    if not tn or tn.startswith("ORD:") or tn.startswith("RAW:"):
        return None
    found, conflicts = 0, 0
    near = sorted({u.idx for j in G for u in m.window(j, 1)})
    for k in near:
        u = m.utterances[k]
        for mm in TIME_RX.finditer(u.text):
            t2 = normalise_time(mm.group(0), ref_date)
            comp = time_compatible(tn, t2)
            if comp == 1.0:
                found += 1
            elif comp == 0.0:
                conflicts += 1
    if found and not conflicts:
        return 0.95
    if found and conflicts:
        return 0.6
    if conflicts:
        return 0.08
    return 0.4    # time slot asserted but nothing in evidence -> weakly unsupported


def dim_ctx(m: Meeting, c: Claim, G: list[int], timeline: list[int]) -> Optional[float]:
    """Latest-state check: after the grounding point, is there a later utterance
    about the same object that revises / rejects / changes its value?"""
    # only claims that assert a *current state* can be stale; a correctly reported
    # proposal or revision is not invalidated by later discussion
    if c.frame.status in ("revised", "proposed") or not timeline or not G:
        return None
    q_claim = set(quantities(c.frame.quantity or c.text))
    # anchor = latest grounding utterance that matches the claim's value (if it has one)
    anchors = [j for j in G if not q_claim or (q_claim & set(quantities(m.utterances[j].text)))]
    g = max(anchors) if anchors else max(G)
    later = [j for j in timeline if j > g]
    if not later:
        return 0.9
    worst = 0.0
    for j in later:
        t = m.utterances[j].text
        q = set(quantities(t))
        if q_claim and (q_claim & q):
            continue            # later utterance restates the claimed value -> support, not revision
        cues = status_cues(t)
        sig = max(cues["revised"], cues["rejected"] * 0.8, cues["decided"] * 0.5)
        if q_claim and q and (cues["decided"] or cues["agreed"] or cues["revised"]):
            sig = max(sig, 0.9)
        worst = max(worst, sig)
    return float(np.clip(1.0 - 0.85 * worst, 0.05, 0.95))


def dim_sem(m: Meeting, c: Claim, G: list[int], status_model: StatusModel) -> Optional[float]:
    cs = c.frame.status
    if cs == "stated" or not G:
        return None
    wins = ["\n".join((">> " if u.idx == j else "   ") + f"{u.speaker}: {u.text}" for u in m.window(j, 2)) for j in G]
    P = status_model.predict(wins)                     # (W, 8)
    compat = np.array([STATUS_COMPAT[cs][l] for l in StatusModel.LABELS])
    s = (P @ compat).max()
    # 'decided/agreed' needs agreement from someone other than the proposer
    if cs in ("decided", "agreed"):
        spk = {u.speaker for j in G for u in m.window(j, 2)}
        if len(spk) < 2:
            s *= 0.6
    return float(np.clip(s, 0.02, 0.98))


SCALE_ENT_ON_UNSEEN = True      # legacy behaviour; frozen/revision config sets numeric.scale_entailment: false
_TXT_CACHE: dict = {}


def _transcript_text(m: Meeting) -> str:
    t = _TXT_CACHE.get(m.meeting_id)
    if t is None:
        t = _TXT_CACHE[m.meeting_id] = " ".join(u.text for u in m.utterances)
    return t


def numeric_check(m: Meeting, c: Claim, allow_arith: bool = False) -> dict:
    """Deterministic numeric-inconsistency check on normalised VALUES (digits, spelled-out numbers, decimals, k/million)."""
    from .numnorm import numeric_report
    return numeric_report(_transcript_text(m), c.text, allow_arith)


def unseen_numbers(m: Meeting, c: Claim) -> bool:
    """True if the claim states a numeric value that occurs nowhere in the transcript (after normalisation)."""
    return bool(numeric_check(m, c)["unseen"])


class DimensionScorer:
    def __init__(self, status_model: StatusModel, calibrators: Calibrators, retriever=None):
        self.sm, self.cal, self.ret = status_model, calibrators, retriever

    def raw(self, m: Meeting, c: Claim, ev: Evidence, nli: dict, ref_date=None) -> dict:
        G = _grounding_idx(m, c, ev)
        tl = self.ret.object_timeline(m, c) if self.ret else []
        nli = dict(nli)
        num_unseen = unseen_numbers(m, c)
        if num_unseen and SCALE_ENT_ON_UNSEEN:        # legacy: veto folded into the NLI feature
            nli["ent"] *= 0.2
        fact_raw = float(np.clip(0.5 + 0.5 * (nli["ent"] - nli["con"]), 0, 1))
        return dict(
            ent=nli["ent"], con=nli["con"], fact=fact_raw, num_unseen=float(num_unseen),
            attr=dim_attr(m, c, ev, G), temp=dim_temp(m, c, ev, G, ref_date),
            ctx=dim_ctx(m, c, G, tl), sem=dim_sem(m, c, G, self.sm), grounding=G,
        )

    def observe(self, m: Meeting, c: Claim, ev: Evidence, nli: dict, ref_date=None, drop: tuple = ()) -> Observation:
        r = self.raw(m, c, ev, nli, ref_date)
        o = Observation(cid=c.cid, ent=r["ent"], con=r["con"], raw=r)
        for d in ("fact", "attr", "temp", "ctx", "sem"):
            if d in drop:
                continue
            setattr(o, d, self.cal(d, r[d]))
        return o
