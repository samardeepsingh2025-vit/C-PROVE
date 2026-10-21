"""Stage 3 — Slot-aware hybrid evidence retrieval.

score(u | c) = 0.7 * cos(BGE(c), BGE(u)) + 0.3 * tfidf(c, u) + slot_boost * #slots of c grounded in u
Top-k hits are expanded to +-2 utterance windows (as in the original notebook) and
merged chronologically. Without sentence-transformers the dense part falls back to
TF-IDF with character n-grams so everything runs on CPU.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .schema import Claim, Evidence, Meeting
from .text import norm, content_tokens, quantities
from .utils import SLOT


class _Encoder:
    def __init__(self, name: str):
        self.name = name
        self.st = None
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            self.ok = True
        except ImportError:
            self.ok = False

    def encode(self, texts: list[str]) -> np.ndarray | None:
        if not self.ok:
            return None
        from sentence_transformers import SentenceTransformer
        from .utils import device
        model = SLOT.get("enc:" + self.name, lambda: SentenceTransformer(self.name, device=device()))
        return model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)


class EvidenceRetriever:
    def __init__(self, encoder: str = "BAAI/bge-base-en-v1.5", dense_weight=0.7, sparse_weight=0.3,
                 top_k=5, context_window=2, slot_boost=0.15, use_dense=True):
        self.enc = _Encoder(encoder) if use_dense else None
        self.wd, self.ws = dense_weight, sparse_weight
        self.k, self.win, self.boost = top_k, context_window, slot_boost
        self._cache = {}

    def _index(self, m: Meeting):
        if m.meeting_id in self._cache:
            return self._cache[m.meeting_id]
        texts = [u.text for u in m.utterances]
        tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(texts + [""])
        U = tf.transform(texts)
        dense = self.enc.encode([f"{u.speaker}: {u.text}" for u in m.utterances]) if self.enc else None
        if dense is None:   # char-ngram tf-idf as a dense surrogate
            ch = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(texts + [""])
            dense = (ch, ch.transform(texts))
        self._cache[m.meeting_id] = (tf, U, dense)
        return self._cache[m.meeting_id]

    def retrieve(self, m: Meeting, claims: list[Claim]) -> dict[str, Evidence]:
        tf, U, dense = self._index(m)
        q_text = [c.text for c in claims]
        S = (tf.transform(q_text) @ U.T).toarray()
        if isinstance(dense, tuple):
            ch, CU = dense
            D = (ch.transform(q_text) @ CU.T).toarray()
        else:
            Q = self.enc.encode(["Represent this sentence for searching relevant passages: " + t for t in q_text])
            D = Q @ dense.T
        score = self.wd * D + self.ws * S
        out = {}
        for i, c in enumerate(claims):
            bonus, slot_hits = self._slot_bonus(c, m)
            sc = score[i] + self.boost * bonus
            top = np.argsort(-sc)[: self.k]
            idxs = sorted({j for t in top for j in range(max(0, t - self.win), min(len(m.utterances), t + self.win + 1))})
            uids = [m.utterances[j].uid for j in idxs]
            text = "\n".join(f"[{m.utterances[j].uid}] {m.utterances[j].speaker}: {m.utterances[j].text}" for j in idxs)
            out[c.cid] = Evidence(cid=c.cid, uids=uids, scores=[float(sc[j]) for j in idxs], text=text,
                                  slot_hits={k: m.utterances[v].uid for k, v in slot_hits.items()})
            out[c.cid].raw_top = [int(t) for t in top]  # type: ignore[attr-defined]
        return out

    def _slot_bonus(self, c: Claim, m: Meeting):
        f = c.frame
        bonus = np.zeros(len(m.utterances))
        hits = {}
        obj_toks = content_tokens(f.obj or "")
        qty = set(quantities(f.quantity or "")) if f.quantity else set()
        for j, u in enumerate(m.utterances):
            ut = norm(u.text)
            if f.agent and u.speaker == f.agent:
                bonus[j] += 0.5
            if obj_toks and len(obj_toks & content_tokens(u.text)) >= max(1, len(obj_toks) // 2):
                bonus[j] += 1.0
                hits.setdefault("object", j)
            if qty and qty & set(quantities(u.text)):
                bonus[j] += 1.0
                hits.setdefault("quantity", j)
            if f.time and norm(f.time) and norm(f.time) in ut:
                bonus[j] += 0.5
                hits.setdefault("time", j)
        return bonus, hits

    def object_timeline(self, m: Meeting, c: Claim, min_overlap: int = 1) -> list[int]:
        """All utterance indices that mention the claim's object (chronological).
        Used by the contextual (latest-state) dimension."""
        toks = content_tokens(c.frame.obj or c.text)
        if not toks:
            return []
        need = max(min_overlap, (len(toks) + 1) // 2)
        return [j for j, u in enumerate(m.utterances) if len(toks & content_tokens(u.text)) >= need]
