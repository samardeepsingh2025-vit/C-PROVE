"""Retrieval variants + ranking metrics for the evidence-retrieval evaluation.

Modes (all share the SAME claim/utterance indexing as EvidenceRetriever):
  lexical  TF-IDF cosine only                 bm25  Okapi BM25 only          dense  BGE only (char-ngram fallback if no encoder)
  hybrid   the shipped retriever: 0.7*dense + 0.3*tfidf + slot bonus
  rrf      reciprocal-rank fusion of BM25 and dense (k=60), no slot bonus
Metrics vs gold evidence utterances: Recall@k, Hit@k, MRR, nDCG@k at utterance level, and WindowRecall@k (gold within the
+-win window actually passed to the NLI model).
"""
from __future__ import annotations
import math, re
import numpy as np
from .evidence import EvidenceRetriever
from .schema import Claim, Evidence, Meeting

MODES = ("lexical", "bm25", "dense", "hybrid", "rrf")


def _toks(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class MultiRetriever(EvidenceRetriever):
    def _bm25(self, m: Meeting, qs: list[str], k1=1.5, b=0.75):
        key = ("bm25", m.meeting_id)
        if key not in self._cache:
            docs = [_toks(u.text) for u in m.utterances]
            df = {}
            for d in docs:
                for t in set(d):
                    df[t] = df.get(t, 0) + 1
            n = len(docs); avg = max(1e-9, sum(map(len, docs)) / n)
            self._cache[key] = (docs, df, n, avg)
        docs, df, n, avg = self._cache[key]
        out = np.zeros((len(qs), n))
        for i, q in enumerate(qs):
            for t in set(_toks(q)):
                if t not in df:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                for j, d in enumerate(docs):
                    f = d.count(t)
                    if f:
                        out[i, j] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avg))
        return out

    def scores(self, m: Meeting, claims: list[Claim], mode: str) -> np.ndarray:
        tf, U, dense = self._index(m)
        q = [c.text for c in claims]
        S = (tf.transform(q) @ U.T).toarray()
        if mode == "lexical":
            return S
        if mode == "bm25":
            return self._bm25(m, q)
        if isinstance(dense, tuple):
            ch, CU = dense; D = (ch.transform(q) @ CU.T).toarray()
        else:
            Q = self.enc.encode(["Represent this sentence for searching relevant passages: " + t for t in q]); D = Q @ dense.T
        if mode == "dense":
            return D
        if mode == "rrf":
            B = self._bm25(m, q)
            rb = np.argsort(np.argsort(-B, axis=1), axis=1); rd = np.argsort(np.argsort(-D, axis=1), axis=1)
            return 1.0 / (60 + rb + 1) + 1.0 / (60 + rd + 1)
        sc = self.wd * D + self.ws * S                           # hybrid = shipped method
        for i, c in enumerate(claims):
            sc[i] += self.boost * self._slot_bonus(c, m)[0]
        return sc

    def rank(self, m, claims, mode):
        S = self.scores(m, claims, mode)
        return {c.cid: [int(j) for j in np.argsort(-S[i], kind="stable")] for i, c in enumerate(claims)}

    def retrieve_mode(self, m, claims, mode, k=5, win=2) -> dict[str, Evidence]:
        S = self.scores(m, claims, mode); out = {}
        for i, c in enumerate(claims):
            top = list(np.argsort(-S[i], kind="stable")[:k])
            idxs = sorted({j for t in top for j in range(max(0, t - win), min(len(m.utterances), t + win + 1))})
            ev = Evidence(cid=c.cid, uids=[m.utterances[j].uid for j in idxs], scores=[float(S[i][j]) for j in idxs],
                          text="\n".join(f"[{m.utterances[j].uid}] {m.utterances[j].speaker}: {m.utterances[j].text}" for j in idxs),
                          slot_hits={})
            ev.raw_top = [int(t) for t in top]  # type: ignore[attr-defined]
            out[c.cid] = ev
        return out

    def retrieve_oracle(self, m, claim, gold_idx: list[int], win=2) -> Evidence:
        top = sorted(gold_idx)[:5]
        idxs = sorted({j for t in top for j in range(max(0, t - win), min(len(m.utterances), t + win + 1))})
        ev = Evidence(cid=claim.cid, uids=[m.utterances[j].uid for j in idxs], scores=[1.0] * len(idxs),
                      text="\n".join(f"[{m.utterances[j].uid}] {m.utterances[j].speaker}: {m.utterances[j].text}" for j in idxs))
        ev.raw_top = [int(t) for t in top]  # type: ignore[attr-defined]
        return ev


def rank_metrics(ranked: list[int], gold: set[int], ks=(1, 3, 5, 10), win=2) -> dict:
    out = {}
    for k in ks:
        top = ranked[:k]
        out[f"recall@{k}"] = len(gold & set(top)) / len(gold)
        out[f"hit@{k}"] = float(bool(gold & set(top)))
        cover = {j for t in top for j in range(t - win, t + win + 1)}
        out[f"wrecall@{k}"] = len(gold & cover) / len(gold)
        dcg = sum(1 / math.log2(r + 2) for r, j in enumerate(top) if j in gold)
        idcg = sum(1 / math.log2(r + 2) for r in range(min(len(gold), k)))
        out[f"ndcg@{k}"] = dcg / idcg
    first = next((r for r, j in enumerate(ranked) if j in gold), None)
    out["mrr"] = 1.0 / (first + 1) if first is not None else 0.0
    return out


def is_multi_turn(gold: set[int], gap=5) -> bool:
    g = sorted(gold)
    return len(g) >= 2 and (g[-1] - g[0]) > gap
