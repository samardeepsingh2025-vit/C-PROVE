"""Stage 2 — Claim Dependency Graph (CDG) with typed edges.

Edge types: before, causes, same_agent, task_depends, coref, revises, conflicts.
Candidate pairs are pruned to (same object) OR (same agent) OR (|sent_i - sent_j| <= 3).
Typing: deterministic frame rules (high precision), optionally overridden by a
fine-tuned DeBERTa-v3-base pair classifier (train/train_edges.py) when present.

Convention: an edge (i -> j, revises) means "j revises i"; (i -> j, task_depends)
means "j depends on i being done"; (i -> j, causes) means "i causes j";
(i -> j, before) means "i happens before j".
"""
from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import networkx as nx

from .schema import Claim, Edge
from .text import jaccard, content_tokens, quantities

CAUSE_CUE = re.compile(r"\b(because|since|so that|therefore|thus|as a result|due to|hence|consequently|which (led|means))\b", re.I)
AFTER_CUE = re.compile(r"\b(then|later|afterwards|after that|subsequently|finally|next)\b", re.I)
DEPEND_CUE = re.compile(r"\b(after|once|based on|following|depending on|using the|when .* (is|are) (done|ready|finished))\b", re.I)
REVISE_CUE = re.compile(r"\b(instead|changed|switched|revised|reconsidered|no longer|later decided|eventually|in the end|but then)\b", re.I)


_TIMEW = set("monday tuesday wednesday thursday friday saturday sunday today tomorrow next week month meeting "
             "time later before after end day morning afternoon will".split())


def _obj_overlap(a: Claim, b: Claim) -> float:
    A = content_tokens(a.frame.obj or a.text) - _TIMEW
    B = content_tokens(b.frame.obj or b.text) - _TIMEW
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _ord(c: Claim):
    t = c.frame.time_norm or ""
    if t.startswith("ORD:"):
        return int(t[4:])
    if t.startswith("MTG:"):
        return 10 * int(t[4:])
    return None


def candidate_pairs(claims: list[Claim], window: int = 3, obj_thr: float = 0.25):
    for a, b in combinations(claims, 2):
        if a.summary_id != b.summary_id:
            continue
        near = abs(a.sent_idx - b.sent_idx) <= window
        same_ag = a.frame.agent and a.frame.agent == b.frame.agent and a.frame.agent != "group"
        same_obj = _obj_overlap(a, b) >= obj_thr
        if near or same_ag or same_obj:
            yield (a, b) if (a.sent_idx, a.cid) <= (b.sent_idx, b.cid) else (b, a)


def rule_edges(a: Claim, b: Claim) -> list[Edge]:
    """a precedes b in the summary."""
    E = []
    fa, fb = a.frame, b.frame
    ov = _obj_overlap(a, b)
    if fa.agent and fa.agent == fb.agent and fa.agent != "group":
        E.append(Edge(a.cid, b.cid, "same_agent", 1.0))
    # coreference: near-duplicate claims
    if jaccard(a.text, b.text) >= 0.6 and fa.status == fb.status:
        E.append(Edge(a.cid, b.cid, "coref", 0.9))
    # temporal order
    oa, ob = _ord(a), _ord(b)
    if oa is not None and ob is not None and oa != ob:
        E.append(Edge(a.cid, b.cid, "before", 0.9) if oa < ob else Edge(b.cid, a.cid, "before", 0.9))
    elif AFTER_CUE.search(b.text) and b.sent_idx - a.sent_idx in (0, 1) and (ov > 0.1 or b.sent_idx == a.sent_idx):
        E.append(Edge(a.cid, b.cid, "before", 0.7))
    # causation (explicit discourse marker in the later claim, adjacent)
    if CAUSE_CUE.search(b.text) and b.sent_idx - a.sent_idx in (0, 1):
        E.append(Edge(a.cid, b.cid, "causes", 0.7))
    # revision: same object, later claim is decided/revised and differs in value or cue
    if ov >= 0.25 and fa.status in ("proposed", "agreed", "stated", "decided") and fb.status in ("decided", "revised", "agreed", "rejected"):
        qa, qb = set(quantities(fa.quantity or a.text)), set(quantities(fb.quantity or b.text))
        value_change = bool(qa) and bool(qb) and not (qa & qb)
        if REVISE_CUE.search(b.text) or (value_change and fb.status in ("decided", "revised")):
            E.append(Edge(a.cid, b.cid, "revises", 0.85))
    # task dependency between action items / decisions
    if fb.status == "assigned" and fa.status in ("assigned", "decided", "agreed") and ov >= 0.15 and \
            (DEPEND_CUE.search(b.text) or fa.status == "assigned"):
        E.append(Edge(a.cid, b.cid, "task_depends", 0.7))
    # conflicts: same object, incompatible value/polarity/status, and NOT a revision
    if ov >= 0.35 and not any(e.etype == "revises" for e in E):
        qa, qb = set(quantities(fa.quantity or "")), set(quantities(fb.quantity or ""))
        clash_q = bool(qa) and bool(qb) and not (qa & qb)
        clash_p = fa.polarity != fb.polarity and fa.act == fb.act
        clash_s = {fa.status, fb.status} == {"decided", "rejected"}
        if clash_q or clash_p or clash_s:
            E.append(Edge(a.cid, b.cid, "conflicts", 0.8))
    return E


class EdgeClassifier:
    """Optional DeBERTa-v3-base pair classifier (train/train_edges.py). Input:
    "[A] <claim a> [SEP] [B] <claim b>" with frame slots appended."""
    LABELS = ["none", "before", "causes", "task_depends", "coref", "revises", "conflicts"]

    def __init__(self, path: str):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from .utils import device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.mdl = AutoModelForSequenceClassification.from_pretrained(path).to(device()).eval()
        self.torch, self.dev = torch, device()

    @staticmethod
    def fmt(c: Claim) -> str:
        f = c.frame
        return f"{c.text} | agent={f.agent} status={f.status} time={f.time} qty={f.quantity}"

    def predict(self, pairs: list[tuple[Claim, Claim]], bs=32):
        out = []
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            enc = self.tok([self.fmt(a) for a, _ in chunk], [self.fmt(b) for _, b in chunk], truncation=True,
                           max_length=160, padding=True, return_tensors="pt").to(self.dev)
            with self.torch.no_grad():
                p = self.torch.softmax(self.mdl(**enc).logits, -1).cpu().numpy()
            out.extend(p)
        return out


def build_cdg(claims: list[Claim], window: int = 3, classifier_path: str | None = None,
              min_conf: float = 0.5, mode: str = "typed") -> list[Edge]:
    """mode: typed (C-PROVE) | similarity (ablation 3: untyped kNN graph) | none (graph off)."""
    if mode == "none":
        return []
    pairs = list(candidate_pairs(claims, window))
    if mode == "similarity":
        E = []
        for a, b in pairs:
            s = jaccard(a.text, b.text)
            if s >= 0.2:
                E.append(Edge(a.cid, b.cid, "similar", s))
        return E
    edges: dict[tuple, Edge] = {}
    for a, b in pairs:
        for e in rule_edges(a, b):
            edges[e.key()] = e
    if classifier_path and Path(classifier_path).exists() and pairs:
        clf = EdgeClassifier(classifier_path)
        for (a, b), p in zip(pairs, clf.predict(pairs)):
            k = int(p.argmax())
            lab = EdgeClassifier.LABELS[k]
            if lab != "none" and p[k] >= min_conf:
                e = Edge(a.cid, b.cid, lab, float(p[k]), source="classifier")
                edges.setdefault(e.key(), e)
    return list(edges.values())


def to_networkx(claims: list[Claim], edges: list[Edge]) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    for c in claims:
        G.add_node(c.cid, text=c.text, status=c.frame.status, ctype=c.claim_type)
    for e in edges:
        G.add_edge(e.src, e.dst, key=e.etype, etype=e.etype, weight=e.weight)
    return G


def in_degree(claims: list[Claim], edges: list[Edge], types=("task_depends", "causes", "before", "revises")) -> dict[str, int]:
    d = {c.cid: 0 for c in claims}
    for e in edges:
        if e.etype in types and e.dst in d:
            d[e.dst] += 1
        if e.etype in ("task_depends", "causes") and e.src in d:
            d[e.src] += 1   # claims others depend on are more important
    return d
