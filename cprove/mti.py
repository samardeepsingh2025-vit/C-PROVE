r"""Stage 6 — Meeting Trust Index (MTI) with a conformal interval.

Headline:   MTI = sum_i w_i y_i* / sum_i w_i ,   w_i = omega(type_i) * (1 + deg_dep(i))
            omega: decision = action_item = 2, other = 1; deg_dep = dependency/causal/temporal degree in the CDG.
Diagnostics (the five original MTI components, each in [0,1], higher = better):
    coverage                 salient transcript moments (decisions / assignments) grounded by some claim
    contradiction_density    1 - mean_i con_i * (1 - y_i)
    attribution              mean attr posterior over claims where attribution applies
    temporal                 mean temp posterior x fraction of 'before' edges consistent with the transcript
    support_quality          weighted mean fact posterior
Optional "MTI-5" = learned convex combination of the five components fit on human ratings
(kept for continuity with the original proposal; reported alongside the headline MTI).
Split conformal: |MTI - target| residuals on a calibration split give an interval with
>= 1 - alpha marginal coverage.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import Claim, Edge, Meeting
from .text import status_cues
from .graph import in_degree

COMPONENTS = ["coverage", "contradiction_density", "attribution", "temporal", "support_quality"]


def claim_weights(claims: list[Claim], edges: list[Edge], omega: dict) -> dict[str, float]:
    deg = in_degree(claims, edges)
    return {c.cid: float(omega.get(c.claim_type, 1.0)) * (1.0 + deg[c.cid]) for c in claims}


def mti(claims, edges, post: dict[str, dict], omega: dict) -> float:
    w = claim_weights(claims, edges, omega)
    num = sum(w[c.cid] * post[c.cid]["y"] for c in claims)
    den = sum(w.values())
    return float(num / den) if den else float("nan")


def salient_utterances(m: Meeting) -> list[int]:
    out = []
    for j, u in enumerate(m.utterances):
        if len(u.text.split()) < 4:
            continue
        c = status_cues(u.text)
        if c["decided"] or c["assigned"] or (c["agreed"] and len(u.text.split()) > 6):
            out.append(j)
    return out


def components(m: Meeting, claims, edges, post, obs, grounding: dict[str, list[int]], order_ok: dict, omega: dict) -> dict:
    w = claim_weights(claims, edges, omega)
    sal = salient_utterances(m)
    covered = set()
    for g in grounding.values():
        for j in g:
            covered.update(range(j - 2, j + 3))
    cov = float(np.mean([j in covered for j in sal])) if sal else float("nan")
    cd = 1.0 - float(np.mean([obs[c.cid].con * (1 - post[c.cid]["y"]) for c in claims])) if claims else float("nan")
    att = [post[c.cid]["attr"] for c in claims if "attr" in post[c.cid]]
    tmp = [post[c.cid]["temp"] for c in claims if "temp" in post[c.cid]]
    oks = list(order_ok.values())
    temporal = (float(np.mean(tmp)) if tmp else 1.0) * (float(np.mean(oks)) if oks else 1.0)
    sq_num = sum(w[c.cid] * post[c.cid]["fact"] for c in claims)
    return {
        "coverage": cov,
        "contradiction_density": cd,
        "attribution": float(np.mean(att)) if att else float("nan"),
        "temporal": temporal,
        "support_quality": float(sq_num / sum(w.values())) if w else float("nan"),
    }


class MTI5:
    """Learned convex combination of the five components (fit by NNLS on human ratings in [0,1])."""

    def __init__(self, weights=None):
        self.w = weights or {k: 1 / len(COMPONENTS) for k in COMPONENTS}

    def fit(self, comps: list[dict], target: list[float]):
        from scipy.optimize import nnls
        X = np.array([[0.5 if np.isnan(c[k]) else c[k] for k in COMPONENTS] for c in comps])
        w, _ = nnls(X, np.asarray(target, float))
        w = w / w.sum() if w.sum() > 0 else np.ones(len(COMPONENTS)) / len(COMPONENTS)
        self.w = dict(zip(COMPONENTS, map(float, w)))
        return self

    def __call__(self, comp: dict) -> float:
        return float(sum(self.w[k] * (0.5 if np.isnan(comp[k]) else comp[k]) for k in COMPONENTS))

    def save(self, path):
        Path(path).write_text(json.dumps(self.w, indent=2))


class ConformalInterval:
    def __init__(self, alpha=0.10):
        self.alpha, self.q = alpha, None

    def fit(self, pred, target):
        r = np.abs(np.asarray(pred) - np.asarray(target))
        n = len(r)
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        self.q = float(np.sort(r)[min(k, n) - 1]) if n else 0.5
        return self

    def __call__(self, v: float):
        q = 0.5 if self.q is None else self.q
        return max(0.0, v - q), min(1.0, v + q)
