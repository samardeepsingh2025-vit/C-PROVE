"""LLM judge used by the adaptive verifier (expensive atom), plus a simulated
judge for budget-curve experiments without a GPU (clearly labelled as simulation)."""
from __future__ import annotations

import random

from .prompts import JUDGE_SCHEMA, JUDGE_SYSTEM, JUDGE_USER


class LLMJudge:
    def __init__(self, llm, claims, evidence, neighbours_fn=None):
        self.llm = llm
        self.claims = {c.cid: c for c in claims}
        self.ev = evidence
        self.nb = neighbours_fn
        self.log = {}

    def __call__(self, cid: str):
        if not self.llm.available:
            return None
        c = self.claims[cid]
        f = c.frame
        nb = self.nb(cid) if self.nb else "(none)"
        user = JUDGE_USER.format(claim=c.text, agent=f.agent, act=f.act, obj=f.obj, time=f.time, status=f.status,
                                 polarity=f.polarity, quantity=f.quantity, recipient=f.recipient,
                                 evidence=self.ev[cid].text, neighbours=nb)
        out = self.llm.json(JUDGE_SYSTEM, user, JUDGE_SCHEMA, max_tokens=400)
        if isinstance(out, list):
            out = next((o for o in out if isinstance(o, dict)), None)
        if not isinstance(out, dict):
            return None
        try:
            p = float(min(1.0, max(0.0, float(out.get("faithful_prob", 0.5)))))
        except (TypeError, ValueError):
            p = 0.5
        v = out.get("verdict")
        if v == "SUPPORTED":
            p = max(p, 0.6)
        elif v == "UNSUPPORTED":
            p = min(p, 0.25)
        else:
            p = min(p, 0.45)
        self.log[cid] = out
        return p


class SimulatedJudge:
    """Returns gold label flipped with prob `noise`, as a calibrated-ish probability.
    Only for RQ3 budget curves when the real judge cache is not available."""

    def __init__(self, gold: dict[str, int], noise=0.15, seed=0):
        self.gold, self.noise, self.rng = gold, noise, random.Random(seed)

    def __call__(self, cid):
        if cid not in self.gold:
            return None
        g = self.gold[cid]
        if self.rng.random() < self.noise:
            g = 1 - g
        return 0.85 if g else 0.15