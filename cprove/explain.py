"""Proof traces: the top-k ground rules that determine each claim's verdict.

For claim i we score every grounding touching one of i's variables by its
'pull' = lambda_r * m_g * max(0, l_g(y*))^2 (energy still paid at the optimum —
i.e. the rules that are actively pushing y* away from where they would like it),
plus the observation atoms that set the dimension values. Each grounding is
rendered with a fixed template so explanations are faithful to the computation.
"""
from __future__ import annotations

import torch

from .rules import GroundProgram

TEMPLATES = {
    "ent->fact": "Transcript entails the claim (p_ent={o:.2f}).",
    "con->-fact": "Transcript contradicts the claim (p_con={o:.2f}).",
    "obs>=": "{dim} evidence: {strength} ({o:.2f}).",
    "obs<=": "{dim} evidence: {strength} ({o:.2f}).",
    "order": "Order conflict: summary says {src} happens before {dst}, transcript order disagrees.",
    "rev-rescue": "{dst} correctly reports that {src} was later revised.",
    "rev-final": "{src} is presented as final, but {dst} revises it.",
    "dep": "{dst} depends on {src}, which is doubtful.",
    "excl": "{src} and {dst} conflict; both cannot be true.",
    "agent": "Shares the agent with a mis-attributed claim ({src}/{dst}).",
    "coref": "Duplicate of another claim ({src}/{dst}); verdicts must agree.",
    "sim": "Similar to another claim ({src}/{dst}).",
    "mean->y": "Average of the dimensions supports faithfulness.",
    "prior-y": "Prior: claims are unverified by default.",
    "prior-rsn": "Prior: no graph inconsistency.",
    "judge>=": "LLM judge: faithful with p={o:.2f}.",
    "judge<=": "LLM judge: faithful with p={o:.2f}.",
}
RULE_DIM = {"R1_evidence": "Factual", "R2_attribution": "Attribution", "R3_temporal": "Temporal",
            "R4_revision": "Contextual (latest state)", "R5_status": "Semantic (status)", "RJ_judge": "LLM judge"}


def proof_trace(prog: GroundProgram, comp, y: torch.Tensor, lam: torch.Tensor, cid: str, k: int = 3):
    vars_i = {v for (c, d), v in prog.var_index.items() if c == cid}
    lin = comp.lin(y).detach()
    rows = []
    for g, gr in enumerate(prog.groundings):
        if not any(v in vars_i for v, _ in gr.terms):
            continue
        kind = gr.meta.get("kind", "")
        if kind.startswith("y->"):
            continue
        h = max(0.0, float(lin[g]))
        lam_r = float(lam[comp.rule[g]]) * gr.mult
        pull = lam_r * h * h
        o = gr.meta.get("o", 0.0)
        # observation atoms that are satisfied still explain the value: credit lam * o
        weight = pull if pull > 1e-6 else (0.1 * lam_r * (o if kind in ("obs>=", "ent->fact", "judge>=") else 0.0))
        rows.append((weight, gr))
    rows.sort(key=lambda r: -r[0])
    out, seen = [], set()
    for w, gr in rows:
        kind = gr.meta.get("kind", "")
        o = gr.meta.get("o", 0.0)
        strength = "supports" if o >= 0.66 else "doubtful" if o >= 0.4 else "contradicts / missing"
        txt = TEMPLATES.get(kind, kind).format(o=o, strength=strength, dim=RULE_DIM.get(gr.rule, gr.rule),
                                               src=gr.meta.get("src", ""), dst=gr.meta.get("dst", ""))
        if txt in seen:
            continue
        seen.add(txt)
        out.append({"rule": gr.rule, "kind": kind, "score": round(w, 4), "text": txt})
        if len(out) >= k:
            break
    return out
