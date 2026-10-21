r"""Stage 5a — Grounding of the C-PROVE rule set into hinge-loss potentials.

Every grounding g is a linear function  l_g(y) = sum_k a_gk * y[v_gk] + b_g  over at most
4 latent variables, and contributes  lambda_r * m_g * max(0, l_g(y))^2  to the energy
(m_g = edge confidence or 1). Łukasiewicz relaxation:
    A -> B          distance  max(0, A - B)
    A & B -> C      distance  max(0, A + B - 1 - C)
    A & B -> not C  distance  max(0, A + B + C - 2)

Latent variables per claim i:
    y_i                faithful
    fact_i, rsn_i      always present
    attr_i, temp_i, ctx_i, sem_i   only if the dimension applies (observation exists)

Rule families (lambda learned per family):
 R1   evidence         ent_i -> fact_i ;   fact_obs <-> fact_i
 R1b  contradiction    con_i -> not fact_i
 R2   attribution      attr_obs <-> attr_i
 R3   temporal         temp_obs <-> temp_i ;  Before_ij & not OrderOK_ij -> not rsn_i , not rsn_j
 R4   revision         ctx_obs <-> ctx_i ;  Revises_ij & y_j -> ctx_i ;  Revises_ij & Final_i & y_j -> not rsn_i
 R5   status           sem_obs <-> sem_i
 R6   dependency       (TaskDep_ij | Causes_ij) & not y_i -> not rsn_j
 R7   exclusion        Conflicts_ij -> not (y_i & y_j)
 R8   agent/coref      SameAgent_ij & not attr_i -> not attr_j (both directions) ;  Coref_ij -> (y_i <-> y_j)
 R9a  aggregate-upper  y_i -> d_i   for every applicable d (incl. rsn)
 R9b  aggregate-lower  mean_d(d_i) -> y_i
 R10  prior            not y_i
 R10b rsn prior        rsn_i
 RJ   judge            judge_obs_i <-> y_i     (added by the adaptive verifier)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schema import Claim, Edge, Observation

RULES = ["R1_evidence", "R1b_contradiction", "R2_attribution", "R3_temporal", "R4_revision", "R5_status",
         "R6_dependency", "R7_exclusion", "R8_agent", "R9a_upper", "R9b_lower", "R10_prior", "R10b_rsn_prior",
         "RJ_judge"]
RULE_ID = {r: i for i, r in enumerate(RULES)}
OPT_DIMS = ("attr", "temp", "ctx", "sem")
MAXT = 6   # max terms per grounding (R9b uses up to 6 dims + y)


@dataclass
class Grounding:
    rule: str
    terms: list            # [(var_index, coef)]
    const: float
    mult: float = 1.0
    meta: dict = field(default_factory=dict)


@dataclass
class GroundProgram:
    var_index: dict                      # (cid, dim) -> int
    groundings: list
    claims: list
    obs: dict                            # cid -> Observation

    @property
    def n(self):
        return len(self.var_index)

    def y_index(self, cid):
        return self.var_index[(cid, "y")]

    def to_arrays(self):
        G = len(self.groundings)
        idx = np.zeros((G, MAXT + 1), dtype=np.int64)
        coef = np.zeros((G, MAXT + 1), dtype=np.float32)
        const = np.zeros(G, dtype=np.float32)
        rule = np.zeros(G, dtype=np.int64)
        mult = np.zeros(G, dtype=np.float32)
        for g, gr in enumerate(self.groundings):
            for t, (v, a) in enumerate(gr.terms):
                idx[g, t], coef[g, t] = v, a
            const[g], rule[g], mult[g] = gr.const, RULE_ID[gr.rule], gr.mult
        return idx, coef, const, rule, mult

    def add_judge(self, cid: str, o: float):
        """Adds the LLM-judge atom for claim cid (returns the index of the 2 new groundings)."""
        v = self.y_index(cid)
        self.groundings.append(Grounding("RJ_judge", [(v, -1.0)], o, 1.0, {"cid": cid, "kind": "judge>=", "o": o}))
        self.groundings.append(Grounding("RJ_judge", [(v, 1.0)], -o, 1.0, {"cid": cid, "kind": "judge<=", "o": o}))


def _both(rule, v, o, meta):
    """o <-> v : two hinges, together (v-o)^2 under squared potentials."""
    return [Grounding(rule, [(v, -1.0)], o, 1.0, dict(meta, kind="obs>=", o=o)),
            Grounding(rule, [(v, 1.0)], -o, 1.0, dict(meta, kind="obs<=", o=o))]


def ground(claims: list[Claim], obs: dict[str, Observation], edges: list[Edge],
           order_ok: dict[tuple, float] | None = None, disabled: tuple = ()) -> GroundProgram:
    order_ok = order_ok or {}
    V: dict = {}

    def var(cid, d):
        if (cid, d) not in V:
            V[(cid, d)] = len(V)
        return V[(cid, d)]

    Gs: list[Grounding] = []
    final_status = {c.cid: float(c.frame.status in ("decided", "agreed")) for c in claims}
    for c in claims:
        o = obs[c.cid]
        y, fact, rsn = var(c.cid, "y"), var(c.cid, "fact"), var(c.cid, "rsn")
        m = {"cid": c.cid}
        # R1 / R1b evidence
        Gs.append(Grounding("R1_evidence", [(fact, -1.0)], o.ent, 1.0, dict(m, kind="ent->fact", o=o.ent)))
        if o.fact is not None:
            Gs += _both("R1_evidence", fact, o.fact, m)
        Gs.append(Grounding("R1b_contradiction", [(fact, 1.0)], o.con - 1.0, 1.0, dict(m, kind="con->-fact", o=o.con)))
        # R2 / R3 / R4 / R5 local observations
        for d, r in (("attr", "R2_attribution"), ("temp", "R3_temporal"), ("ctx", "R4_revision"), ("sem", "R5_status")):
            val = getattr(o, d)
            if val is not None:
                Gs += _both(r, var(c.cid, d), float(val), m)
        if o.llm is not None:
            Gs += _both("RJ_judge", y, float(o.llm), m)
        # R9 aggregation over applicable dims
        dims = ["fact", "rsn"] + [d for d in OPT_DIMS if (c.cid, d) in V]
        for d in dims:
            Gs.append(Grounding("R9a_upper", [(y, 1.0), (V[(c.cid, d)], -1.0)], 0.0, 1.0, dict(m, kind="y->" + d)))
        K = len(dims)
        Gs.append(Grounding("R9b_lower", [(V[(c.cid, d)], 1.0 / K) for d in dims] + [(y, -1.0)], 0.0, 1.0,
                            dict(m, kind="mean->y")))
        # priors
        Gs.append(Grounding("R10_prior", [(y, 1.0)], 0.0, 1.0, dict(m, kind="prior-y")))
        Gs.append(Grounding("R10b_rsn_prior", [(rsn, -1.0)], 1.0, 1.0, dict(m, kind="prior-rsn")))

    # graph rules
    for e in edges:
        if e.src not in obs or e.dst not in obs:
            continue
        i, j, w = e.src, e.dst, float(e.weight)
        mi = {"src": i, "dst": j, "etype": e.etype}
        if e.etype == "before":
            ok = order_ok.get((i, j), 1.0)
            # w & not ok -> not rsn ;  distance max(0, w - ok + rsn - 1)
            for k in (i, j):
                Gs.append(Grounding("R3_temporal", [(V[(k, "rsn")], 1.0)], w - ok - 1.0, 0.5, dict(mi, kind="order", ok=ok, on=k)))
        elif e.etype == "revises":
            # Revises & y_j -> ctx_i : max(0, w + y_j - 1 - ctx_i)
            if (i, "ctx") in V:
                Gs.append(Grounding("R4_revision", [(V[(j, "y")], 1.0), (V[(i, "ctx")], -1.0)], w - 1.0, 1.0, dict(mi, kind="rev-rescue")))
            # Revises & Final_i & y_j -> not rsn_i : max(0, w + F_i + y_j + rsn_i - 3)
            if final_status[i] > 0:
                Gs.append(Grounding("R4_revision", [(V[(j, "y")], 1.0), (V[(i, "rsn")], 1.0)], w + final_status[i] - 3.0, 1.0,
                                    dict(mi, kind="rev-final")))
        elif e.etype in ("task_depends", "causes"):
            # dep & not y_i -> not rsn_j : max(0, w - y_i + rsn_j - 1)
            mult = 1.0 if e.etype == "task_depends" else 0.7
            Gs.append(Grounding("R6_dependency", [(V[(i, "y")], -1.0), (V[(j, "rsn")], 1.0)], w - 1.0, mult, dict(mi, kind="dep")))
        elif e.etype == "conflicts":
            Gs.append(Grounding("R7_exclusion", [(V[(i, "y")], 1.0), (V[(j, "y")], 1.0)], w - 2.0, 1.0, dict(mi, kind="excl")))
        elif e.etype == "same_agent":
            if (i, "attr") in V and (j, "attr") in V:
                Gs.append(Grounding("R8_agent", [(V[(i, "attr")], -1.0), (V[(j, "attr")], 1.0)], w - 1.0, 0.5, dict(mi, kind="agent")))
                Gs.append(Grounding("R8_agent", [(V[(j, "attr")], -1.0), (V[(i, "attr")], 1.0)], w - 1.0, 0.5, dict(mi, kind="agent")))
        elif e.etype == "coref":
            Gs.append(Grounding("R8_agent", [(V[(i, "y")], 1.0), (V[(j, "y")], -1.0)], w - 1.0, 1.0, dict(mi, kind="coref")))
            Gs.append(Grounding("R8_agent", [(V[(j, "y")], 1.0), (V[(i, "y")], -1.0)], w - 1.0, 1.0, dict(mi, kind="coref")))
        elif e.etype == "similar":   # ablation: untyped graph -> generic smoothing
            Gs.append(Grounding("R8_agent", [(V[(i, "y")], 1.0), (V[(j, "y")], -1.0)], w - 1.0, 1.0, dict(mi, kind="sim")))
            Gs.append(Grounding("R8_agent", [(V[(j, "y")], 1.0), (V[(i, "y")], -1.0)], w - 1.0, 1.0, dict(mi, kind="sim")))

    if disabled:
        Gs = [g for g in Gs if g.rule not in disabled]
    return GroundProgram(V, Gs, claims, obs)


def order_consistency(claims: list[Claim], edges: list[Edge], grounding: dict[str, list[int]], slack: int = 2):
    """OrderOK_ij for 'before' edges: transcript grounding of i precedes that of j."""
    ok = {}
    for e in edges:
        if e.etype != "before":
            continue
        gi, gj = grounding.get(e.src) or [], grounding.get(e.dst) or []
        if not gi or not gj:
            ok[(e.src, e.dst)] = 1.0
            continue
        ok[(e.src, e.dst)] = 1.0 if min(gi) <= max(gj) + slack else 0.0
    return ok
