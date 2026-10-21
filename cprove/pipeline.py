"""End-to-end C-PROVE pipeline (Algorithm 1).

    claims  = Frames(summary)                         # Stage 1
    edges   = CDG(claims)                             # Stage 2
    evid    = Retrieve(meeting, claims)               # Stage 3
    nli     = NLI(evid, claims)
    obs     = Dimensions(meeting, claims, evid, nli)  # Stage 4
    prog    = Ground(claims, obs, edges)              # Stage 5a
    y*      = HLMRF.solve(prog)                       # Stage 5b
    y*      = AdaptiveVerify(prog, judge, VoI)        # Stage 5c
    labels  = CRC(y*) ; MTI, interval, components ; proof traces   # Stage 6
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .schema import Claim, Meeting, Edge
from .utils import load_config, SLOT
from .llm import make_llm
from .frames import FrameExtractor
from .graph import build_cdg
from .evidence import EvidenceRetriever
from .nli import make_nli, score_claims
from .dimensions import DimensionScorer, StatusModel, Calibrators
from .rules import ground, order_consistency
from .hlmrf import HLMRF, Weights
from .adaptive import adaptive_verify
from .judge import LLMJudge
from .conformal import CRCLabeler
from .mti import mti, components, ConformalInterval
from .explain import proof_trace


@dataclass
class Ablation:
    frames_off: bool = False                 # 1: one claim per sentence, no slots
    graph_mode: str = "typed"                # typed | similarity (3) | none (2)
    drop_dims: tuple = ()                    # 6: e.g. ("attr",)
    disabled_rules: tuple = ()
    gold_edges: Optional[dict] = None        # 7: summary_id -> list[Edge]
    hand_weights: bool = False               # 5: use init weights
    scheduler: Optional[str] = None          # RQ3 override


@dataclass
class Result:
    summary_id: str
    claims: list
    edges: list
    obs: dict
    post: dict
    labels: dict
    mti: float
    mti_interval: tuple
    components: dict
    proofs: dict
    trace: object = None
    program: object = None
    extras: dict = field(default_factory=dict)


class CProve:
    def __init__(self, cfg: dict | None = None, ablation: Ablation | None = None, judge_llm=None):
        self.cfg = cfg or load_config()
        self.ab = ablation or Ablation()
        c = self.cfg
        self.llm = make_llm(c["llm"])
        self.judge_llm = judge_llm or self.llm
        self.frames = FrameExtractor(self.llm, c["frames"]["max_claims_per_sentence"], c["frames"]["use_llm"])
        r = c["retrieval"]
        self.retriever = EvidenceRetriever(r["encoder"], r["dense_weight"], r["sparse_weight"], r["top_k"],
                                           r["context_window"], r["slot_boost"])
        self.nli = make_nli(c["nli"])
        d = c["dimensions"]
        self.dims = DimensionScorer(StatusModel(d.get("status_classifier")),
                                    Calibrators.load(d.get("calibrators"), d.get("calibration", "isotonic")),
                                    self.retriever)
        h = c["hlmrf"]
        init = h["init_weights"]
        w = Weights(init) if self.ab.hand_weights else Weights.load(h.get("learned_weights"), init)
        self.model = HLMRF(w, h["iters"], h["tol"])
        self.labeler = CRCLabeler.load(str(self._out("crc.json")), c["labels"]["risk_alpha"])
        self.mti_ci = ConformalInterval(c["mti"]["conformal_alpha"])
        q = self._out("mti_conformal.json")
        if q.exists():
            import json
            self.mti_ci.q = json.loads(q.read_text())["q"]

    def _out(self, name):
        from pathlib import Path
        return Path(self.cfg["paths"]["models"]) / name

    # ----------------------------------------------------------------- stages
    def decompose(self, m: Meeting, summary: str, summary_id: str) -> list[Claim]:
        parts = m.meta.get("participants") or m.speakers
        return self.frames.decompose_summary(summary, summary_id, parts, m.meta.get("date"), self.ab.frames_off)

    def edges_for(self, claims, summary_id) -> list[Edge]:
        if self.ab.gold_edges is not None:
            return self.ab.gold_edges.get(summary_id, [])
        g = self.cfg["graph"]
        return build_cdg(claims, g["window_sents"], g.get("edge_classifier"), g["min_edge_conf"], self.ab.graph_mode)

    def observe(self, m: Meeting, claims: list[Claim]):
        evid = self.retriever.retrieve(m, claims)
        nli = score_claims(self.nli, m, claims, evid, self.cfg["retrieval"]["context_window"])
        obs = {c.cid: self.dims.observe(m, c, evid[c.cid], nli[c.cid], m.meta.get("date"), self.ab.drop_dims) for c in claims}
        grounding = {cid: o.raw.get("grounding", []) for cid, o in obs.items()}
        return evid, nli, obs, grounding

    def program(self, claims, obs, edges, grounding):
        ok = order_consistency(claims, edges, grounding)
        return ground(claims, obs, edges, ok, self.ab.disabled_rules), ok

    # ----------------------------------------------------------------- full run
    def verify(self, m: Meeting, summary: str, summary_id: str = "S0", claims: Optional[list[Claim]] = None,
               judge=None, scheduler: Optional[str] = None, explain_k: int = 3) -> Result:
        claims = claims if claims is not None else self.decompose(m, summary, summary_id)
        edges = self.edges_for(claims, summary_id)
        evid, nli, obs, grounding = self.observe(m, claims)
        prog, ok = self.program(claims, obs, edges, grounding)
        sch = scheduler or self.ab.scheduler or self.cfg["adaptive"]["scheduler"]
        if judge is None and self.judge_llm.available and sch != "none":
            judge = LLMJudge(self.judge_llm, claims, evid, self._neighbour_fn(claims, edges))
        a = self.cfg["adaptive"]
        y, trace = adaptive_verify(self.model, prog, judge, sch, a["budget_frac"], a["epsilon"])
        post = self.model.posteriors(prog, y)
        from .dimensions import unseen_numbers
        veto = {}
        for c in claims:                              # hard vetoes: absent figure / strong transcript contradiction
            if unseen_numbers(m, c):
                veto[c.cid] = "A number in this claim does not appear anywhere in the transcript."
            elif obs[c.cid].con >= 0.9:
                veto[c.cid] = f"Transcript contradicts the claim (p_con={obs[c.cid].con:.2f})."
            if c.cid in veto:
                post[c.cid]["y"] = min(post[c.cid]["y"], 0.25)
        labels = dict(zip([c.cid for c in claims], self.labeler([post[c.cid]["y"] for c in claims])))
        for cid in veto:
            labels[cid] = "UNSUPPORTED"
        omega = self.cfg["mti"]["omega"]
        score = mti(claims, edges, post, omega)
        comp = components(m, claims, edges, post, obs, grounding, ok, omega)
        from .rules import RULES  # noqa
        from .hlmrf import Compiled
        compd = Compiled.from_program(prog)
        lam = self.model.w().detach()
        proofs = {}
        for c in claims:
            tr = proof_trace(prog, compd, y, lam, c.cid, explain_k + 3)
            tr = [t for t in tr if not t["text"].startswith(("Average of the dimensions", "Prior:"))]
            if c.cid in veto:
                tr = [t for t in tr if "supports" not in t["text"] and "entails" not in t["text"] and not t["text"].startswith("Transcript contradicts")]
                tr.insert(0, {"rule": "veto", "kind": "veto", "score": 1.0, "text": veto[c.cid]})
            proofs[c.cid] = tr[:explain_k]
        return Result(summary_id, claims, edges, obs, post, labels, score, self.mti_ci(score), comp, proofs,
                      trace, prog, {"evidence": evid, "nli": nli, "order_ok": ok})

    @staticmethod
    def _neighbour_fn(claims, edges):
        txt = {c.cid: c.text for c in claims}

        def f(cid):
            rows = [f"- ({e.etype}) {txt[e.dst] if e.src == cid else txt[e.src]}" for e in edges
                    if cid in (e.src, e.dst) and e.etype != "same_agent"]
            return "\n".join(rows[:6]) or "(none)"
        return f

    def release(self):
        SLOT.release()
