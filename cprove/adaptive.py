r"""Stage 5c — Adaptive verification by graph Value-of-Information.

    VoI(i) = H(y_i*) * (1 + sum_{j != i} |d y_j* / d o_i|)

H is the binary entropy of the current posterior; the second factor is how much
a judge observation on i would move the rest of the graph (autograd through the
unrolled HL-MRF solver). Loop:
    solve with cheap atoms -> pick argmax VoI -> call the LLM judge on that claim ->
    add judge atom -> warm-start re-solve -> repeat until budget B or max VoI < eps.

Schedulers for RQ3: none | all | random | entropy | voi.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from .hlmrf import HLMRF, binary_entropy
from .rules import GroundProgram


@dataclass
class AdaptiveTrace:
    order: list = field(default_factory=list)       # cids judged, in order
    voi: list = field(default_factory=list)         # VoI of the chosen claim at each step
    calls: int = 0
    y_history: list = field(default_factory=list)


def voi_scores(model: HLMRF, prog: GroundProgram, y: torch.Tensor, candidates: list[str]) -> dict[str, float]:
    if not candidates:
        return {}
    J, rows = model.judge_sensitivity(prog, y, candidates)
    rix = {c: k for k, c in enumerate(rows)}
    out = {}
    for k, cid in enumerate(candidates):
        h = float(binary_entropy(float(y[prog.y_index(cid)])))
        col = np.abs(J[:, k]).copy()
        col[rix[cid]] = 0.0
        out[cid] = h * (1.0 + float(col.sum()))
    return out


def adaptive_verify(model: HLMRF, prog: GroundProgram, judge: Optional[Callable[[str], Optional[float]]],
                    scheduler: str = "voi", budget_frac: float = 0.25, eps: float = 0.02, seed: int = 0,
                    max_calls: Optional[int] = None):
    """judge(cid) -> faithful probability from the LLM (or None if unavailable).
    Mutates prog (adds judge atoms). Returns final y and trace."""
    y, _ = model.infer(prog)
    tr = AdaptiveTrace(y_history=[y.clone()])
    cids = [c.cid for c in prog.claims]
    B = len(cids) if scheduler == "all" else int(np.ceil(budget_frac * len(cids)))
    if max_calls is not None:
        B = min(B, max_calls)
    if scheduler == "none" or judge is None or B == 0:
        return y, tr
    rng = random.Random(seed)
    done: set = set()
    pool = list(cids)
    if scheduler == "random":
        rng.shuffle(pool)
    while tr.calls < B:
        cand = [c for c in pool if c not in done]
        if not cand:
            break
        if scheduler in ("all", "random"):
            pick, score = cand[0], float("nan")
        elif scheduler == "entropy":
            ent = {c: float(binary_entropy(float(y[prog.y_index(c)]))) for c in cand}
            pick = max(ent, key=ent.get)
            score = ent[pick]
            if score < eps:
                break
        else:   # voi
            v = voi_scores(model, prog, y, cand)
            pick = max(v, key=v.get)
            score = v[pick]
            if score < eps:
                break
        done.add(pick)
        o = judge(pick)
        tr.calls += 1
        tr.order.append(pick)
        tr.voi.append(score)
        if o is None:
            continue
        prog.add_judge(pick, float(o))          # never mutate prog.obs: it is shared with cached items
        tr.judged = getattr(tr, "judged", {})
        tr.judged[pick] = float(o)
        y, _ = model.infer(prog, y0=y)          # warm start
        tr.y_history.append(y.clone())
    return y, tr
