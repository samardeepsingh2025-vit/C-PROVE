r"""Stage 5b — Hinge-loss MRF inference, weight learning, and sensitivities.

Energy (convex in y for fixed lambda):
    E(y; lambda, o) = sum_g  lambda_{r(g)} * m_g * max(0, a_g . y + b_g)^2 ,   y in [0,1]^n

Inference: majorise-minimise projected gradient with a Gershgorin diagonal
preconditioner  D_v = 2 * sum_g lambda m |a_gv| * ||a_g||_1, so each step
    y <- clip(y - grad / D, 0, 1)
monotonically decreases E (no learning-rate tuning; converges in ~50-200 steps).

Learning: lambda_r = softplus(theta_r). y_K(theta) is K unrolled solver steps
from a detached warm start; loss = BCE(y_K[y-vars], gold) + l2 * ||theta - theta_0||^2.
Every step is differentiable, so autograd also gives d y* / d o for the VoI scheduler.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .rules import RULES, RULE_ID, GroundProgram


def inv_softplus(x: float) -> float:
    return float(math.log(math.expm1(max(x, 1e-4))))


class Weights(torch.nn.Module):
    def __init__(self, init: dict[str, float]):
        super().__init__()
        theta0 = torch.tensor([inv_softplus(init.get(r, 1.0)) for r in RULES], dtype=torch.float32)
        self.theta = torch.nn.Parameter(theta0.clone())
        self.register_buffer("theta0", theta0)

    def forward(self):
        return F.softplus(self.theta)

    def as_dict(self):
        return {r: float(v) for r, v in zip(RULES, self().detach().cpu())}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))

    @staticmethod
    def load(path, init):
        w = dict(init)
        if path and Path(path).exists():
            w.update(json.loads(Path(path).read_text()))
        return Weights(w)


class Compiled:
    """Tensorised program (optionally several programs merged block-diagonally)."""

    def __init__(self, idx, coef, const, rule, mult, n):
        self.idx = torch.as_tensor(idx)
        self.coef = torch.as_tensor(coef)
        self.const = torch.as_tensor(const)
        self.rule = torch.as_tensor(rule)
        self.mult = torch.as_tensor(mult)
        self.n = n
        self.l1 = self.coef.abs().sum(1)

    @staticmethod
    def from_program(p: GroundProgram):
        return Compiled(*p.to_arrays(), p.n)

    @staticmethod
    def merge(programs: list[GroundProgram]):
        parts, off = [], 0
        offsets = []
        for p in programs:
            idx, coef, const, rule, mult = p.to_arrays()
            parts.append((idx + off * (coef != 0), coef, const, rule, mult))
            offsets.append(off)
            off += p.n
        cat = [np.concatenate([q[k] for q in parts]) for k in range(5)]
        return Compiled(*cat, off), offsets

    def lin(self, y, const=None):
        c = self.const if const is None else const
        return (self.coef * y[self.idx]).sum(1) + c

    def energy(self, y, lam, const=None):
        h = F.relu(self.lin(y, const))
        return (lam[self.rule] * self.mult * h * h).sum()

    def grad(self, y, lam, const=None):
        h = F.relu(self.lin(y, const))
        gw = 2.0 * lam[self.rule] * self.mult * h                     # (G,)
        g = torch.zeros(self.n, dtype=y.dtype)
        g = g.index_add(0, self.idx.reshape(-1), (gw[:, None] * self.coef).reshape(-1))
        return g

    def precond(self, lam):
        w = 2.0 * lam[self.rule] * self.mult * self.l1                 # (G,)
        D = torch.zeros(self.n)
        D = D.index_add(0, self.idx.reshape(-1), (w[:, None] * self.coef.abs()).reshape(-1))
        return D.clamp_min(1e-3)

    def solve(self, lam, y0=None, iters=200, tol=1e-5, const=None, differentiable=False):
        y = torch.full((self.n,), 0.5) if y0 is None else y0.clone()
        D = self.precond(lam)
        ctx = torch.enable_grad() if differentiable else torch.no_grad()
        with ctx:
            for t in range(iters):
                y_new = (y - self.grad(y, lam, const) / D).clamp(0.0, 1.0)
                if not differentiable and (y_new - y).abs().max() < tol:
                    y = y_new
                    break
                y = y_new
        return y


class HLMRF:
    def __init__(self, weights: Weights, iters=200, tol=1e-5):
        self.w, self.iters, self.tol = weights, iters, tol

    def infer(self, prog: GroundProgram, y0=None):
        comp = Compiled.from_program(prog)
        with torch.no_grad():
            lam = self.w()
        y = comp.solve(lam, y0, self.iters, self.tol)
        return y, comp

    def posteriors(self, prog: GroundProgram, y: torch.Tensor) -> dict[str, dict]:
        out = {}
        for (cid, d), v in prog.var_index.items():
            out.setdefault(cid, {})[d] = float(y[v])
        return out

    # ------------------------------------------------------------------ learning
    def fit(self, programs: list[GroundProgram], gold: list[dict[str, int]], epochs=30, lr=0.05, unroll=50,
            l2=1e-3, batch=16, val=None, verbose=True, freeze=("RJ_judge",)):
        opt = torch.optim.Adam(self.w.parameters(), lr=lr)
        freeze_mask = torch.tensor([r in freeze for r in RULES])
        hist = []
        order = np.arange(len(programs))
        rng = np.random.default_rng(0)
        for ep in range(epochs):
            rng.shuffle(order)
            tot = 0.0
            for b in range(0, len(order), batch):
                ids = order[b:b + batch]
                comp, offs = Compiled.merge([programs[i] for i in ids])
                tgt_idx, tgt = [], []
                for i, off in zip(ids, offs):
                    for cid, lab in gold[i].items():
                        if (cid, "y") in programs[i].var_index:
                            tgt_idx.append(off + programs[i].var_index[(cid, "y")])
                            tgt.append(float(lab))
                if not tgt_idx:
                    continue
                lam = self.w()
                with torch.no_grad():
                    y0 = comp.solve(lam.detach(), None, self.iters, self.tol)
                yK = comp.solve(lam, y0, unroll, differentiable=True)
                p = yK[torch.tensor(tgt_idx)].clamp(1e-4, 1 - 1e-4)
                loss = F.binary_cross_entropy(p, torch.tensor(tgt)) + l2 * ((self.w.theta - self.w.theta0) ** 2).sum()
                opt.zero_grad()
                loss.backward()
                self.w.theta.grad[freeze_mask] = 0.0
                opt.step()
                tot += loss.item() * len(ids)
            rec = {"epoch": ep, "train_loss": tot / max(1, len(order))}
            if val is not None:
                rec["val_loss"] = self.eval_loss(*val)
            hist.append(rec)
            if verbose:
                print(rec)
        return hist

    def eval_loss(self, programs, gold):
        losses = []
        for p, g in zip(programs, gold):
            y, _ = self.infer(p)
            for cid, lab in g.items():
                q = float(np.clip(y[p.y_index(cid)], 1e-4, 1 - 1e-4))
                losses.append(-(lab * math.log(q) + (1 - lab) * math.log(1 - q)))
        return float(np.mean(losses)) if losses else float("nan")

    # ------------------------------------------------------------------ sensitivities
    def judge_sensitivity(self, prog: GroundProgram, y_star: torch.Tensor, candidates: list[str], unroll=40):
        """J[j, i] = d y_j* / d o_i where o_i is a (hypothetical) judge observation on claim i,
        evaluated at o_i = y_i* so adding the atoms does not move the optimum.
        Returns (J as numpy [n_claims x n_cand], list of claim ids for rows)."""
        if not candidates:
            return np.zeros((0, 0)), []
        base = len(prog.groundings)
        for cid in candidates:
            prog.add_judge(cid, float(y_star[prog.y_index(cid)]))
        comp = Compiled.from_program(prog)
        del prog.groundings[base:]
        lam = self.w().detach()
        o0 = torch.tensor([float(y_star[prog.y_index(c)]) for c in candidates])
        rows = [c.cid for c in prog.claims]
        yrows = torch.tensor([prog.y_index(c) for c in rows])
        # judge groundings are pairs: (o - y) and (y - o) -> constants +o and -o
        jg = torch.arange(base, base + 2 * len(candidates))

        def f(o):
            const = comp.const.clone()
            const = const.index_put((jg[0::2],), o).index_put((jg[1::2],), -o)
            y = comp.solve(lam, y_star.detach(), unroll, const=const, differentiable=True)
            return y[yrows]

        J = torch.autograd.functional.jacobian(f, o0)
        return J.detach().numpy(), rows


def binary_entropy(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
