"""All paper experiments (RQ1-RQ6 + ablations) from cached stage outputs.

  python -m experiments.run rq1  --bench data/meetverify/human_test.jsonl  --dev data/meetverify/human_dev.jsonl
  python -m experiments.run rq2  --bench data/meetverify/synth_test.jsonl
  python -m experiments.run ablate --bench data/meetverify/synth_test.jsonl
  python -m experiments.run rq3  --bench data/meetverify/human_test.jsonl [--simulate_judge]
  python -m experiments.run rq4  --bench data/meetverify/synth_test.jsonl [--ratings data/meetverify/ratings.jsonl]
  python -m experiments.run rq5  --bench data/meetverify/human_test.jsonl
  python -m experiments.run rq6  --bench data/meetverify/human_test.jsonl
  python -m experiments.run all  --bench ... --dev ...

RQ1  Does C-PROVE detect unfaithful claims better than strong verifiers?          (AUROC / AUPRC / BAcc, bootstrap)
RQ2  Which error types does each dimension / the graph catch?                       (per-type recall & AUROC)
RQ3  Does VoI-adaptive verification reach 'judge-everything' quality at a fraction of LLM calls?  (budget curves)
RQ4  Is MTI a valid summary-level trust score?                                      (tau/rho vs humans, dose-response, conformal coverage)
RQ5  Are proof traces faithful to the decision?                                     (deletion vs random-deletion test)
RQ6  Cost on an 8 GB laptop GPU, and transfer to TofuEval-MeetingBank               (latency, VRAM, LLM calls)
Outputs: outputs/tables/*.csv|.md, outputs/preds/*.jsonl, outputs/figures/*.pdf
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cprove.utils import load_config, seed_everything
from cprove.data import load_corpus
from cprove.pipeline import CProve, Ablation
from cprove.dimensions import Calibrators
from cprove.graph import build_cdg
from cprove.rules import ground, order_consistency
from cprove.hlmrf import HLMRF, Weights
from cprove.adaptive import adaptive_verify
from cprove.judge import LLMJudge, SimulatedJudge
from cprove.mti import mti, components, ConformalInterval
from cprove.explain import proof_trace
from experiments.common import load_records, build_items, programs, units, unit_scores, metrics, best_threshold, ece
from experiments import stats
from train.fit_cprove import recalibrate


class Ctx:
    def __init__(self, a):
        self.a = a
        self.cfg = load_config(a.config)
        seed_everything(self.cfg["seed"])
        self.out = Path(self.cfg["paths"]["outputs"])
        for d in ("preds", "tables", "figures"):
            (self.out / d).mkdir(parents=True, exist_ok=True)
        self.meetings = {m.meeting_id: m for m in load_corpus(self.cfg)[0]}
        self.cp = CProve(self.cfg, Ablation(hand_weights=True))
        self.cal = Calibrators.load(self.cfg["dimensions"]["calibrators"], self.cfg["dimensions"]["calibration"])
        self.cp.dims.cal = Calibrators("none")
        h = self.cfg["hlmrf"]
        self.w_learned = Weights.load(h["learned_weights"], h["init_weights"])
        self.w_hand = Weights(h["init_weights"])
        self.bench = Path(a.bench).stem

    def items(self, path=None, frames_off=False):
        path = path or self.a.bench
        tag = Path(path).stem + ("_noframes" if frames_off else "")
        cp = self.cp
        if frames_off:
            cp = CProve(self.cfg, Ablation(hand_weights=True, frames_off=True))
            cp.dims.cal = Calibrators("none")
        its = build_items(cp, self.meetings, load_records(path), str(Path(self.cfg["paths"]["cache"]) / f"items_{tag}_raw.pkl"))
        its = copy.deepcopy(its)
        recalibrate(its, self.cal)
        return its

    def judge_for(self, it, sim_seed=0):
        if self.a.simulate_judge:
            return SimulatedJudge(it.gold, self.a.judge_noise, seed=sim_seed)
        if self.cp.llm.available:
            m = self.meetings[it.meeting_id]
            return LLMJudge(self.cp.llm, it.claims, self.cp.retriever.retrieve(m, it.claims), self.cp._neighbour_fn(it.claims, it.edges))
        return None

    def save_rows(self, name, rows):
        with open(self.out / "preds" / f"{self.bench}__{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def table(self, name, df: pd.DataFrame):
        df.to_csv(self.out / "tables" / f"{self.bench}__{name}.csv", index=False)
        (self.out / "tables" / f"{self.bench}__{name}.md").write_text(df.to_markdown(index=False, floatfmt=".3f"))
        print(f"\n## {name}\n" + df.to_markdown(index=False, floatfmt=".3f"))


def run_cprove(ctx: Ctx, items, weights=None, scheduler=None, budget=None, disabled=(), edges_fn=None,
               drop_dims=(), name="cprove", seed=0, timing=None):
    model = HLMRF(weights or ctx.w_learned, ctx.cfg["hlmrf"]["iters"], ctx.cfg["hlmrf"]["tol"])
    sch = scheduler or ctx.cfg["adaptive"]["scheduler"]
    bud = ctx.cfg["adaptive"]["budget_frac"] if budget is None else budget
    rows, calls, posts = [], 0, {}
    for k, it in enumerate(items):
        t0 = time.perf_counter()
        edges = edges_fn(it) if edges_fn else it.edges
        obs = it.obs
        if drop_dims:
            obs = {cid: copy.copy(o) for cid, o in obs.items()}
            for o in obs.values():
                for d in drop_dims:
                    setattr(o, d, None)
        ok = order_consistency(it.claims, edges, it.grounding)
        prog = ground(it.claims, obs, edges, ok, disabled)
        judge = ctx.judge_for(it, seed + k) if sch != "none" else None
        y, tr = adaptive_verify(model, prog, judge, sch, bud, ctx.cfg["adaptive"]["epsilon"], seed=seed + k)
        post = model.posteriors(prog, y)
        posts[it.rec["summary_id"]] = (it, post, prog, y, edges, ok)
        calls += tr.calls
        for (uid, _, g, e), s in zip(units(it), unit_scores(it, post)):
            rows.append({"summary_id": it.rec["summary_id"], "meeting_id": it.meeting_id, "unit": uid, "score": float(s),
                         "gold": g, "error_type": e, "system": name, "llm_calls": tr.calls})
        if timing is not None:
            timing.append({"summary_id": it.rec["summary_id"], "solve_s": time.perf_counter() - t0, "claims": len(it.claims),
                           "llm_calls": tr.calls, "edges": len(edges)})
    return rows, calls / max(1, len(items)), posts, model


def summarise_rows(rows, thr=0.5):
    g = [r["gold"] for r in rows]
    s = [r["score"] for r in rows]
    m = metrics(g, s, thr)
    m["ece"] = ece(g, s)
    return m


def dev_threshold(ctx, system):
    if not ctx.a.dev:
        return 0.5
    f = ctx.out / "preds" / f"{Path(ctx.a.dev).stem}__{system}.jsonl"
    if not f.exists():
        return 0.5
    rows = [json.loads(l) for l in open(f)]
    return best_threshold([r["gold"] for r in rows], [r["score"] for r in rows])


# ------------------------------------------------------------------------------------------ RQ1
def rq1(ctx):
    if ctx.a.dev:   # dev predictions first (for thresholds), same code path
        dctx = copy.copy(ctx); dctx.bench = Path(ctx.a.dev).stem
        rows, _, _, _ = run_cprove(dctx, ctx.items(ctx.a.dev))
        dctx.save_rows("cprove", rows)
    items = ctx.items()
    rows, cps, _, _ = run_cprove(ctx, items)
    ctx.save_rows("cprove", rows)
    systems = {"cprove": rows}
    for f in sorted((ctx.out / "preds").glob(f"{ctx.bench}__*.jsonl")):
        n = f.stem.split("__", 1)[1]
        if n not in systems and not n.startswith(("abl_", "rq3_")):
            systems[n] = [json.loads(l) for l in open(f)]
    res, pv = [], {}
    for n, r in systems.items():
        thr = dev_threshold(ctx, n)
        m = summarise_rows(r, thr)
        m.update(system=n, thr=thr, llm_calls_per_summary=cps if n == "cprove" else (np.nan if n != "llm_judge" else len(r) / max(1, len({x['summary_id'] for x in r}))))
        if n != "cprove":
            bs = stats.paired_bootstrap(rows, r, "auroc", ctx.a.bootstrap)
            m.update(d_auroc=bs["diff"], ci=f"[{bs['ci_low']:.3f},{bs['ci_high']:.3f}]", p_raw=bs["p"])
            pv[n] = bs["p"]
        res.append(m)
    adj = stats.holm(pv)
    for m in res:
        m["p_holm"] = adj.get(m["system"], np.nan)
    df = pd.DataFrame(res)[["system", "auroc", "auprc_err", "bacc", "f1_err", "ece", "thr", "llm_calls_per_summary",
                            *(c for c in ("d_auroc", "ci", "p_raw", "p_holm") if c in pd.DataFrame(res))]]
    ctx.table("rq1_main", df.sort_values("auroc", ascending=False))


# ------------------------------------------------------------------------------------------ RQ2
def rq2(ctx):
    files = sorted((ctx.out / "preds").glob(f"{ctx.bench}__*.jsonl"))
    if not any("cprove" in f.stem for f in files):
        rows, _, _, _ = run_cprove(ctx, ctx.items())
        ctx.save_rows("cprove", rows)
        files = sorted((ctx.out / "preds").glob(f"{ctx.bench}__*.jsonl"))
    res = []
    for f in files:
        n = f.stem.split("__", 1)[1]
        if n.startswith("rq3_"):
            continue
        rows = [json.loads(l) for l in open(f)]
        thr = dev_threshold(ctx, n)
        clean = [r for r in rows if r["gold"] == 1]
        for et in sorted({r["error_type"] for r in rows if r["gold"] == 0}):
            err = [r for r in rows if r["gold"] == 0 and r["error_type"] == et]
            sub = clean + err
            m = metrics([r["gold"] for r in sub], [r["score"] for r in sub], thr)
            res.append({"system": n, "error_type": et, "n_err": len(err), "auroc": m.get("auroc", np.nan),
                        "recall@thr": float(np.mean([r["score"] < thr for r in err]))})
    df = pd.DataFrame(res)
    ctx.table("rq2_per_error_long", df)
    ctx.table("rq2_per_error_auroc", df.pivot(index="system", columns="error_type", values="auroc").reset_index())


# ------------------------------------------------------------------------------------------ ablations
def heuristic_propagation(ctx, items, alpha=0.3):
    """Ablation 4: same calibrated dimensions, but min-aggregation + neighbour averaging instead of HL-MRF."""
    rows = []
    for it in items:
        base = {}
        for c in it.claims:
            o = it.obs[c.cid]
            ds = [v for v in (o.fact, o.attr, o.temp, o.ctx, o.sem) if v is not None]
            base[c.cid] = float(np.mean(ds)) if ds else 0.5
        nb = {c.cid: [] for c in it.claims}
        for e in it.edges:
            nb[e.src].append(e.dst); nb[e.dst].append(e.src)
        post = {cid: {"y": (1 - alpha) * v + alpha * (np.mean([base[j] for j in nb[cid]]) if nb[cid] else v)} for cid, v in base.items()}
        for (uid, _, g, e), s in zip(units(it), unit_scores(it, post)):
            rows.append({"summary_id": it.rec["summary_id"], "meeting_id": it.meeting_id, "unit": uid, "score": s, "gold": g, "error_type": e})
    return rows


def ablate(ctx):
    items = ctx.items()
    sch = "none" if ctx.a.no_judge_in_ablation else None
    full, _, _, _ = run_cprove(ctx, items, scheduler=sch)
    variants = {"full": full}
    variants["-frames (1)"] = run_cprove(ctx, ctx.items(frames_off=True), scheduler=sch)[0]
    variants["-graph (2)"] = run_cprove(ctx, items, scheduler=sch, edges_fn=lambda it: [])[0]
    variants["untyped graph (3)"] = run_cprove(ctx, items, scheduler=sch, edges_fn=lambda it: build_cdg(it.claims, mode="similarity"))[0]
    variants["heuristic propagation (4)"] = heuristic_propagation(ctx, items)
    variants["hand-set lambda (5)"] = run_cprove(ctx, items, weights=ctx.w_hand, scheduler=sch)[0]
    for d in ("fact", "attr", "temp", "ctx", "sem"):
        variants[f"-{d} (6)"] = run_cprove(ctx, items, scheduler=sch, drop_dims=(d,))[0]
    variants["-rsn/graph rules (6)"] = run_cprove(ctx, items, scheduler=sch,
                                                  disabled=("R3_temporal", "R4_revision", "R6_dependency", "R7_exclusion", "R8_agent"))[0]
    if ctx.a.gold_edges:
        ge = {}
        from cprove.schema import Edge
        for l in open(ctx.a.gold_edges):
            r = json.loads(l)
            ge.setdefault(r["summary_id"], []).append(Edge(r["src"], r["dst"], r["etype"], 1.0, "gold"))
        variants["gold edges (7)"] = run_cprove(ctx, items, scheduler=sch, edges_fn=lambda it: ge.get(it.rec["summary_id"], []))[0]
    res = []
    for n, r in variants.items():
        m = summarise_rows(r)
        if n != "full":
            bs = stats.paired_bootstrap(full, r, "auroc", ctx.a.bootstrap)
            m.update(delta_vs_full=-bs["diff"], p=bs["p"])
        m["variant"] = n
        res.append(m)
        ctx.save_rows("abl_" + n.split(" (")[0].replace(" ", "_").replace("/", "_"), [dict(x, system=n) for x in r])
    ctx.table("ablations", pd.DataFrame(res)[["variant", "auroc", "auprc_err", "bacc", "ece", *(c for c in ("delta_vs_full", "p") if c in pd.DataFrame(res))]])


# ------------------------------------------------------------------------------------------ RQ3
def rq3(ctx):
    items = ctx.items()
    res = []
    budgets = [0.0, 0.1, 0.25, 0.5, 1.0]
    for sch in ("random", "entropy", "voi"):
        for b in budgets:
            for seed in (ctx.cfg["eval"]["seeds"] if sch == "random" else [0]):
                rows, cps, _, _ = run_cprove(ctx, items, scheduler="none" if b == 0 else sch, budget=b, seed=seed)
                m = summarise_rows(rows)
                res.append({"scheduler": sch, "budget": b, "seed": seed, "calls_per_summary": cps, **m})
    rows, cps, _, _ = run_cprove(ctx, items, scheduler="all")
    res.append({"scheduler": "all", "budget": 1.0, "seed": 0, "calls_per_summary": cps, **summarise_rows(rows)})
    df = pd.DataFrame(res).groupby(["scheduler", "budget"], as_index=False).mean(numeric_only=True)
    ctx.table("rq3_budget", df[["scheduler", "budget", "calls_per_summary", "auroc", "auprc_err", "bacc", "ece"]])
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 3.2))
        for sch, g in df[df.scheduler != "all"].groupby("scheduler"):
            ax.plot(g.calls_per_summary, g.auroc, marker="o", label=sch)
        a = df[df.scheduler == "all"]
        ax.axhline(float(a.auroc.iloc[0]), ls="--", c="grey", label="judge all")
        ax.set_xlabel("LLM-judge calls per summary"); ax.set_ylabel("AUROC (unfaithful)")
        ax.legend(frameon=False); fig.tight_layout()
        fig.savefig(ctx.out / "figures" / f"{ctx.bench}__rq3_budget.pdf")
    except Exception as e:  # pragma: no cover
        print("plot skipped:", e)


# ------------------------------------------------------------------------------------------ RQ4
def rq4(ctx):
    from scipy.stats import kendalltau, spearmanr
    from bench.inject import inject_one, INJECTORS
    from dataclasses import asdict
    omega = ctx.cfg["mti"]["omega"]
    ci = ConformalInterval(ctx.cfg["mti"]["conformal_alpha"])
    q = Path(ctx.cfg["paths"]["models"]) / "mti_conformal.json"
    if q.exists():
        ci.q = json.loads(q.read_text())["q"]
    out = {}
    # (a) agreement with human 1-5 ratings
    if ctx.a.ratings:
        ratings = {r["summary_id"]: r["rating"] for r in load_records(ctx.a.ratings)}
        items = [it for it in ctx.items() if it.rec["summary_id"] in ratings]
        _, _, posts, _ = run_cprove(ctx, items)
        xs, ys, cover = [], [], []
        for sid, (it, post, prog, y, edges, ok) in posts.items():
            s = mti(it.claims, edges, post, omega)
            xs.append(s); ys.append(ratings[sid])
            lo, hi = ci(s)
            cover.append(lo <= (ratings[sid] - 1) / 4 <= hi)
        out["kendall_tau"] = kendalltau(xs, ys).statistic
        out["spearman_rho"] = spearmanr(xs, ys).statistic
        out["conformal_coverage"] = float(np.mean(cover))
        out["n_rated"] = len(xs)
    # (b) dose-response on clean summaries with 0/25/50/75% corrupted sentences
    rng = random.Random(ctx.cfg["seed"])
    clean = [r for r in load_records(ctx.a.bench) if r.get("error_type") == "none"][: ctx.a.max_dose]
    recs = []
    for r in clean:
        m = ctx.meetings[r["meeting_id"]]
        md = {"meeting_id": m.meeting_id, "utterances": [asdict(u) for u in m.utterances], "meta": m.meta}
        n = len(r["sentences"])
        for frac in (0.0, 0.25, 0.5, 0.75):
            sents, gold, k = list(r["sentences"]), [1] * n, 0
            target = int(round(frac * n))
            tries = 0
            while k < target and tries < 30:
                tries += 1
                cand = inject_one(" ".join(sents), md, rng.choice(INJECTORS[:-1]), rng)
                if cand and len(cand["sentences"]) == n and all(gold[i] == 1 for i in cand["sent_idx"]):
                    sents = cand["sentences"]
                    for i in cand["sent_idx"]:
                        gold[i] = 0
                    k += len(cand["sent_idx"])
            recs.append({"summary_id": f"{r['summary_id']}_dose{int(frac * 100)}", "meeting_id": r["meeting_id"],
                         "sentences": sents, "gold": gold, "error_type": "mixed", "sent_idx": [i for i, g in enumerate(gold) if g == 0],
                         "dose": frac, "true_frac": 1 - np.mean(gold)})
    p = ctx.out / "preds" / f"{ctx.bench}__dose_records.jsonl"
    with open(p, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    items = ctx.items(str(p))
    _, _, posts, _ = run_cprove(ctx, items)
    rows = []
    for sid, (it, post, prog, y, edges, ok) in posts.items():
        comp = components(ctx.meetings[it.meeting_id], it.claims, edges, post, it.obs, it.grounding, ok, omega)
        rows.append({"dose": it.rec["dose"], "true_frac": it.rec["true_frac"], "mti": mti(it.claims, edges, post, omega), **comp})
    df = pd.DataFrame(rows)
    out["dose_spearman"] = spearmanr(df.true_frac, df.mti).statistic
    ctx.table("rq4_dose_response", df.groupby("dose", as_index=False).agg(
        mti_mean=("mti", "mean"), mti_sd=("mti", "std"), n=("mti", "size"),
        **{c: (c, "mean") for c in ["coverage", "contradiction_density", "attribution", "temporal", "support_quality"]}))
    ctx.table("rq4_summary", pd.DataFrame([out]))


# ------------------------------------------------------------------------------------------ RQ5
def rq5(ctx, k=3, n_random=5):
    """Deletion test: remove the top-k proof groundings of a claim and re-solve; a faithful
    explanation should move y_i more than removing k random groundings that touch the claim."""
    import torch
    from cprove.hlmrf import Compiled
    items = ctx.items()
    _, _, posts, model = run_cprove(ctx, items, scheduler="none")
    rng = random.Random(0)
    lam = model.w().detach()
    res = []
    for sid, (it, post, prog, y, edges, ok) in list(posts.items())[: ctx.a.max_rq5]:
        comp = Compiled.from_program(prog)
        for c in it.claims:
            pt = proof_trace(prog, comp, y, lam, c.cid, k)
            vars_i = {v for (cc, d), v in prog.var_index.items() if cc == c.cid}
            touching = [g for g, gr in enumerate(prog.groundings) if any(v in vars_i for v, _ in gr.terms)
                        and not gr.meta.get("kind", "").startswith("y->") and gr.meta.get("kind") != "mean->y"]
            kinds = [p["kind"] for p in pt]
            top = [g for g in touching if prog.groundings[g].meta.get("kind") in kinds][: 2 * k]

            def drop(ids):
                p2 = copy.copy(prog)
                p2.groundings = [gr for g, gr in enumerate(prog.groundings) if g not in set(ids)]
                y2, _ = model.infer(p2, y0=y)
                return abs(float(y2[prog.y_index(c.cid)]) - post[c.cid]["y"])
            d_top = drop(top)
            d_rand = np.mean([drop(rng.sample(touching, min(len(top), len(touching)))) for _ in range(n_random)]) if touching else 0
            res.append({"cid": c.cid, "delta_top": d_top, "delta_random": d_rand, "y": post[c.cid]["y"]})
    df = pd.DataFrame(res)
    from scipy.stats import wilcoxon
    w = wilcoxon(df.delta_top, df.delta_random, alternative="greater") if len(df) > 10 else None
    ctx.table("rq5_deletion", pd.DataFrame([{"n_claims": len(df), "mean_delta_top": df.delta_top.mean(),
                                              "mean_delta_random": df.delta_random.mean(),
                                              "win_rate": float((df.delta_top > df.delta_random).mean()),
                                              "wilcoxon_p": w.pvalue if w else np.nan}]))


# ------------------------------------------------------------------------------------------ RQ6
def rq6(ctx):
    import torch
    items = ctx.items()
    timing = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    run_cprove(ctx, items, timing=timing)
    df = pd.DataFrame(timing)
    stage_log = Path(ctx.cfg["paths"]["cache"]) / "stage_timing.jsonl"
    extra = {}
    if stage_log.exists():
        st = pd.DataFrame([json.loads(l) for l in open(stage_log)])
        extra = st.mean(numeric_only=True).to_dict()
    ctx.table("rq6_cost", pd.DataFrame([{
        "summaries": len(df), "claims_per_summary": df.claims.mean(), "edges_per_summary": df.edges.mean(),
        "solve+adaptive_s": df.solve_s.mean(), "llm_calls_per_summary": df.llm_calls.mean(),
        "peak_vram_gb_solver": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0, **extra}]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["rq1", "rq2", "ablate", "rq3", "rq4", "rq5", "rq6", "all"])
    ap.add_argument("--bench", required=True)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--ratings", default=None)
    ap.add_argument("--gold_edges", default=None)
    ap.add_argument("--simulate_judge", action="store_true", help="RQ3 without GPU: noisy-oracle judge (report as simulation)")
    ap.add_argument("--judge_noise", type=float, default=0.15)
    ap.add_argument("--no_judge_in_ablation", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--max_dose", type=int, default=200)
    ap.add_argument("--max_rq5", type=int, default=200)
    a = ap.parse_args()
    ctx = Ctx(a)
    fns = {"rq1": rq1, "rq2": rq2, "ablate": ablate, "rq3": rq3, "rq4": rq4, "rq5": rq5, "rq6": rq6}
    for k in (fns if a.what == "all" else [a.what]):
        print(f"\n######## {k}")
        fns[k](ctx)


if __name__ == "__main__":
    main()
