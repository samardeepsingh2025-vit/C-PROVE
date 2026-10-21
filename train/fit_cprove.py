"""Fit everything C-PROVE learns from labelled data (CPU is enough; ~minutes):

  1. per-dimension calibrators (isotonic) on DEV       -> models/calibrators.pkl
  2. rule weights lambda by unrolled HL-MRF learning on TRAIN (early stop on DEV) -> models/rule_weights.json
  3. CRC abstention band on DEV                         -> models/crc.json
  4. MTI split-conformal radius + MTI-5 component weights on DEV -> models/mti_conformal.json, models/mti5.json

  python -m train.fit_cprove --train data/meetverify/synth_train.jsonl --dev data/meetverify/synth_dev.jsonl \
                             [--human_dev data/meetverify/human_dev.jsonl] [--ratings data/meetverify/ratings.jsonl]

Stages 1-4 (LLM frames, retrieval, NLI, dimension raw scores) are computed once and cached
(cache/items_*.pkl); calibration re-maps the cached raw scores, so refits are instant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cprove.utils import load_config, seed_everything
from cprove.data import load_corpus
from cprove.pipeline import CProve, Ablation
from cprove.dimensions import Calibrators
from cprove.hlmrf import HLMRF, Weights
from cprove.conformal import CRCLabeler
from cprove.mti import mti, components, ConformalInterval, MTI5
from experiments.common import load_records, build_items, programs, sentence_scores, metrics

# which dimension an injected/annotated error type falsifies (for dimension-specific calibration targets)
ERROR2DIM = {"entity": "fact", "negation": "fact", "fabrication": "fact", "attribution_swap": "attr",
             "time_shift": "temp", "revision_ignored": "ctx", "status_flip": "sem", "dependency_cascade": "temp"}
DIMS = ("fact", "attr", "temp", "ctx", "sem")


def recalibrate(items, cal: Calibrators):
    for it in items:
        for o in it.obs.values():
            for d in DIMS:
                r = o.raw.get(d)
                setattr(o, d, None if r is None else cal(d, r))


def fit_calibrators(items, method="isotonic"):
    raw = {d: [] for d in DIMS}
    tgt = {d: [] for d in DIMS}
    for it in items:
        for c in it.claims:
            if c.cid not in it.gold:
                continue
            o = it.obs[c.cid]
            for d in DIMS:
                r = o.raw.get(d)
                if r is None:
                    continue
                g = it.gold[c.cid]
                if g == 0 and c.gold_error in ERROR2DIM:
                    # dimension-specific target: only the falsified dimension (and fact for fabrication) is "wrong"
                    t = 0 if ERROR2DIM[c.gold_error] == d else 1
                else:
                    t = g
                raw[d].append(r); tgt[d].append(t)
    return Calibrators(method).fit(raw, tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--human_dev", default=None, help="claim-level human gold (dev split) — preferred for CRC/calibration")
    ap.add_argument("--ratings", default=None, help="jsonl {summary_id, rating 1-5} for MTI conformal / MTI-5")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed_everything(cfg["seed"])
    cache = Path(cfg["paths"]["cache"])
    models = Path(cfg["paths"]["models"])
    meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
    cp = CProve(cfg, Ablation(hand_weights=True))
    # raw (uncalibrated) observations: make sure the pipeline's calibrators are identity while building
    cp.dims.cal = Calibrators("none")
    tr = build_items(cp, meetings, load_records(a.train), str(cache / ("items_" + Path(a.train).stem + ".pkl")))
    dv = build_items(cp, meetings, load_records(a.dev), str(cache / ("items_" + Path(a.dev).stem + ".pkl")))
    hd = build_items(cp, meetings, load_records(a.human_dev), str(cache / ("items_" + Path(a.human_dev).stem + ".pkl"))) if a.human_dev else []

    # 1. calibrators
    cal = fit_calibrators(dv + hd, cfg["dimensions"]["calibration"])
    cal.save(models / "calibrators.pkl")
    for its in (tr, dv, hd):
        recalibrate(its, cal)
    print("calibrators:", sorted(cal.maps))

    # 2. rule weights
    h = cfg["hlmrf"]
    model = HLMRF(Weights(h["init_weights"]), h["iters"], h["tol"])
    ptr, pdv = programs(tr), programs(dv)
    best, best_state, patience = 1e9, None, 0
    L = h["learn"]
    for ep in range(a.epochs or L["epochs"]):
        model.fit(ptr, [i.gold for i in tr], epochs=1, lr=L["lr"], unroll=L["unroll"], l2=L["l2"], verbose=False)
        v = model.eval_loss(pdv, [i.gold for i in dv])
        print(f"epoch {ep}: dev BCE {v:.4f}")
        if v < best - 1e-4:
            best, best_state, patience = v, model.w.theta.detach().clone(), 0
        else:
            patience += 1
            if patience >= 4:
                break
    if best_state is not None:
        model.w.theta.data = best_state
    model.w.save(models / "rule_weights.json")
    print("lambda:", {k: round(v, 3) for k, v in model.w.as_dict().items()})

    # 3. CRC on claim-level dev posteriors (human dev if available)
    cal_items = hd or dv
    ys, gs = [], []
    post_by = {}
    for it, p in zip(cal_items, programs(cal_items)):
        y, _ = model.infer(p)
        post = model.posteriors(p, y)
        post_by[it.rec["summary_id"]] = (it, post)
        for cid, g in it.gold.items():
            ys.append(post[cid]["y"]); gs.append(g)
    crc = CRCLabeler(cfg["labels"]["risk_alpha"]).fit(ys, gs)
    crc.save(models / "crc.json")
    print("CRC:", crc.report(ys, gs))
    print("dev claim metrics:", metrics(gs, ys))

    # 4. MTI conformal: target = human rating (1-5 -> 0..1) if given, else gold fraction of faithful claims
    ratings = {}
    if a.ratings:
        for r in load_records(a.ratings):
            ratings[r["summary_id"]] = (float(r["rating"]) - 1) / 4
    omega = cfg["mti"]["omega"]
    preds, tgts, comps = [], [], []
    for sid, (it, post) in post_by.items():
        s = mti(it.claims, it.edges, post, omega)
        t = ratings.get(sid, np.mean(list(it.gold.values())) if it.gold else None)
        if t is None:
            continue
        preds.append(s); tgts.append(t)
        comps.append(components(meetings[it.meeting_id], it.claims, it.edges, post, it.obs, it.grounding, it.order_ok, omega))
    ci = ConformalInterval(cfg["mti"]["conformal_alpha"]).fit(preds, tgts)
    (models / "mti_conformal.json").write_text(json.dumps({"q": ci.q, "alpha": ci.alpha, "n": len(preds)}))
    MTI5().fit(comps, tgts).save(models / "mti5.json")
    print(f"MTI conformal radius q={ci.q:.3f} on n={len(preds)}")


if __name__ == "__main__":
    main()
