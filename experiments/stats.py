"""Significance testing used in every results table.

  paired_bootstrap  resample MEETINGS (not sentences) with replacement, 10,000 times;
                    returns mean diff, 95% CI and two-sided p for metric(A) - metric(B)
  mcnemar           exact binomial McNemar on binary decisions (thresholds fixed on dev)
  holm              Holm-Bonferroni correction across the baseline family
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import binomtest


def _auroc(gold, score):
    from sklearn.metrics import roc_auc_score
    gold = np.asarray(gold)
    if len(set(gold)) < 2:
        return np.nan
    return roc_auc_score(1 - gold, 1 - np.asarray(score))


def _bacc(gold, score, thr=0.5):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(gold, (np.asarray(score) >= thr).astype(int))


METRICS = {"auroc": _auroc, "bacc": _bacc}


def align(rows_a, rows_b):
    b = {r["unit"]: r for r in rows_b}
    pairs = [(r, b[r["unit"]]) for r in rows_a if r["unit"] in b and r["gold"] is not None]
    return pairs


def paired_bootstrap(rows_a, rows_b, metric="auroc", n=10000, seed=0, thr_a=0.5, thr_b=0.5):
    pairs = align(rows_a, rows_b)
    by_m = defaultdict(list)
    for ra, rb in pairs:
        by_m[ra["meeting_id"]].append((ra["gold"], ra["score"], rb["score"]))
    mids = sorted(by_m)
    f = METRICS[metric]
    kw_a = {} if metric == "auroc" else {"thr": thr_a}
    kw_b = {} if metric == "auroc" else {"thr": thr_b}

    def stat(ms):
        rows = [x for m in ms for x in by_m[m]]
        g = [r[0] for r in rows]
        return f(g, [r[1] for r in rows], **kw_a) - f(g, [r[2] for r in rows], **kw_b)

    obs = stat(mids)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        d = stat(list(rng.choice(mids, len(mids), replace=True)))
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.array(diffs)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(obs), "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
            "p": float(min(1.0, p)), "n_meetings": len(mids), "n_units": len(pairs)}


def mcnemar(rows_a, rows_b, thr_a=0.5, thr_b=0.5):
    pairs = align(rows_a, rows_b)
    ca = np.array([(ra["score"] >= thr_a) == bool(ra["gold"]) for ra, _ in pairs])
    cb = np.array([(rb["score"] >= thr_b) == bool(ra["gold"]) for ra, rb in pairs])
    b01 = int((ca & ~cb).sum()); b10 = int((~ca & cb).sum())
    p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    return {"a_only_correct": b01, "b_only_correct": b10, "p": float(p)}


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for k, (name, p) in enumerate(items):
        adj = min(1.0, (m - k) * p)
        running = max(running, adj)
        out[name] = running
    return out
