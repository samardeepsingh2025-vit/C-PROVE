"""Evaluation metrics + meeting-clustered uncertainty + paired tests. Unsupported = positive class everywhere.

All functions take numpy arrays: y (1 = supported, 0 = unsupported), p (P(supported)), groups (meeting ids).
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, confusion_matrix


def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(roc_auc_score(1 - y, 1 - p)) if len(set(y)) == 2 else float("nan")


def auprc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(average_precision_score(1 - y, 1 - p)) if len(set(y)) == 2 else float("nan")


def brier(y, p):
    return float(brier_score_loss(np.asarray(y), np.clip(p, 0, 1)))


def logloss(y, p):
    return float(log_loss(np.asarray(y), np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]))


def ece(y, p, bins=10, mode="width"):
    """Expected calibration error. mode='width': equal-width bins on [0,1]; 'mass': equal-count bins (quantiles)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if mode == "mass":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    else:
        edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & ((p < hi) if i < len(edges) - 2 else (p <= hi))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def prf(y, p, thr):
    """Precision/recall/F1 for the UNSUPPORTED class, per-class F1, macro-F1, accuracy, confusion matrix [[TN,FP],[FN,TP]]
    with unsupported = positive."""
    y = np.asarray(y); pred = (np.asarray(p) >= thr).astype(int)
    pos, ppos = 1 - y, 1 - pred
    tp = int(((pos == 1) & (ppos == 1)).sum()); fp = int(((pos == 0) & (ppos == 1)).sum())
    fn = int(((pos == 1) & (ppos == 0)).sum()); tn = int(((pos == 0) & (ppos == 0)).sum())
    def f1(a, b, c): return 2 * a / (2 * a + b + c) if (2 * a + b + c) else 0.0
    f1_u, f1_s = f1(tp, fp, fn), f1(tn, fn, fp)
    return {"precision_unsup": tp / (tp + fp) if tp + fp else 0.0, "recall_unsup": tp / (tp + fn) if tp + fn else 0.0,
            "f1_unsup": f1_u, "f1_sup": f1_s, "macro_f1": (f1_u + f1_s) / 2, "accuracy": (tp + tn) / len(y),
            "confusion": [[tn, fp], [fn, tp]]}


def recall_at_review(y, p, fracs=(0.1, 0.2, 0.3, 0.5)):
    """Review prioritisation: fraction of all unsupported claims found when only the lowest-trust `frac` of claims is reviewed."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p, kind="stable"); n_unsup = int((y == 0).sum())
    out = {}
    for f in fracs:
        k = max(1, int(round(f * len(y)))); out[f] = float((y[order[:k]] == 0).sum() / n_unsup) if n_unsup else float("nan")
    return out


def _cluster_idx(groups, rng):
    ug = np.unique(groups); pick = rng.choice(ug, size=len(ug), replace=True)
    idx = {g: np.where(groups == g)[0] for g in ug}
    return np.concatenate([idx[g] for g in pick])


def cluster_bootstrap(fn, y, p, groups, n=1000, seed=0, **kw):
    """Meeting-clustered percentile 95% CI of fn(y, p). Resamples MEETINGS, not claims."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    rng = np.random.default_rng(seed); s = []
    for _ in range(n):
        i = _cluster_idx(groups, rng)
        v = fn(y[i], p[i], **kw)
        if not np.isnan(v):
            s.append(v)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def paired_bootstrap(fn, y, p_a, p_b, groups, n=2000, seed=0, **kw):
    """Paired meeting-clustered bootstrap of fn(A) - fn(B). Returns (delta, ci_lo, ci_hi, two-sided p)."""
    y, pa, pb, groups = np.asarray(y), np.asarray(p_a), np.asarray(p_b), np.asarray(groups)
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = _cluster_idx(groups, rng)
        va, vb = fn(y[i], pa[i], **kw), fn(y[i], pb[i], **kw)
        if not (np.isnan(va) or np.isnan(vb)):
            d.append(va - vb)
    d = np.array(d); obs = fn(y, pa, **kw) - fn(y, pb, **kw)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(obs), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(min(1.0, max(p, 1.0 / n)))


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a dict name -> p."""
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); out = {}; run = 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); out[k] = run
    return out
