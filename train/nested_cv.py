"""Nested, meeting-grouped cross-validation on the feature table (CPU only, seconds).

  python -m train.nested_cv --features cache/frozen/features.jsonl --out outputs/nested_cv.json

Outer loop : leave-one-meeting-out. EVERY learned component is refitted inside each outer fold from that fold's
             training meetings only: feature-set choice, LR regularisation C, Platt calibration, decision threshold.
Inner loop : 5-fold meeting-grouped CV on the outer-training meetings -> out-of-fold scores -> selects (feature set, C),
             fits the Platt calibrator and the threshold (maximising macro-F1 on inner OOF scores).
Nothing from the held-out meeting touches any choice. Pooled outer predictions are then scored with meeting-clustered CIs.
"""
from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from cprove import metrics as M

FEATS = {"ent": ["ent"], "con": ["con"], "attr": ["attr", "attr_flag"], "sem": ["sem", "sem_flag"], "temp": ["temp", "temp_flag"],
         "ctx": ["ctx", "ctx_flag"], "mc": ["mc"], "num": ["num_unseen"], "conhi": ["con_high"]}
CANDIDATES = {   # feature-set name -> blocks
    "ent": ["ent"], "ent+con": ["ent", "con"], "ent+con+attr": ["ent", "con", "attr"], "ent+con+attr+sem": ["ent", "con", "attr", "sem"],
    "mc": ["mc"], "ent+con+mc": ["ent", "con", "mc"], "ent+con+attr+sem+mc": ["ent", "con", "attr", "sem", "mc"],
    "ent+con+attr+sem+mc+num": ["ent", "con", "attr", "sem", "mc", "num"],
    "all": ["ent", "con", "attr", "sem", "temp", "ctx", "mc", "num"]}
C_GRID = [0.1, 1.0, 10.0]


def load_rows(path, labelled_only=True, sets=None, unver_as_unsup=False):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if unver_as_unsup:      # sensitivity: claims judged UNVERIFIABLE by the author are counted as not supported
        for r in rows:
            if r.get("author_label") == "UNVERIFIABLE":
                r["author_label"] = "UNSUPPORTED"
    for r in rows:
        r["con_high"] = float(r["con"] >= 0.9)
        r.setdefault("mc", float("nan"))
    seen, uniq = set(), []
    for r in rows:   # a claim can appear in several sets (pilot/dev overlap): keep one copy
        if r["cid"] not in seen:
            seen.add(r["cid"]); uniq.append(r)
    rows = uniq
    if sets:
        rows = [r for r in rows if r["set"] in sets]
    if labelled_only:
        rows = [r for r in rows if r.get("author_label") in ("SUPPORTED", "UNSUPPORTED")]
    return rows


def matrix(rows, blocks):
    cols = [c for b in blocks for c in FEATS[b]]
    return np.array([[r[c] for c in cols] for r in rows], float)


def _fit(X, y, C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(X, y)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def inner_oof(X, y, g, C, k=5):
    oof = np.zeros(len(y)); gkf = GroupKFold(n_splits=min(k, len(set(g))))
    for tr, va in gkf.split(X, y, g):
        if len(set(y[tr])) < 2:
            oof[va] = y[tr].mean(); continue
        oof[va] = _fit(X[tr], y[tr], C).decision_function(X[va])
    return oof


def best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[int(np.argmax([M.prf(y, p, t)["macro_f1"] for t in grid]))])


def fit_select(rows_tr, cands):
    """Everything learned from the outer-training meetings only. Returns a predictor + the choices made."""
    y = np.array([int(r["author_label"] == "SUPPORTED") for r in rows_tr]); g = np.array([r["meeting_id"] for r in rows_tr])
    best = None
    for name, blocks in cands.items():
        X = matrix(rows_tr, blocks)
        if np.isnan(X).any():
            continue
        for C in C_GRID:
            oof = inner_oof(X, y, g, C)
            sc = M.auroc(y, oof)
            if best is None or sc > best[0]:
                best = (sc, name, C, oof)
    sc, name, C, oof = best
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(oof[:, None], y)          # calibration inside the fold
    thr = best_threshold(y, platt.predict_proba(oof[:, None])[:, 1])
    clf = _fit(matrix(rows_tr, cands[name]), y, C)
    blocks = cands[name]

    def predict(rows):
        s = clf.decision_function(matrix(rows, blocks))
        return platt.predict_proba(s[:, None])[:, 1]
    return predict, {"features": name, "C": C, "inner_auroc": round(sc, 4), "threshold": round(thr, 3)}


def run(rows, cands=None, fixed=None):
    """LOMO. `fixed`: dict name->blocks to also report fixed-feature-set fusions (still refit + Platt per fold)."""
    cands = cands or CANDIDATES
    meetings = sorted({r["meeting_id"] for r in rows})
    out = {"selected": np.full(len(rows), np.nan), "thr": np.full(len(rows), np.nan)}
    for k in (fixed or {}):
        out[k] = np.full(len(rows), np.nan)
    choices = []
    mid = np.array([r["meeting_id"] for r in rows])
    for m in meetings:
        te = np.where(mid == m)[0]; tr = np.where(mid != m)[0]
        rtr, rte = [rows[i] for i in tr], [rows[i] for i in te]
        pred, ch = fit_select(rtr, cands); ch["meeting"] = m; choices.append(ch)
        out["selected"][te] = pred(rte); out["thr"][te] = ch["threshold"]
        for k, blocks in (fixed or {}).items():
            p2, _ = fit_select(rtr, {k: blocks}); out[k][te] = p2(rte)
    return out, choices


def baselines(rows):
    import glob, os
    extra = {}
    for f in sorted(glob.glob("cache/frozen/baseline_*.json")):
        d = json.load(open(f)); nm = os.path.basename(f)[9:-5]
        extra[nm] = np.array([d.get(r["cid"], np.nan) for r in rows])
    return {**extra, "entailment": np.array([r["ent"] for r in rows]), "minicheck": np.array([r["mc"] for r in rows]),
            "cprove_hand_graph": np.array([r["y_graph"] for r in rows]), "cprove_hand_nograph": np.array([r["y_nograph"] for r in rows])}


def report(rows, preds, thr=None, boots=1000):
    y = np.array([int(r["author_label"] == "SUPPORTED") for r in rows]); g = np.array([r["meeting_id"] for r in rows])
    table = {}
    for k, p in preds.items():
        if np.isnan(p).any():
            continue
        t = thr if thr is not None else 0.5
        row = {"n": int(len(y)), "auroc": M.auroc(y, p), "auroc_ci": M.cluster_bootstrap(M.auroc, y, p, g, boots),
               "auprc": M.auprc(y, p), "auprc_ci": M.cluster_bootstrap(M.auprc, y, p, g, boots),
               "brier": M.brier(y, p), "logloss": M.logloss(y, p), "ece10": M.ece(y, p, 10), "ece10_ci": M.cluster_bootstrap(M.ece, y, p, g, boots),
               "recall_at_review": M.recall_at_review(y, p)}
        row.update(M.prf(y, p, t)); table[k] = row
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="cache/frozen/features.jsonl")
    ap.add_argument("--out", default="outputs/nested_cv.json")
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--boots", type=int, default=1000)
    ap.add_argument("--unverifiable_as_unsupported", action="store_true", help="include the UNVERIFIABLE claims (552 instead of 523) as unsupported")
    a = ap.parse_args()
    rows = load_rows(a.features, sets=a.sets, unver_as_unsup=a.unverifiable_as_unsupported)
    y = np.array([int(r["author_label"] == "SUPPORTED") for r in rows])
    print(f"{len(rows)} labelled claims, {len({r['meeting_id'] for r in rows})} meetings, supported {y.mean():.2f}")
    fixed = {"fusion ent+con": ["ent", "con"], "fusion +attr+sem": ["ent", "con", "attr", "sem"], "fusion +mc": ["ent", "con", "attr", "sem", "mc"]}
    out, choices = run(rows, fixed=fixed)
    preds = {**baselines(rows), "fusion (selected in inner CV)": out["selected"], **{k: out[k] for k in fixed}}
    # platt-calibrated baselines, fold-wise, for a fair ECE comparison
    meet = np.array([r["meeting_id"] for r in rows])
    for k in ("entailment", "minicheck", "cprove_hand_graph"):
        p = preds[k]; cal = np.zeros(len(p))
        for m in np.unique(meet):
            te, tr = meet == m, meet != m
            pl = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p[tr])[:, None], y[tr]); cal[te] = pl.predict_proba(_logit(p[te])[:, None])[:, 1]
        preds[k + " (Platt in fold)"] = cal
    thr = {k: None for k in preds}
    table = report(rows, preds, boots=a.boots)
    print(f"{'method':38} AUROC [95% meeting-clustered CI]   AUPRC   Brier   ECE10   macroF1")
    for k, r in table.items():
        print(f"{k:38} {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f},{r['auroc_ci'][1]:.3f}]   {r['auprc']:.3f}   {r['brier']:.3f}   {r['ece10']:.3f}   {r['macro_f1']:.3f}")
    print("feature sets chosen per outer fold:", dict(Counter(c["features"] for c in choices)))
    ref = "minicheck"
    ps = {}
    for k in ("fusion (selected in inner CV)", "fusion +mc", "cprove_hand_graph", "entailment"):
        d, lo, hi, p = M.paired_bootstrap(M.auroc, y, preds[k], preds[ref], meet)
        ps[k] = p; print(f"paired AUROC diff {k} - {ref}: {d:+.3f} [{lo:+.3f},{hi:+.3f}] p={p:.3f}")
    print("Holm-adjusted p:", {k: round(v, 3) for k, v in M.holm(ps).items()})
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"table": table, "choices": choices, "paired_vs_minicheck": ps}, open(a.out, "w"), indent=1, default=float)
    np.save(Path(a.out).with_suffix(".preds.npy"), {"cids": [r["cid"] for r in rows], **{k: v.tolist() for k, v in preds.items()}}, allow_pickle=True)


if __name__ == "__main__":
    main()
