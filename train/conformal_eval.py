"""Abstention with finite-sample risk control, evaluated honestly (CPU, seconds).

  python -m train.conformal_eval --features cache/frozen/features.jsonl --method "fusion (selected in inner CV)"

Setting. Scores p = out-of-fold calibrated P(supported) from the nested CV (so no claim was scored by a model that saw its
meeting). The system ACCEPTS a claim if max(p, 1-p) >= lambda (and predicts supported iff p >= 0.5), otherwise ABSTAINS
(= send to human review).  Two controls:
  CRC  conformal risk control (Angelopoulos et al.): loss_i(lambda) = 1[accepted_i and wrong_i] (monotone in lambda),
       lambda_hat = min{lambda : n/(n+1) * Rhat(lambda) + 1/(n+1) <= alpha}  =>  E[accepted-and-wrong rate] <= alpha.
  LTT  learn-then-test, fixed-sequence over lambda (high -> low), exact binomial (Clopper-Pearson) test of
       'error rate among accepted claims > alpha' at level delta  =>  P(selective error <= alpha) >= 1-delta.
Protocol. Exchangeable UNIT = MEETING (claims inside a meeting are dependent). Repeated random meeting-level splits
(50% calibration / 50% test, 1000 repeats); we report realised test risk, coverage, and how often the guarantee was violated.
"""
import argparse, json
import numpy as np
from scipy.stats import binom
from train.nested_cv import load_rows, run

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="cache/frozen/features.jsonl")
ap.add_argument("--method", default="fusion (selected in inner CV)")
ap.add_argument("--splits", type=int, default=1000); ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--out", default="outputs/conformal.json")
a = ap.parse_args()
rows = load_rows(a.features)
y = np.array([int(r["author_label"] == "SUPPORTED") for r in rows]); mid = np.array([r["meeting_id"] for r in rows])
out, _ = run(rows)
p = out["selected"]; conf = np.maximum(p, 1 - p); wrong = ((p >= 0.5).astype(int) != y).astype(float)
grid = np.unique(np.round(np.linspace(0.5, 0.99, 99), 3))


def crc(cal, alpha):
    n = len(cal)
    for lam in grid:
        acc = conf[cal] >= lam
        if n / (n + 1) * float((acc * wrong[cal]).mean()) + 1 / (n + 1) <= alpha:
            return lam
    return 1.01                                    # abstain on everything


def ltt(cal, alpha, delta):
    best = 1.01
    for lam in grid[::-1]:                         # fixed sequence: from most conservative downwards
        acc = cal[conf[cal] >= lam]
        k, m = int(wrong[acc].sum()), len(acc)
        pval = float(binom.cdf(k, m, alpha)) if m else 1.0          # P(Bin(m, alpha) <= k): small => selective error < alpha
        if pval <= delta:
            best = lam
        else:
            break
    return best


def test_stats(te, lam):
    acc = conf[te] >= lam
    return float((acc * wrong[te]).mean()), float(acc.mean()), (float(wrong[te][acc].mean()) if acc.any() else 0.0)


rng = np.random.default_rng(0); ms = np.unique(mid); res = {}
print(f"{len(rows)} claims, {len(ms)} meetings; unabstained error rate {wrong.mean():.3f}")
print("alpha  control  coverage  joint-risk(acc&wrong)  selective-error(acc)  violated (CRC: per-split risk>alpha; its guarantee is on the MEAN)")
for alpha in (0.05, 0.10, 0.20):
    for name in ("CRC", "LTT"):
        R, C, S = [], [], []
        for _ in range(a.splits):
            perm = rng.permutation(ms); cal_m = set(perm[: len(ms) // 2])
            cal = np.where(np.isin(mid, list(cal_m)))[0]; te = np.where(~np.isin(mid, list(cal_m)))[0]
            lam = crc(cal, alpha) if name == "CRC" else ltt(cal, alpha, a.delta)
            r, c, s = test_stats(te, lam); R.append(r); C.append(c); S.append(s)
        R, C, S = map(np.array, (R, C, S))
        viol = float((R > alpha).mean()) if name == "CRC" else float(((S > alpha) & (C > 0)).mean())
        res[f"{name}@{alpha}"] = {"coverage_mean": C.mean(), "coverage_sd": C.std(), "joint_risk_mean": R.mean(), "joint_risk_p95": float(np.percentile(R, 95)),
                                  "selective_error_mean": S.mean(), "violation_rate": viol}
        print(f"{alpha:<5}  {name:7}  {C.mean():.3f}±{C.std():.3f}   {R.mean():.3f} (p95 {np.percentile(R, 95):.3f})        {S.mean():.3f}               {viol:.3f}")
print("CRC guarantee: mean joint risk <= alpha.  LTT guarantee: P(selective error <= alpha) >= 1 - delta =", 1 - a.delta)
import pathlib; pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)
