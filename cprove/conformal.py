r"""Selective claim labelling with conformal risk control (CRC, Angelopoulos et al. 2024).

Label rule with band half-width t:
    SUPPORTED   if y >= 0.5 + t
    UNSUPPORTED if y <= 0.5 - t
    ABSTAIN     otherwise
Loss per claim: 1[label is wrong and not ABSTAIN]  (non-increasing in t).
CRC picks the smallest t such that  n/(n+1) * Rhat(t) + 1/(n+1) <= alpha,
which guarantees E[loss] <= alpha on exchangeable test claims.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def labels_for(y, t: float, center: float = 0.5):
    y = np.asarray(y, float)
    out = np.full(len(y), "ABSTAIN", dtype=object)
    out[y >= center + t] = "SUPPORTED"
    out[y <= center - t] = "UNSUPPORTED"
    return out


class CRCLabeler:
    def __init__(self, alpha=0.10, center=0.5):
        self.alpha, self.center, self.t = alpha, center, 0.0

    def fit(self, y, gold):
        y, gold = np.asarray(y, float), np.asarray(gold, int)
        n = len(y)
        grid = np.linspace(0, 0.5, 101)
        self.t = 0.5
        for t in grid:
            lab = labels_for(y, t, self.center)
            wrong = ((lab == "SUPPORTED") & (gold == 0)) | ((lab == "UNSUPPORTED") & (gold == 1))
            if n / (n + 1) * wrong.mean() + 1 / (n + 1) <= self.alpha:
                self.t = float(t)
                break
        return self

    def __call__(self, y):
        return labels_for(y, self.t, self.center)

    def report(self, y, gold):
        lab = self(y)
        gold = np.asarray(gold, int)
        keep = lab != "ABSTAIN"
        wrong = ((lab == "SUPPORTED") & (gold == 0)) | ((lab == "UNSUPPORTED") & (gold == 1))
        return {"t": self.t, "coverage": float(keep.mean()), "risk": float(wrong.mean()),
                "selective_error": float(wrong[keep].mean()) if keep.any() else float("nan")}

    def save(self, path):
        Path(path).write_text(json.dumps({"alpha": self.alpha, "t": self.t, "center": self.center}))

    @staticmethod
    def load(path, alpha=0.10):
        c = CRCLabeler(alpha)
        if path and Path(path).exists():
            d = json.loads(Path(path).read_text())
            c.t, c.center = d["t"], d.get("center", 0.5)
        return c
