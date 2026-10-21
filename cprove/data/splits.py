"""Meeting-level splits. AMI scenario meetings come in series of four (ES2002a-d);
all meetings of a series go to the same split so no speaker/scenario leaks across."""
from __future__ import annotations

import re

import numpy as np


def series_of(mid: str) -> str:
    m = re.match(r"([A-Z]{2}\d{4})[a-d]$", mid)
    return m.group(1) if m else mid


def meeting_split(meeting_ids: list[str], ratio=(0.70, 0.15, 0.15), seed: int = 42, by_series: bool = True):
    groups: dict[str, list[str]] = {}
    for mid in sorted(meeting_ids):
        groups.setdefault(series_of(mid) if by_series else mid, []).append(mid)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n = len(meeting_ids)
    out = {"train": [], "dev": [], "test": []}
    targets = [ratio[0] * n, (ratio[0] + ratio[1]) * n]
    count = 0
    for k in keys:
        split = "train" if count < targets[0] else "dev" if count < targets[1] else "test"
        out[split].extend(groups[k])
        count += len(groups[k])
    return out
