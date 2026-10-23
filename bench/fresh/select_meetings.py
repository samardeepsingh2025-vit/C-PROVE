"""Pick FRESH AMI meetings that were never used for dev/test/labelling.

  python -m bench.fresh.select_meetings --n 22 --out data/meetverify_v2/meetings.json

Excludes every meeting id that appears in any data/meetverify/*.jsonl (dev, test, pilot, summaries, drafts).
Only scenario meetings (ES/IS/TS) with a sensible length are eligible; selection is seeded and logged,
so the reviewer can see it was not cherry-picked.
"""
from __future__ import annotations
import argparse, glob, json, random, re
from pathlib import Path
from cprove.utils import load_config
from cprove.data import load_corpus


# Meetings that carry HUMAN claim labels / drafts (dev + test). The synthetic files (synth_*, summaries_train)
# cover every meeting and were only used for hand-weight/calibrator tuning without human labels, so they are
# NOT excluded here (this is disclosed in the paper).
LABELLED = ["annotations_*.jsonl", "items_*.jsonl", "*drafts*.jsonl", "qwen_*.jsonl"]


def used_ids(folder: str) -> set[str]:
    used = set()
    pat = re.compile(r"\b((?:ES|IS|TS)\d{4}[a-z])\b")
    for g in LABELLED:
        for f in glob.glob(str(Path(folder) / g)):
            for line in open(f, encoding="utf-8"):
                used |= set(pat.findall(line))
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=22)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min_utt", type=int, default=250)
    ap.add_argument("--max_utt", type=int, default=1400)
    ap.add_argument("--used_dir", default="data/meetverify")
    ap.add_argument("--out", default="data/meetverify_v2/meetings.json")
    a = ap.parse_args()
    cfg = load_config()
    meetings = load_corpus(cfg)[0]
    used = used_ids(a.used_dir)
    elig = [m for m in meetings if m.meeting_id not in used and a.min_utt <= len(m.utterances) <= a.max_utt
            and re.match(r"(ES|IS|TS)\d{4}[a-z]$", m.meeting_id)]
    # at most one meeting per scenario group (e.g. ES2002) in the first pass, to spread across groups
    random.Random(a.seed).shuffle(elig)
    chosen, groups = [], set()
    for m in elig:
        g = m.meeting_id[:-1]
        if g not in groups:
            chosen.append(m); groups.add(g)
        if len(chosen) == a.n:
            break
    for m in elig:
        if len(chosen) >= a.n:
            break
        if m not in chosen:
            chosen.append(m)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"seed": a.seed, "n_used_excluded": len(used), "n_eligible": len(elig),
               "meetings": sorted(m.meeting_id for m in chosen)}, open(a.out, "w"), indent=1)
    print(f"{len(meetings)} meetings, {len(used)} used/excluded, {len(elig)} eligible -> chose {len(chosen)}: "
          f"{sorted(m.meeting_id for m in chosen)}")


if __name__ == "__main__":
    main()
