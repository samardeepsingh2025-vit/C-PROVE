"""Join adjudicated gold labels with their summaries -> MeetVerify-Human split files.

  python -m bench.annotate.export --summaries data/meetverify/summaries_test.jsonl \
      --gold data/meetverify/gold.jsonl --dev_frac 0.3
Writes human_dev.jsonl and human_test.jsonl, split by MEETING (dev is used for calibration,
CRC and lambda early-stopping; test is never touched before the final run).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--dev_frac", type=float, default=0.3)
    ap.add_argument("--outdir", default="data/meetverify")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    gold = defaultdict(list)
    for l in open(a.gold, encoding="utf-8"):
        g = json.loads(l)
        if g["label"] != "UNVERIFIABLE":
            gold[g["summary_id"]].append({"cid": g["cid"], "gold": g["gold"], "error_type": g["error_type"]})
    sums = [json.loads(l) for l in open(a.summaries, encoding="utf-8")]
    sums = [s for s in sums if s["summary_id"] in gold]
    mids = sorted({s["meeting_id"] for s in sums})
    random.Random(a.seed).shuffle(mids)
    dev = set(mids[: int(len(mids) * a.dev_frac)])
    with open(f"{a.outdir}/human_dev.jsonl", "w", encoding="utf-8") as fd, open(f"{a.outdir}/human_test.jsonl", "w", encoding="utf-8") as ft:
        for s in sums:
            s = dict(s, claims=gold[s["summary_id"]])
            (fd if s["meeting_id"] in dev else ft).write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"dev meetings {len(dev)}, test meetings {len(mids) - len(dev)}")


if __name__ == "__main__":
    main()
