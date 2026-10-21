"""Convert TofuEval-MeetingBank sentence labels into our bench format (UNTESTED: needs the real files).
  python -m bench.prepare_tofueval --ann data/tofueval/annotations.csv --transcripts data/tofueval/meetings.jsonl --out data/meetverify/tofueval_test.jsonl
Transcripts jsonl: our normalised format ({meeting_id, utterances[{speaker,text}]}); build it from HF 'huuuyeah/meetingbank'."""
import argparse, json
from collections import defaultdict
from cprove.data.loaders import load_tofueval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True); ap.add_argument("--transcripts", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meetings, rows = load_tofueval(a.ann, a.transcripts)
    by = defaultdict(list)
    for r in rows:
        by[(r["doc_id"], r["summ_id"])].append(r)
    with open(a.out, "w", encoding="utf-8") as f:
        for (doc, sid), rs in by.items():
            rs.sort(key=lambda r: r["sent_idx"])
            f.write(json.dumps({"summary_id": f"{doc}__{sid}", "meeting_id": doc, "sentences": [r["sentence"] for r in rs],
                                "gold": [r["label"] for r in rs], "error_type": "unknown", "sent_idx": [i for i, r in enumerate(rs) if not r["label"]]}) + "\n")
    print(len(by), "summaries ->", a.out)


if __name__ == "__main__":
    main()
