"""D8: candidates for the refreshed qualitative example and the status-error failure box, from REAL claims scored by the CURRENT frozen pipeline
(out-of-fold selected fusion from outputs/nested_cv.preds.npy). CPU, seconds.   python -m train.qualitative
Prints (1) the three meetings with the most mixed supported/unsupported claims and their claims with trust, label, error type and key scores;
(2) author-labelled status errors (and all unsupported claims with an error type) ranked by how HIGH the fusion trusts them (the misses), with the
evidence lines the author attached so the failure can be diagnosed. Pick one meeting + one miss; paste the output and I write the table/box LaTeX."""
import argparse, json
from collections import Counter, defaultdict
import numpy as np
from train.nested_cv import load_rows

ap = argparse.ArgumentParser(); ap.add_argument("--features", default="cache/frozen/features.jsonl")
ap.add_argument("--preds", default="outputs/nested_cv.preds.npy"); ap.add_argument("--meetings", default="data/ami_meetings.jsonl"); ap.add_argument("--key", default="fusion (selected in inner CV)")
a = ap.parse_args()
rows = load_rows(a.features); P = np.load(a.preds, allow_pickle=True).item(); tr = dict(zip(P["cids"], P[a.key]))
utt = {}
for l in open(a.meetings, encoding="utf-8"):
    r = json.loads(l)
    for u in r["utterances"]: utt[u["uid"]] = f"{u['speaker']}: {u['text']}"
print("error types among unsupported claims:", dict(Counter(r.get("author_error") for r in rows if r["author_label"] != "SUPPORTED")))
by = defaultdict(list)
for r in rows: by[r["meeting_id"]].append(r)
cand = sorted(by.items(), key=lambda kv: -min(sum(x["author_label"] == "SUPPORTED" for x in kv[1]), sum(x["author_label"] != "SUPPORTED" for x in kv[1]))
              if 6 <= len(kv[1]) <= 25 else 0)[:3]
for mid, rs in cand:
    print(f"\n=== {mid}: {len(rs)} claims")
    for x in sorted(rs, key=lambda x: tr[x["cid"]]):
        print(f"  trust {tr[x['cid']]:.2f}  {x['author_label'][:3]}  err={x.get('author_error')}  ent {x['ent']:.2f} con {x['con']:.2f}  | {x['text']}")
print("\n=== status errors and other unsupported claims with the HIGHEST fusion trust (misses)")
miss = sorted([r for r in rows if r["author_label"] != "SUPPORTED"], key=lambda r: -tr[r["cid"]])
shown = 0
for r in miss:
    if shown >= 12: break
    st = "status" in str(r.get("author_error")).lower()
    print(f"\n[{'STATUS' if st else r.get('author_error')}] trust {tr[r['cid']]:.2f} ent {r['ent']:.2f} con {r['con']:.2f} {r['meeting_id']}\n  claim: {r['text']}")
    for u in (r.get("author_evidence") or [])[:4]: print("   evidence:", utt.get(u, u))
    shown += 1
