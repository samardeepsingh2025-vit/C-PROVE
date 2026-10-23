"""Convert a hand-made JSON annotation file {public_id: {label, error_type, evidence, atomicity, note}} into the
ann_<name>.jsonl format used by agreement.py. Validates every record against pack.json.

  python -m bench.fresh.convert_manual --in friend1.json --annotator B1 --pack data/meetverify_v2/agreement/pack.json
"""
import argparse, json, re, time
from pathlib import Path
from bench.fresh.annotate_app import parse_lines, LABELS

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True); ap.add_argument("--annotator", required=True)
ap.add_argument("--pack", default="data/meetverify_v2/agreement/pack.json")
a = ap.parse_args()
pack = json.load(open(a.pack, encoding="utf-8"))
where = {c["public_id"]: s for s in pack["summaries"] for c in s["claims"]}
raw = open(a.inp, encoding="utf-8").read().strip()
data = json.loads(raw)
if "summaries" in data:  # nested export format
    data = {c["public_id"]: {"label": c["label"], "error_type": c.get("error_type"), "evidence": c.get("evidence_lines", ""),
                             "atomicity": c.get("atomicity"), "note": c.get("notes", "")}
            for sm in data["summaries"] for c in sm["claims"]}
recs, bad = [], []
for pid, r in data.items():
    s = where.get(pid)
    if not s:
        bad.append((pid, "id not in pack")); continue
    lab = r["label"].strip().upper()
    if lab == "UNSUPPORTED":
        bad.append((pid, "label UNSUPPORTED is not allowed; use CONTRADICTED or INSUFFICIENT")); continue
    if lab not in LABELS:
        bad.append((pid, f"bad label {lab}")); continue
    T = pack["transcripts"][s["meeting_id"]]
    lines = parse_lines(str(r.get("evidence", "")), len(T))
    if lab != "INSUFFICIENT" and not lines:
        bad.append((pid, f"{lab} without evidence lines"))
    atom = (r.get("atomicity") or "atomic").split()[0].lower()
    err = (r.get("error_type") or "none") if lab != "SUPPORTED" else "none"
    recs.append({"type": "claim", "public_id": pid, "sid": s["sid"], "annotator": a.annotator, "label": lab, "error_type": err,
                 "evidence_lines": lines, "evidence_uids": [T[i - 1]["uid"] for i in lines], "atomicity": atom,
                 "note": r.get("note", ""), "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
missing = sorted(set(where) - set(data))
out = Path(a.pack).parent / f"ann_{a.annotator}.jsonl"
out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs), encoding="utf-8")
print(f"{len(recs)} records -> {out}; claims in pack {len(where)}; missing {len(missing)} {missing[:5]}; problems {len(bad)}")
for b in bad:
    print("  PROBLEM", b)
