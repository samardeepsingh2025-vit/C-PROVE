"""E7 part 1+2 (CPU, seconds): decomposition statistics + example claims.
  python -m train.decomp_stats
(a) Atomicity flags of the two independent annotators on the 50 gold claims (per annotator; agreement; 'adjudicated' = both agree, otherwise disputed).
(b) Claims-per-sentence distribution over the whole claim pool (over-/under-split indicators).
(c) Example sentences -> claims for five categories (multi-actor, multi-decision, negation, quantity, revision): prints 3 candidates each; writes
    outputs/decomp_examples.tex using the first candidate of each category (replace by hand if a better one is printed)."""
import argparse, glob, json, re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from cprove.text import split_sentences

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="data/meetverify_v2/agreement"); ap.add_argument("--a", default="B1"); ap.add_argument("--b", default="B2")
ap.add_argument("--features", default="cache/frozen/features.jsonl"); ap.add_argument("--summaries", nargs="+", default=glob.glob("data/meetverify/qwen_*.jsonl"))
a = ap.parse_args()
L = lambda p: [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
A = {r["public_id"]: r for r in L(f"{a.dir}/ann_{a.a}.jsonl") if r["type"] == "claim"}
B = {r["public_id"]: r for r in L(f"{a.dir}/ann_{a.b}.jsonl") if r["type"] == "claim"}
ids = sorted(set(A) & set(B)); K = ["atomic", "compound", "fragment"]
print(f"== atomicity on {len(ids)} gold claims")
for nm, D in ((a.a, A), (a.b, B)):
    c = Counter(D[i]["atomicity"] for i in ids); print(f"  {nm}: " + ", ".join(f"{k} {c[k]} ({100 * c[k] / len(ids):.0f}%)" for k in K))
agree = [i for i in ids if A[i]["atomicity"] == B[i]["atomicity"]]; c = Counter(A[i]["atomicity"] for i in agree)
print(f"  adjudicated (both agree, n={len(agree)}): " + ", ".join(f"{k} {c[k]}" for k in K) + f"; disputed {len(ids) - len(agree)}")
anyflag = [i for i in ids if A[i]["atomicity"] != "atomic" or B[i]["atomicity"] != "atomic"]
print(f"  claims flagged non-atomic by at least one annotator: {len(anyflag)} ({100 * len(anyflag) / len(ids):.0f}%)")
rows = {}
for l in open(a.features, encoding="utf-8"):
    r = json.loads(l); rows.setdefault(r["cid"], r)
rows = list(rows.values())
per = defaultdict(list)
for r in rows:
    per[(r["summary_id"], r["sent_idx"])].append(r)
n = np.array([len(v) for v in per.values()]); cc = Counter(n.tolist())
print(f"\n== claims per summary sentence ({len(per)} sentences, {len(rows)} claims): mean {n.mean():.2f}; distribution " + ", ".join(f"{k}: {cc[k]}" for k in sorted(cc)))
print(f"  sentences kept as a single claim: {100 * (n == 1).mean():.0f}%; split into >=3 claims: {100 * (n >= 3).mean():.0f}%")
ROLES = re.compile(r"\b(Project Manager|Industrial Designer|User Interface|Marketing)\b")
summ = {}
import os
for p in [q for q in a.summaries if os.path.exists(q)]:
    for r in L(p):
        if r.get("summary"):
            summ[r["summary_id"]] = r["summary"]
def sent_of(r):
    s = summ.get(r["summary_id"]); 
    if not s: return "(summary text not found)"
    ss = split_sentences(s); return ss[r["sent_idx"]] if r["sent_idx"] < len(ss) else "(?)"
CAT = {"multi-actor": lambda r: len(set(ROLES.findall(r["text"]))) >= 2 or len({x["frame"].get("agent") for x in per[(r["summary_id"], r["sent_idx"])] if x["frame"].get("agent")}) >= 2,
       "multi-decision": lambda r: len(re.findall(r"\b(decid|agree|chose|select|adopt)\w*", r["text"], re.I)) >= 2 or (r.get("claim_type") == "decision" and " and " in r["text"]),
       "negation": lambda r: (r["frame"].get("polarity") == 0) or bool(re.search(r"\b(not|no longer|never|n't|without|rejected|dropped)\b", r["text"], re.I)),
       "quantity": lambda r: bool(r["frame"].get("quantity")) or bool(re.search(r"\d", r["text"])),
       "revision": lambda r: r["frame"].get("status") == "revised" or bool(re.search(r"\b(instead|changed|switched|revised|replaced|originally|later)\b", r["text"], re.I))}
ex = {}
print("\n== example candidates (sentence -> its claims; * = the matching claim)")
for cat, fn in CAT.items():
    hits = [r for r in rows if fn(r) and len(per[(r["summary_id"], r["sent_idx"])]) >= 2 and r.get("author_label")]
    hits.sort(key=lambda r: (-len(per[(r["summary_id"], r["sent_idx"])]), r["cid"])); ex[cat] = []
    for r in hits[:3]:
        sib = per[(r["summary_id"], r["sent_idx"])]
        ex[cat].append({"sentence": sent_of(r), "claims": [(x["text"], x.get("author_label")) for x in sib], "cid": r["cid"]})
        print(f"[{cat}] {r['cid']}\n   sentence: {sent_of(r)}")
        for x in sib:
            print(f"   {'*' if x['cid'] == r['cid'] else ' '} {x['text']}  [{x.get('author_label')}]")
esc = lambda s: s.replace("\\", "").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("$", "\\$").replace("#", "\\#")
t = ["\\begin{table}[htbp]", "\\centering", "\\caption{Decomposition examples: a summary sentence and the atomic claims extracted from it, with the author's label (S: supported, U: unsupported, V: unverifiable).}", "\\label{tab:decompex}", "\\small", "\\begin{tabularx}{\\textwidth}{@{}lXX@{}}", "\\toprule", "Category & Summary sentence & Extracted claims \\\\", "\\midrule"]
tag = {"SUPPORTED": "S", "UNSUPPORTED": "U", "UNVERIFIABLE": "V"}
for cat, v in ex.items():
    if v:
        e = v[0]; t.append(f"{cat} & {esc(e['sentence'])} & " + " \\newline ".join(f"({tag.get(l, '?')}) {esc(c)}" for c, l in e["claims"]) + " \\\\")
t += ["\\bottomrule", "\\end{tabularx}", "\\end{table}"]
Path("outputs").mkdir(exist_ok=True); open("outputs/decomp_examples.tex", "w", encoding="utf-8").write("\n".join(t)); print("\nwrote outputs/decomp_examples.tex")
