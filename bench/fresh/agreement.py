"""Inter-annotator agreement + adjudication + gold-label export.

  python -m bench.fresh.agreement --dir data/meetverify_v2 --a B1 --b B2 [--author data/meetverify/annotations_A1.jsonl]
  python -m bench.fresh.agreement --dir data/meetverify_v2 --adjudicate        # interactive, resumable
  python -m bench.fresh.agreement --dir data/meetverify_v2 --export-gold       # gold_claims.jsonl, gold_summaries.jsonl

Reports: Cohen's kappa (4-class, binary supported-vs-not, atomicity, needs_review), raw agreement, per-class
positive agreement, confusion matrix, evidence-line overlap, summary-rating quadratic-weighted kappa + Spearman.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.stats import spearmanr

LABELS = ["SUPPORTED", "CONTRADICTED", "CONFLICTING", "INSUFFICIENT"]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if Path(p).exists() else []


def by_key(recs, typ, key):
    return {r[key]: r for r in recs if r["type"] == typ}


def per_class_agreement(x, y, labels):
    out = {}
    for c in labels:
        both = sum(1 for p, q in zip(x, y) if p == c and q == c)
        tot = sum(1 for p in x if p == c) + sum(1 for q in y if q == c)
        out[c] = round(2 * both / tot, 3) if tot else None
    return out


def report(d, a_name, b_name, author=None):
    A, B = load(d / f"ann_{a_name}.jsonl"), load(d / f"ann_{b_name}.jsonl")
    ca, cb = by_key(A, "claim", "public_id"), by_key(B, "claim", "public_id")
    ids = sorted(set(ca) & set(cb))
    x = [ca[i]["label"] for i in ids]; y = [cb[i]["label"] for i in ids]
    print(f"== claims labelled by both: {len(ids)}")
    print(f"4-class  kappa {cohen_kappa_score(x, y, labels=LABELS):.3f}   raw agreement {np.mean([p == q for p, q in zip(x, y)]):.3f}")
    bx = [p == "SUPPORTED" for p in x]; by_ = [q == "SUPPORTED" for q in y]
    print(f"binary (SUPPORTED vs not) kappa {cohen_kappa_score(bx, by_):.3f}   raw agreement {np.mean([p == q for p, q in zip(bx, by_)]):.3f}")
    print("label counts A:", dict(Counter(x)), " B:", dict(Counter(y)))
    print("per-class positive agreement:", per_class_agreement(x, y, LABELS))
    print("confusion matrix (rows = A, cols = B)", LABELS)
    print(confusion_matrix(x, y, labels=LABELS))
    ax = [ca[i]["atomicity"] for i in ids]; ay = [cb[i]["atomicity"] for i in ids]
    print(f"atomicity kappa {cohen_kappa_score(ax, ay):.3f}  counts A {dict(Counter(ax))} B {dict(Counter(ay))}")
    jac = []
    for i in ids:
        sa, sb = set(ca[i]["evidence_lines"]), set(cb[i]["evidence_lines"])
        if sa or sb:
            jac.append(len(sa & sb) / len(sa | sb))
    print(f"evidence-line Jaccard (mean over claims with evidence) {np.mean(jac):.3f}   any-overlap {np.mean([j > 0 for j in jac]):.3f}")
    sa, sb = by_key(A, "summary", "sid"), by_key(B, "summary", "sid")
    sids = sorted(set(sa) & set(sb))
    if sids:
        ra = [sa[s]["rating"] for s in sids]; rb = [sb[s]["rating"] for s in sids]
        print(f"== summaries rated by both: {len(sids)}  quadratic-weighted kappa {cohen_kappa_score(ra, rb, weights='quadratic'):.3f}  "
              f"Spearman {spearmanr(ra, rb)[0]:.3f}  needs_review kappa {cohen_kappa_score([sa[s]['needs_review'] for s in sids], [sb[s]['needs_review'] for s in sids]):.3f}")
    if author:
        key = json.load(open(d / "key_private.json"))
        authlab = {}
        for l in open(author, encoding="utf-8"):
            r = json.loads(l); authlab[r["cid"]] = r["label"]
        gold = final_labels(d, a_name, b_name)
        rows = []
        for sid, k in key.items():
            for j, cid in enumerate(k.get("old_cids", [])):
                pid = f"{sid}.{j + 1}"
                if pid in gold and cid in authlab:
                    rows.append((authlab[cid], gold[pid]))
        if rows:
            ap_ = [r[0] == "SUPPORTED" for r in rows]; gp = [r[1] == "SUPPORTED" for r in rows]
            print(f"== author-reviewed labels vs adjudicated independent gold on {len(rows)} old claims: binary kappa {cohen_kappa_score(ap_, gp):.3f}  "
                  f"agreement {np.mean([p == q for p, q in zip(ap_, gp)]):.3f}  changed {sum(p != q for p, q in zip(ap_, gp))}")
            print("author label x gold label:", dict(Counter(rows)))


def final_labels(d, a_name, b_name):
    A, B = load(d / f"ann_{a_name}.jsonl"), load(d / f"ann_{b_name}.jsonl")
    ca, cb = by_key(A, "claim", "public_id"), by_key(B, "claim", "public_id")
    adj = {r["public_id"]: r for r in load(d / "adjudicated.jsonl")}
    out = {}
    for i in set(ca) & set(cb):
        if ca[i]["label"] == cb[i]["label"]:
            out[i] = ca[i]["label"]
        elif i in adj:
            out[i] = adj[i]["label"]
    return out


def adjudicate(d, a_name, b_name):
    """Third-person (or joint) adjudication of every disagreement. Resumable."""
    pack = json.load(open(d / "pack.json", encoding="utf-8"))
    T = pack["transcripts"]; where = {}
    for s in pack["summaries"]:
        for c in s["claims"]:
            where[c["public_id"]] = (s, c)
    A, B = load(d / f"ann_{a_name}.jsonl"), load(d / f"ann_{b_name}.jsonl")
    ca, cb = by_key(A, "claim", "public_id"), by_key(B, "claim", "public_id")
    adj_path = d / "adjudicated.jsonl"
    done = {r["public_id"] for r in load(adj_path)}
    todo = [i for i in sorted(set(ca) & set(cb)) if ca[i]["label"] != cb[i]["label"] and i not in done]
    print(f"{len(todo)} disagreements to adjudicate")
    for i in todo:
        s, c = where[i]; tr = T[s["meeting_id"]]
        print("\n" + "=" * 90 + f"\nCLAIM {i}: {c['text']}")
        for nm, r in ((a_name, ca[i]), (b_name, cb[i])):
            print(f"  {nm}: {r['label']:13} lines {r['evidence_lines']}  error={r['error_type']}  note={r['note']}")
        lines = sorted(set(ca[i]["evidence_lines"]) | set(cb[i]["evidence_lines"]))
        for n in lines:
            u = tr[n - 1]; print(f"   [{n}] {u['speaker']}: {u['text']}")
        while True:
            ans = input("final label  1=SUPPORTED 2=CONTRADICTED 3=CONFLICTING 4=INSUFFICIENT  (s=stop): ").strip()
            if ans == "s":
                return
            if ans in "1234" and ans:
                break
        why = input("one-line reason: ").strip()
        with open(adj_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"public_id": i, "label": LABELS[int(ans) - 1], "reason": why}) + "\n")


def export_gold(d, a_name, b_name):
    pack = json.load(open(d / "pack.json", encoding="utf-8"))
    key = json.load(open(d / "key_private.json"))
    gold = final_labels(d, a_name, b_name)
    A, B = load(d / f"ann_{a_name}.jsonl"), load(d / f"ann_{b_name}.jsonl")
    ca, cb = by_key(A, "claim", "public_id"), by_key(B, "claim", "public_id")
    n = 0
    with open(d / "gold_claims.jsonl", "w", encoding="utf-8") as f:
        for s in pack["summaries"]:
            k = key[s["sid"]]
            for j, c in enumerate(s["claims"]):
                pid = c["public_id"]
                if pid not in gold:
                    continue
                lines = sorted(set(ca[pid]["evidence_lines"]) | set(cb[pid]["evidence_lines"]))
                uids = [pack["transcripts"][s["meeting_id"]][x - 1]["uid"] for x in lines]
                f.write(json.dumps({"public_id": pid, "cid": c["cid"] if k["system"] != "OLD" else k["old_cids"][j], "meeting_id": s["meeting_id"],
                                    "set": "old" if k["system"] == "OLD" else "fresh", "system": k["system"], "text": c["text"],
                                    "label": gold[pid], "label_binary": "SUPPORTED" if gold[pid] == "SUPPORTED" else "UNSUPPORTED",
                                    "error_type": ca[pid]["error_type"] if ca[pid]["label"] != "SUPPORTED" else cb[pid]["error_type"],
                                    "gold_evidence_uids": uids, "atomicity_a": ca[pid]["atomicity"], "atomicity_b": cb[pid]["atomicity"],
                                    "agreed_before_adjudication": ca[pid]["label"] == cb[pid]["label"]}, ensure_ascii=False) + "\n")
                n += 1
    SA, SB = by_key(A, "summary", "sid"), by_key(B, "summary", "sid")
    with open(d / "gold_summaries.jsonl", "w", encoding="utf-8") as f:
        for sid in sorted(set(SA) & set(SB)):
            f.write(json.dumps({"sid": sid, **key[sid], "rating_a": SA[sid]["rating"], "rating_b": SB[sid]["rating"],
                                "rating_mean": (SA[sid]["rating"] + SB[sid]["rating"]) / 2,
                                "needs_review_a": SA[sid]["needs_review"], "needs_review_b": SB[sid]["needs_review"]}) + "\n")
    print(f"gold labels for {n} claims -> {d}/gold_claims.jsonl, gold_summaries.jsonl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/meetverify_v2")
    ap.add_argument("--a", default="B1"); ap.add_argument("--b", default="B2")
    ap.add_argument("--author", default=None)
    ap.add_argument("--adjudicate", action="store_true"); ap.add_argument("--export-gold", action="store_true")
    a = ap.parse_args(); d = Path(a.dir)
    if a.adjudicate:
        adjudicate(d, a.a, a.b)
    elif a.export_gold:
        export_gold(d, a.a, a.b)
    else:
        report(d, a.a, a.b, a.author)
