"""Decision-status check: a prompted status classifier compares the CLAIM's status with the status the transcript evidence expresses.

  python -m train.status_check --stage label            # Ollama (Qwen2.5-7B, cached): ~1 s per item; resumable
  python -m train.status_check --stage eval             # CPU, seconds
States (7 + one for absent evidence): proposed, discussed, accepted, rejected, deferred, revised, unresolved | not_discussed.
Groups: COMMIT = {accepted, revised}; OPEN = {proposed, discussed, deferred, unresolved}; REJECT = {rejected}.
Features (added to the learned fusion): stat_over  = claim in COMMIT while evidence in OPEN/REJECT ("decided" but only discussed / rejected)
                                        stat_under = claim in OPEN while evidence in COMMIT (understated).  Evidence 'not_discussed' flags nothing.
Targets: the 523 natural claims AND the robustness variants (clean, status-changed, paraphrase) from cache/frozen/robust.jsonl.
No human span annotation exists: the classifier is validated only against the claim-level labels (author error type 'status') and the injected changes."""
import argparse, json, hashlib
from collections import Counter
from pathlib import Path
import numpy as np

STATES = ["proposed", "discussed", "accepted", "rejected", "deferred", "revised", "unresolved"]
COMMIT, OPEN, REJ = {"accepted", "revised"}, {"proposed", "discussed", "deferred", "unresolved"}, {"rejected"}
SYS = ("You label the STATUS of one item in a meeting. Statuses: proposed (someone suggested it, no response yet); discussed (talked about, "
       "no decision); accepted (the participants agreed or decided to do it); rejected (decided against it); deferred (postponed to later or "
       "another person); revised (an earlier decision was changed); unresolved (disagreement or open question remains). "
       "Answer from the quoted text only. If the quoted transcript does not talk about the item, transcript_status is not_discussed.")
SCHEMA = {"type": "object", "required": ["claim_status", "transcript_status"], "properties": {
    "claim_status": {"type": "string", "enum": STATES}, "transcript_status": {"type": "string", "enum": STATES + ["not_discussed"]}}}


def group(s):
    return "C" if s in COMMIT else "O" if s in OPEN else "R" if s in REJ else None


def flags(claim_status, ev_status):
    gc, ge = group(claim_status), group(ev_status)
    if gc is None or ge is None:
        return 0.0, 0.0
    return float(gc == "C" and ge in ("O", "R")), float(gc == "O" and ge == "C")


def label(a):
    from cprove.utils import load_config, seed_everything
    from cprove.data import load_corpus
    from cprove.llm import make_llm
    from cprove.schema import Claim, Frame
    from cprove.retrieval_eval import MultiRetriever
    cfg = load_config(a.config); seed_everything(cfg["seed"])
    meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}; rc = cfg["retrieval"]
    ret = MultiRetriever(rc["encoder"], rc["dense_weight"], rc["sparse_weight"], rc["top_k"], rc["context_window"], rc["slot_boost"]); llm = make_llm(cfg["llm"])
    feats = {}
    for l in open(a.features, encoding="utf-8"):
        r = json.loads(l); feats.setdefault(r["cid"], r)
    items = [(r["cid"], r["meeting_id"], r["text"], r.get("frame")) for r in feats.values() if r.get("author_label") in ("SUPPORTED", "UNSUPPORTED")]
    for l in open(a.robust, encoding="utf-8"):
        r = json.loads(l)
        if r["kind"] in ("clean", "status", "paraphrase"):
            items.append((r["vid"], r["meeting_id"], r["text"], (feats.get(r["src"]) or {}).get("frame")))
    if a.limit: items = items[: a.limit]
    path = Path(a.out); out = json.load(open(path)) if path.exists() else {}
    for n, (iid, mid, text, fr) in enumerate(items):
        if iid in out: continue
        m = meetings[mid]; frame = Frame(**{k: v for k, v in (fr or {}).items() if k in Frame.__dataclass_fields__})
        ev = ret.retrieve_mode(m, [Claim(cid=iid, text=text, frame=frame)], "hybrid", 5, 2)[iid]
        user = f"Claim: {text}\n\nTranscript lines:\n{ev.text}\n\nGive claim_status (the status the claim asserts) and transcript_status (the status the transcript lines show for the same item)."
        r = llm.json(SYS, user, SCHEMA, 128) or {}
        out[iid] = {"claim_status": r.get("claim_status"), "transcript_status": r.get("transcript_status")}
        if n % 25 == 0:
            json.dump(out, open(path, "w")); print(f"{n}/{len(items)}", flush=True)
    json.dump(out, open(path, "w")); print(f"{len(out)} items -> {path}")


def evaluate(a):
    from cprove import metrics as M
    import train.nested_cv as N
    N.FEATS["stat"] = ["stat_over", "stat_under"]
    base = dict(N.CANDIDATES)
    plus = {**{k + "+stat": v + ["stat"] for k, v in base.items() if "attr" in k or k == "all"}}
    lab = json.load(open(a.out))
    rows = N.load_rows(a.features)
    for r in rows:
        r["stat_over"], r["stat_under"] = flags(*(lab.get(r["cid"], {}).get(k) for k in ("claim_status", "transcript_status")))
    y = np.array([int(r["author_label"] == "SUPPORTED") for r in rows]); g = np.array([r["meeting_id"] for r in rows])
    err = np.array([("status" in (r.get("author_error") or "").lower()) and r["author_label"] != "SUPPORTED" for r in rows])
    print(f"{len(rows)} claims; unsupported error types:", dict(Counter(r.get("author_error") for r in rows if r["author_label"] != "SUPPORTED")))
    print(f"NATURAL status errors in the 523 claims: {int(err.sum())} (in {len({m for m, e in zip(g, err) if e})} meetings)")
    ov, un = np.array([r["stat_over"] for r in rows]), np.array([r["stat_under"] for r in rows])
    print(f"classifier flags: over {int(ov.sum())}, under {int(un.sum())}; on natural status errors: over-flag {int(ov[err].sum())}/{int(err.sum())}, "
          f"any flag {int(((ov + un) > 0)[err].sum())}/{int(err.sum())}; on supported claims: any flag {int(((ov + un) > 0)[y == 1].sum())}/{int((y == 1).sum())}")
    print(f"claim-status vs author: precision of any flag for status errors {((ov + un > 0) & err).sum() / max(1, (ov + un > 0).sum()):.2f}")
    res = {}
    for name, cands in (("without status feature", base), ("with status feature", plus)):
        out, ch = N.run(rows, cands=cands); p = out["selected"]; res[name] = p
        sel = Counter(c["features"] for c in ch)
        sub = (y == 1) | err
        auc_e = M.auroc(y[sub], p[sub]) if err.any() else float("nan")
        k = int(np.ceil(0.2 * len(p))); low = np.argsort(p)[:k]
        print(f"[{name}] overall AUROC {M.auroc(y, p):.3f} | natural status errors vs supported AUROC {auc_e:.3f} | status errors in lowest-20% {int(err[low].sum())}/{int(err.sum())} | sets {dict(sel)}")
    d, lo, hi, pv = M.paired_bootstrap(M.auroc, y, res["with status feature"], res["without status feature"], g)
    print(f"overall paired AUROC change {d:+.3f} [{lo:+.3f},{hi:+.3f}] p={pv:.3f}")
    # ---- injected: leave-one-meeting-out fits on natural rows, detect below the 20th percentile of clean scores
    V = [json.loads(l) for l in open(a.robust, encoding="utf-8")]; V = [v for v in V if v["kind"] in ("clean", "status", "paraphrase") and v["vid"] in lab]
    for v in V:
        v["con_high"] = float(v["con"] >= 0.9); v["num_unseen"] = float(v.get("num_unseen", 0)); v.setdefault("mc", float("nan"))
        cs, es = lab[v["vid"]].get("claim_status"), lab[v["vid"]].get("transcript_status")
        v["stat_over"], v["stat_under"] = flags(cs, es); v["_cs"] = cs
    for name, cands in (("without status feature", base), ("with status feature", {**base, **plus})):
        for v in V: v["_s_" + name] = np.nan
        for m in sorted({v["meeting_id"] for v in V}):
            tr = [r for r in rows if r["meeting_id"] != m]; pred, _ = N.fit_select(tr, cands)
            te = [v for v in V if v["meeting_id"] == m]
            for v, s in zip(te, pred(te)): v["_s_" + name] = float(s)
        key = "_s_" + name; cl = [v[key] for v in V if v["kind"] == "clean" and not np.isnan(v[key])]; thr = np.percentile(cl, 20)
        for kind, sel in (("status (all)", lambda v: v["kind"] == "status"),
                          ("status: overclaim (claim now decided)", lambda v: v["kind"] == "status" and group(v["_cs"]) == "C"),
                          ("status: understatement (claim now discussed)", lambda v: v["kind"] == "status" and group(v["_cs"]) == "O"),
                          ("paraphrase (false flags)", lambda v: v["kind"] == "paraphrase")):
            S = [v for v in V if sel(v) and not np.isnan(v[key])]
            if S:
                print(f"[injected, {name}] {kind:44} n={len(S):3} detection {np.mean([v[key] < thr for v in S]):.2f}   status-flag rate {np.mean([(v['stat_over'] + v['stat_under']) > 0 for v in S]):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["label", "eval"]); ap.add_argument("--config", default="configs/frozen.yaml")
    ap.add_argument("--features", default="cache/frozen/features.jsonl"); ap.add_argument("--robust", default="cache/frozen/robust.jsonl")
    ap.add_argument("--out", default="cache/frozen/status_llm.json"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); label(a) if a.stage == "label" else evaluate(a)
