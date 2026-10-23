"""Controlled robustness test (E12): inject ONE error into verified-supported claims and check detection.
Error types: number, negation, speaker, time, status.  Control: meaning-preserving paraphrase (should NOT be flagged).
Every claim (clean, corrupted, paraphrased) is re-framed with the SAME heuristic extractor, so frame extraction is held fixed.

  python -m train.robustness --stage build      # GPU: C-PROVE features + MiniCheck for all variants -> cache/frozen/robust.jsonl
  python -m train.robustness --stage eval       # CPU: detection tables -> outputs/robustness.json
Methods: entailment, MiniCheck, C-PROVE hand posterior (no graph), learned fusion (leave-one-meeting-out fit, ent+con+attr+sem+temp+ctx+mc+num),
and fusion + numeric rule (min(p, cap) when a claim number is absent from the transcript).
Score = P(supported); a corruption is 'detected' if its score is below the 20th percentile of clean-claim scores (80% of clean claims pass).
Limitation (stated in the paper): synthetic, single-error, template-based corruptions; dependency errors need multi-claim graphs and are excluded."""
from __future__ import annotations
import argparse, json, random, re, gc
from pathlib import Path

ROLES = ["Project Manager", "Industrial Designer", "User Interface", "Marketing"]
PARA = [(r"\bdecided to\b", "chose to"), (r"\bproposed\b", "suggested"), (r"\bdiscussed\b", "talked about"), (r"\bagreed to\b", "consented to"),
        (r"\bwants to\b", "would like to"), (r"\bpresented\b", "showed"), (r"\buse\b", "utilise"), (r"\bcost\b", "price"), (r"\bsimple\b", "straightforward"),
        (r"\bthe team\b", "the group"), (r"\bimportant\b", "significant")]
NEG = [(r"\bdecided to\b", "decided not to"), (r"\bagreed to\b", "refused to"), (r"\bwill\b", "will not"), (r"\bis to\b", "is not to"),
       (r"\bwas\b", "was not"), (r"\bwere\b", "were not"), (r"\bcan\b", "cannot"), (r"\bwants to\b", "does not want to"), (r"\bproposed\b", "did not propose"),
       (r"\bsuggested\b", "did not suggest")]
STATUS = [(r"\bdecided to\b", "discussed whether to"), (r"\bagreed to\b", "considered whether to"), (r"\bdecided\b", "discussed"), (r"\bagreed\b", "suggested"), (r"\bdiscussed\b", "decided"), (r"\bsuggested\b", "decided"),
          (r"\bproposed\b", "decided"), (r"\bconsidered\b", "decided"), (r"\bwill\b", "might")]
TIME = [(r"\bnext meeting\b", "previous meeting"), (r"\bearlier\b", "later"), (r"\blater\b", "earlier"), (r"\bfirst\b", "last"), (r"\bbefore\b", "after"),
        (r"\bafter\b", "before"), (r"\bby the next meeting\b", "in the previous meeting"), (r"\bpreviously\b", "afterwards")]


def _sub(rules, text):
    for pat, rep in rules:
        if re.search(pat, text, re.I):
            return re.sub(pat, rep, text, count=1, flags=re.I)
    return None


def corrupt_number(text, rng):
    ms = list(re.finditer(r"\d+(?:\.\d+)?", text))
    if not ms:
        return None
    m = rng.choice(ms); v = float(m.group(0)); new = v * 2 if v >= 2 else v + 3
    s = f"{new:g}" if "." not in m.group(0) else f"{new:.2f}".rstrip("0").rstrip(".")
    return text[:m.start()] + s + text[m.end():]


def corrupt_speaker(text, rng):
    for r in ROLES:
        if re.search(rf"\b{r}\b", text):
            return re.sub(rf"(?:\b[Tt]he )?\b{r}\b", rng.choice([x for x in ROLES if x != r]), text, count=1)
    return None


STATUS_FINAL = [(r"\bproposed\b", "decided"), (r"\bsuggested\b", "decided"), (r"\bdiscussed whether to\b", "decided to"), (r"\bconsidered\b", "decided"),
                (r"\bwould like to\b", "decided to"), (r"\bwants to\b", "decided to")]


def insert_number(text, rng):
    """Insert a quantity that is not in the claim (and, with high probability, not in the transcript) into a claim WITHOUT numbers."""
    if re.search(r"\d", text):
        return None
    return text.rstrip(". ") + rng.choice([" at a cost of 37 Euros", " for 83 Euros", " within 14 days", " with 6 buttons"]) + "."


def insert_time(text, rng):
    if re.search(r"\b(next|previous|earlier|later|before|after|first|last|previously)\b", text, re.I):
        return None
    return text.rstrip(". ") + rng.choice([" in the previous meeting", " last week"]) + "."


CORRUPT = {"number": corrupt_number, "number_ins": insert_number, "time_ins": insert_time, "negation": lambda t, r: _sub(NEG, t), "speaker": corrupt_speaker,
           "time": lambda t, r: _sub(TIME, t), "status": lambda t, r: _sub(STATUS, t), "paraphrase": lambda t, r: _sub(PARA, t)}


def build(a):
    import torch
    from cprove.utils import load_config, seed_everything
    from cprove.data import load_corpus
    from cprove.pipeline import CProve, Ablation
    from cprove.dimensions import Calibrators, numeric_check
    from cprove.hlmrf import HLMRF, Weights
    from cprove.schema import Claim, Frame
    from cprove.frames import heuristic_decompose
    from cprove.text import normalise_time
    from cprove.rules import ground
    from baselines.run_baselines import transcript_text
    cfg = load_config(a.config); seed_everything(cfg["seed"]); rng = random.Random(a.seed)
    meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
    feats = {}
    for l in open(a.features, encoding="utf-8"):
        r = json.loads(l); feats.setdefault(r["cid"], r)
    gold = {}
    if Path(a.gold).exists():
        for l in open(a.gold, encoding="utf-8"):
            g = json.loads(l); gold[g["cid"]] = g["label"]
    clean = [r for r in feats.values() if (gold.get(r["cid"], r.get("author_label")) in ("SUPPORTED",)) and len(r["text"].split()) >= 4]
    rng.shuffle(clean)
    done_v = set()
    if a.only_new and Path(a.out).exists():
        done_v = {json.loads(l)["vid"] for l in open(a.out, encoding="utf-8")}
    import glob, pickle
    cl_ids = {r["cid"] for r in clean}; pairs = {"revises": [], "task_depends": []}
    for f in sorted(glob.glob(str(Path(cfg["paths"]["cache"]) / "items_*.pkl"))):
        for it in pickle.load(open(f, "rb")):
            for e in it.edges:
                if e.etype in pairs and e.src in cl_ids and e.dst in cl_ids:
                    pairs[e.etype].append((e.src, e.dst, float(e.weight)))
    print("supported-supported edge pairs available:", {k: len(v) for k, v in pairs.items()}, flush=True)
    variants = []   # (vid, kind, source cid, meeting, text[, extra])
    n_by = {k: 0 for k in CORRUPT}
    for r in clean:
        variants.append((f"{r['cid']}::clean", "clean", r["cid"], r["meeting_id"], r["text"]))
        for k, fn in CORRUPT.items():
            if n_by[k] >= a.per_type:
                continue
            t = fn(r["text"], rng)
            if t and t != r["text"]:
                variants.append((f"{r['cid']}::{k}", k, r["cid"], r["meeting_id"], t)); n_by[k] += 1
    # revision: the earlier PROPOSAL (superseded by a rule-detected revising claim) restated as the final decision
    rev_n = dep_n = 0; seen = set()
    for i, j, w in pairs["revises"]:
        if rev_n >= a.per_type or i in seen: continue
        seen.add(i)
        t = _sub(STATUS_FINAL, feats[i]["text"])
        if t and t != feats[i]["text"]:
            variants.append((f"{i}::revision", "revision", i, feats[i]["meeting_id"], t)); rev_n += 1
    # dependency: corrupt the decision i, score the (unchanged, supported) dependent claim j through the graph
    seen = set()
    for i, j, w in pairs["task_depends"]:
        if dep_n >= a.per_type or j in seen: continue
        seen.add(j)
        ti = _sub(NEG, feats[i]["text"]) or _sub(STATUS, feats[i]["text"])
        if ti and ti != feats[i]["text"]:
            variants.append((f"{j}::dependency", "dependency", j, feats[j]["meeting_id"], feats[j]["text"], {"i_text": ti, "i": i, "w": w})); dep_n += 1
    n_by["revision"], n_by["dependency"] = rev_n, dep_n
    variants = [v for v in variants if v[0] not in done_v]
    print("variants per type:", {k: v for k, v in n_by.items()}, "new variants:", len(variants), "clean:", sum(v[1] == "clean" for v in variants), flush=True)
    cp = CProve(cfg, Ablation(hand_weights=True)); cp.dims.cal = Calibrators("none")
    h = cfg["hlmrf"]; model = HLMRF(Weights(h["init_weights"]), h["iters"], h["tol"])
    rows = []
    def mkclaim(cid, text, m):
        parts = m.meta.get("participants") or m.speakers
        d = heuristic_decompose(text, parts, None, m.meta.get("date"), False)[0]
        fr = Frame(agent=d["agent"], act=d["act"], obj=d["object"], time=d["time"], time_norm=normalise_time(d["time"], m.meta.get("date")),
                   status=d["status"], polarity=d["polarity"], quantity=d["quantity"], recipient=d["recipient"])
        return Claim(cid=cid, text=text, frame=fr, summary_id="R", sent_idx=0, claim_type=d["claim_type"])

    for k, (vid, kind, src, mid, text, *extra) in enumerate(variants):
        m = meetings[mid]
        c = mkclaim(vid, text, m)
        _, _, obs, grounding = cp.observe(m, [c])
        pr = ground([c], obs, [], True, ())
        y, _ = model.infer(pr)
        y_graph = float(y[pr.y_index(vid)])
        if kind == "dependency":      # same claim j, but its premise i is corrupted and linked by a task_depends edge
            from cprove.schema import Edge
            ci = mkclaim(vid + "::prem", extra[0]["i_text"], m)
            _, _, obs2, _ = cp.observe(m, [ci, c])
            pr2 = ground([ci, c], obs2, [Edge(ci.cid, vid, "task_depends", extra[0]["w"])], True, ())
            y2, _ = model.infer(pr2); y_graph = float(y2[pr2.y_index(vid)])
        raw = obs[vid].raw; nc = numeric_check(m, c)
        row = {"vid": vid, "kind": kind, "src": src, "meeting_id": mid, "text": text, "ent": float(raw.get("ent") or 0), "con": float(raw.get("con") or 0),
               "num_unseen": float(bool(nc["unseen"])), "y_nograph": float(y[pr.y_index(vid)]), "y_graph": y_graph}
        for dd in ("attr", "temp", "ctx", "sem"):
            v = raw.get(dd); row[dd] = 0.0 if v is None else float(v) - 0.5; row[dd + "_flag"] = 0.0 if v is None else 1.0
        rows.append(row)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(variants)}", flush=True)
    del cp, model; gc.collect()
    try:
        from cprove.utils import SLOT; SLOT.release()
    except Exception:
        pass
    torch.cuda.empty_cache()
    from minicheck.minicheck import MiniCheck
    mc = MiniCheck(model_name="flan-t5-large", cache_dir="models/hf")
    by = {}
    for r in rows:
        by.setdefault(r["meeting_id"], []).append(r)
    for mid, rs in by.items():
        doc = transcript_text(meetings[mid])
        for b in range(0, len(rs), 2):
            ch = rs[b:b + 2]
            _, p, _, _ = mc.score(docs=[doc] * len(ch), claims=[x["text"] for x in ch])
            for x, v in zip(ch, p):
                x["mc"] = float(v)
            torch.cuda.empty_cache()
        print("  minicheck", mid, flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a" if a.only_new else "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} variants -> {a.out}")


def rehand(a):
    """Recompute y_nograph (hand-weight posterior, configured init weights only) for every row of robust.jsonl. NLI scores come from the disk cache;
    MiniCheck scores are kept. y_graph is set to y_nograph except for 'dependency' rows (kept as is)."""
    import torch
    from cprove.utils import load_config, seed_everything
    from cprove.data import load_corpus
    from cprove.pipeline import CProve, Ablation
    from cprove.dimensions import Calibrators
    from cprove.hlmrf import HLMRF, Weights
    from cprove.schema import Claim, Frame
    from cprove.frames import heuristic_decompose
    from cprove.text import normalise_time
    from cprove.rules import ground
    cfg = load_config(a.config); seed_everything(cfg["seed"])
    meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
    cp = CProve(cfg, Ablation(hand_weights=True)); cp.dims.cal = Calibrators("none")
    h = cfg["hlmrf"]; model = HLMRF(Weights(h["init_weights"]), h["iters"], h["tol"])
    R = [json.loads(l) for l in open(a.out, encoding="utf-8")]
    for k, r in enumerate(R):
        m = meetings[r["meeting_id"]]; parts = m.meta.get("participants") or m.speakers
        d = heuristic_decompose(r["text"], parts, None, m.meta.get("date"), False)[0]
        fr = Frame(agent=d["agent"], act=d["act"], obj=d["object"], time=d["time"], time_norm=normalise_time(d["time"], m.meta.get("date")),
                   status=d["status"], polarity=d["polarity"], quantity=d["quantity"], recipient=d["recipient"])
        c = Claim(cid=r["vid"], text=r["text"], frame=fr, summary_id="R", sent_idx=0, claim_type=d["claim_type"])
        _, _, obs, _ = cp.observe(m, [c]); pr = ground([c], obs, [], True, ()); y, _ = model.infer(pr)
        r["y_nograph"] = float(y[pr.y_index(r["vid"])])
        if r["kind"] != "dependency": r["y_graph"] = r["y_nograph"]
        if (k + 1) % 200 == 0: print(f"  {k + 1}/{len(R)}", flush=True)
    import shutil; shutil.copy(a.out, a.out + ".bak")
    with open(a.out, "w", encoding="utf-8") as f:
        for r in R: f.write(json.dumps(r) + "\n")
    print(f"{len(R)} rows re-scored with the configured hand weights -> {a.out}")


def evaluate(a):
    import numpy as np
    from cprove import metrics as M
    from sklearn.linear_model import LogisticRegression
    from train.nested_cv import load_rows, matrix, _fit
    R = [json.loads(l) for l in open(a.out, encoding="utf-8")]
    train = load_rows(a.features)
    blocks = ["ent", "con", "attr", "sem", "temp", "ctx", "mc", "num"]
    for r in R:
        r.setdefault("mc", float("nan"))
    ok = [r for r in train if not np.isnan(r["mc"])]
    Xt, yt = matrix(ok, blocks), np.array([int(r["author_label"] == "SUPPORTED") for r in ok]); gt = np.array([r["meeting_id"] for r in ok])
    meetings = sorted({r["meeting_id"] for r in R}); fus = {}
    for mid in meetings:      # leave-one-meeting-out: the model never saw the meeting of the claim it scores
        msk = gt != mid; clf = _fit(Xt[msk], yt[msk], 1.0)
        rs = [r for r in R if r["meeting_id"] == mid]
        p = clf.predict_proba(matrix(rs, blocks))[:, 1]
        for r, v in zip(rs, p):
            fus[r["vid"]] = float(v)
    for r in R:
        r["fusion"] = fus[r["vid"]]; r["fusion_num"] = min(r["fusion"], a.cap) if r["num_unseen"] else r["fusion"]
    meth = {"entailment": "ent", "minicheck": "mc", "cprove_hand": "y_nograph", "hand+graph": "y_graph", "fusion": "fusion", "fusion+numeric_rule": "fusion_num"}
    for r in R:
        r.setdefault("y_graph", r["y_nograph"])
    cl = [r for r in R if r["kind"] == "clean"]
    kinds = [("number", ["number"]), ("number_ins", ["number_ins"]), ("number (pooled)", ["number", "number_ins"]), ("time", ["time"]), ("time_ins", ["time_ins"]),
             ("time (pooled)", ["time", "time_ins"]), ("negation", ["negation"]), ("speaker", ["speaker"]), ("status", ["status"]),
             ("revision", ["revision"]), ("dependency", ["dependency"]), ("paraphrase", ["paraphrase"])]
    rng = np.random.default_rng(0); mids = np.array(sorted({r["meeting_id"] for r in R})); B = 500
    print(f"clean n={len(cl)}  (threshold = 20th percentile of clean scores; detection = share of corrupted claims below it; 95% meeting-clustered CI)")
    print(f"{'type':16} {'n':>4}  " + "  ".join(f"{m:>22}" for m in meth))
    out, L = {}, []
    for name, ks in kinds:
        cs = [r for r in R if r["kind"] in ks]
        if not cs:
            continue
        line, out[name] = [], {"n": len(cs)}; cells = []
        for m, key in meth.items():
            thr = float(np.percentile([r[key] for r in cl], 20)); det = float(np.mean([r[key] < thr for r in cs]))
            bs = []
            for _ in range(B):
                pick = set(rng.choice(mids, len(mids)).tolist()); c2 = [r for r in cs if r["meeting_id"] in pick]; l2 = [r for r in cl if r["meeting_id"] in pick]
                if len(c2) and len(l2):
                    bs.append(float(np.mean([r[key] < np.percentile([x[key] for x in l2], 20) for r in c2])))
            lo, hi = (np.percentile(bs, 2.5), np.percentile(bs, 97.5)) if bs else (float("nan"),) * 2
            out[name][m] = {"detect@80%clean": det, "ci": [float(lo), float(hi)]}
            line.append(f"{det:5.2f} [{lo:.2f},{hi:.2f}]"); cells.append(f"{det:.2f} [{lo:.2f}, {hi:.2f}]")
        print(f"{name:16} {len(cs):>4}  " + "  ".join(f"{x:>22}" for x in line))
        L.append(f"{name.replace('_', ' ')} & {len(cs)} & " + " & ".join(cells) + r" \\")
    print("paraphrase row = FALSE-FLAG rate (lower is better; ~0.20 is chance by construction); other rows = detection (higher is better)")
    print("dependency: the claim text is unchanged; only the graph methods can see the corrupted premise, so every other column just shows how often the clean claim is already low")
    hdr = r"\begin{tabular}{@{}lr" + "c" * len(meth) + r"@{}}\toprule Error type & n & " + " & ".join(m.replace("_", " ") for m in meth) + r" \\ \midrule"
    open("outputs/table_robust.tex", "w").write(hdr + "\n" + "\n".join(L) + "\n" + r"\bottomrule\end{tabular}")
    Path("outputs").mkdir(exist_ok=True); json.dump(out, open("outputs/robustness.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "eval", "rehand"], required=True); ap.add_argument("--config", default="configs/frozen.yaml")
    ap.add_argument("--features", default="cache/frozen/features.jsonl"); ap.add_argument("--gold", default="data/meetverify_v2/agreement/gold_claims.jsonl")
    ap.add_argument("--out", default="cache/frozen/robust.jsonl"); ap.add_argument("--per_type", type=int, default=120)
    ap.add_argument("--seed", type=int, default=3); ap.add_argument("--cap", type=float, default=0.25)
    ap.add_argument("--only_new", action="store_true", help="build: append only variants not already in --out (new templates)")
    a = ap.parse_args()
    {"build": build, "eval": evaluate, "rehand": rehand}[a.stage](a)
