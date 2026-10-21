"""Does VoI-adaptive judging help on REAL held-out claims? Judge (Qwen via Ollama) is called ONCE per claim and cached;
schedulers none/random/entropy/voi/all are then replayed offline at several budgets.
Needs Ollama running. Resumable (judge answers are saved after every claim).
python -m train.adaptive_real --ann data\\meetverify\\annotations_A1.jsonl --test data\\meetverify\\qwen_test.jsonl"""
import argparse, json, time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from cprove.utils import load_config, seed_everything
from cprove.data import load_corpus
from cprove.pipeline import CProve, Ablation
from cprove.dimensions import Calibrators
from cprove.hlmrf import HLMRF, Weights
from cprove.adaptive import adaptive_verify
from experiments.common import load_records, build_items, programs
from train.fit_cprove import recalibrate

ap = argparse.ArgumentParser()
ap.add_argument("--ann", required=True); ap.add_argument("--test", required=True)
ap.add_argument("--cache", default="cache/judge_test.json"); ap.add_argument("--config", default=None)
ap.add_argument("--budgets", nargs="+", type=float, default=[0.1, 0.25, 0.5]); ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args()
cfg = load_config(a.config); seed_everything(cfg["seed"])
ann = {}
for l in open(a.ann, encoding="utf-8"):
    r = json.loads(l); ann[r["cid"]] = r
meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
recs = [r for r in load_records(a.test) if r.get("system") != "reference"]
cp = CProve(cfg, Ablation(hand_weights=True)); cp.dims.cal = Calibrators("none")
items = build_items(cp, meetings, recs, str(Path(cfg["paths"]["cache"]) / ("items_" + Path(a.test).stem + ".pkl")))
recalibrate(items, Calibrators.load(cfg["paths"]["models"] + "/calibrators.pkl"))
h = cfg["hlmrf"]
model = HLMRF(Weights.load(h.get("learned_weights"), h["init_weights"]), h["iters"], h["tol"])

# ---- 1. judge every claim once (cached)
cpath = Path(a.cache); cpath.parent.mkdir(parents=True, exist_ok=True)
jc = json.load(open(cpath)) if cpath.exists() else {}
if not cp.judge_llm.available:
    raise SystemExit("LLM judge unavailable: start Ollama (ollama serve) and check the model in the config.")
from cprove.judge import LLMJudge
todo = sum(1 for it in items for c in it.claims if c.cid not in jc); done = 0; t0 = time.time()
for it in items:
    if all(c.cid in jc for c in it.claims): continue
    m = meetings[it.meeting_id]; ev = cp.retriever.retrieve(m, it.claims)
    j = LLMJudge(cp.judge_llm, it.claims, ev, cp._neighbour_fn(it.claims, it.edges))
    for c in it.claims:
        if c.cid in jc: continue
        p = j(c.cid); jc[c.cid] = None if p is None else float(p); done += 1
        if done % 10 == 0:
            json.dump(jc, open(cpath, "w")); print(f"  judged {done}/{todo}  ({(time.time()-t0)/done:.1f}s/claim)", flush=True)
json.dump(jc, open(cpath, "w"))
print(f"judge answers: {sum(v is not None for v in jc.values())}/{len(jc)} usable")

# ---- 2. replay schedulers offline
lab = {c: int(r["label"] == "SUPPORTED") for c, r in ann.items() if r["label"] != "UNVERIFIABLE"}
def run(sch, bud, seed=0):
    progs = programs(items); ys, yt, calls = [], [], 0
    for k, (it, pr) in enumerate(zip(items, progs)):
        y, tr = adaptive_verify(model, pr, lambda cid: jc.get(cid), sch, bud, cfg["adaptive"]["epsilon"], seed=seed + k)
        calls += tr.calls
        for c in it.claims:
            if c.cid in lab: ys.append(float(y[pr.y_index(c.cid)])); yt.append(lab[c.cid])
    ys, yt = np.array(ys), np.array(yt)
    return roc_auc_score(1 - yt, 1 - ys), average_precision_score(1 - yt, 1 - ys), calls / len(items)
rows = [("none", 0.0, *run("none", 0.0))]
for sch in ("random", "entropy", "voi"):
    for b in a.budgets:
        rs = [run(sch, b, seed=s * 1000) for s in (range(a.seeds) if sch == "random" else [0])]
        rows.append((sch, b, *np.mean(rs, axis=0)))
        print(f"{sch:8} budget={b:<5} AUROC={rows[-1][2]:.3f}", flush=True)
rows.append(("all", 1.0, *run("all", 1.0)))
print(f"\n{'scheduler':9} {'budget':>6} {'calls/summary':>14} {'AUROC':>7} {'AUPRC(unsup)':>13}")
for s, b, au, ap_, cps in rows:
    print(f"{s:9} {b:>6.2f} {cps:>14.1f} {au:>7.3f} {ap_:>13.3f}")
json.dump([dict(scheduler=s, budget=b, auroc=au, auprc=ap_, calls_per_summary=cps) for s, b, au, ap_, cps in rows], open("cache/adaptive_results.json", "w"), indent=1)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for sch, c in (("random", "#8a8f98"), ("entropy", "#d9822b"), ("voi", "#1f5fa8")):
        r = [x for x in rows if x[0] == sch]; ax.plot([x[4] for x in r], [x[2] for x in r], "o-", color=c, label=sch)
    n = [x for x in rows if x[0] == "none"][0]; al = [x for x in rows if x[0] == "all"][0]
    ax.axhline(n[2], color="k", ls=":", lw=1, label="no judge"); ax.plot(al[4], al[2], "s", color="#2a8f6a", label="judge all")
    ax.set_xlabel("LLM judge calls per summary"); ax.set_ylabel("AUROC"); ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); fig.savefig("cache/adaptive_budget.png", dpi=300)
    print("saved cache/adaptive_budget.png")
except Exception as e:
    print("plot skipped:", e)
