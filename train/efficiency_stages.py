"""E14 (fix): measure stage timings directly on N summaries (the old log was empty because the items were cached). GPU, Ollama must be running.
  python -m train.efficiency_stages --n 12 --mc_s 4.42
Times claim decomposition (LLM via Ollama), edge construction, retrieval+NLI+attribution/status/temporal/context features (in-process, peak GPU memory
via torch), and the HL-MRF/fusion inference. MiniCheck seconds per claim can be passed with --mc_s (from train.efficiency --minicheck).
All disk caches are bypassed for reading. -> outputs/table_efficiency.tex"""
import argparse, json, platform, time
import numpy as np, torch
from cprove.utils import load_config, seed_everything
from cprove.data import load_corpus
from cprove.pipeline import CProve, Ablation
from cprove.dimensions import Calibrators
from cprove.hlmrf import HLMRF, Weights
from cprove.rules import ground, order_consistency
from experiments.common import load_records
from cprove import utils as _u
_u.JsonCache.get = lambda self, key: None      # disable ALL disk caches (LLM, NLI) so the timings are real compute, not cache hits

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/frozen.yaml"); ap.add_argument("--records", default="data/meetverify/qwen_test.jsonl")
ap.add_argument("--n", type=int, default=12); ap.add_argument("--mc_s", type=float, default=None); a = ap.parse_args()
cfg = load_config(a.config); seed_everything(cfg["seed"]); meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
recs = [r for r in load_records(a.records) if r.get("system") != "reference"][: a.n]
cp = CProve(cfg, Ablation(hand_weights=True)); cp.dims.cal = Calibrators("none")
h = cfg["hlmrf"]; model = HLMRF(Weights.load(h.get("learned_weights"), h["init_weights"]), h["iters"], h["tol"])
first = recs[0]; cp.decompose(meetings[first["meeting_id"]], first.get("summary") or " ".join(first["sentences"]), "warm")   # warm-up (model load)
T = {"dec": [], "edge": [], "obs": [], "inf": []}; nc = []; peak = 0.0
for r in recs:
    m = meetings[r["meeting_id"]]; text = r.get("summary") or " ".join(r["sentences"])
    t0 = time.perf_counter(); claims = cp.decompose(m, text, r["summary_id"]); t1 = time.perf_counter()
    edges = cp.edges_for(claims, r["summary_id"]); t2 = time.perf_counter()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    _, _, obs, grounding = cp.observe(m, claims); torch.cuda.synchronize(); t3 = time.perf_counter(); peak = max(peak, torch.cuda.max_memory_allocated() / 2**20)
    ok = order_consistency(claims, edges, grounding); pr = ground(claims, obs, edges, ok, ()); model.infer(pr); t4 = time.perf_counter()
    for k, v in zip(T, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)): T[k].append(v)
    nc.append(len(claims))
nc = np.array(nc); print(f"{len(recs)} summaries, {nc.mean():.1f} claims per summary")
names = {"dec": "Claim decomposition (Qwen2.5-7B via Ollama)", "edge": "Edge construction", "obs": "Retrieval, NLI, attribution/status/temporal/context features", "inf": "HL-MRF inference (hand weights)"}
rows = [(names[k], np.mean(v), np.mean(np.array(v) / nc)) for k, v in T.items()]
tot = np.sum([T[k] for k in T], axis=0); rows.append(("Total without MiniCheck", tot.mean(), (tot / nc).mean()))
for r in rows: print(f"{r[0]:62}{r[1]:>8.2f} s/summary {r[2]:>7.2f} s/claim")
print(f"peak GPU memory allocated in-process during retrieval+NLI+features: {peak:.0f} MB")
env = {"python": platform.python_version(), "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}
import transformers; env["transformers"] = transformers.__version__; print(env)
L = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule", r"Stage & s/summary & s/claim \\", r"\midrule"] + [f"{r[0]} & {r[1]:.2f} & {r[2]:.2f} \\\\" for r in rows]
if a.mc_s: L.append(f"MiniCheck (flan-t5-large, full transcript) & {a.mc_s * nc.mean():.1f} & {a.mc_s:.2f} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]; open("outputs/table_efficiency.tex", "w").write("\n".join(L)); print("wrote outputs/table_efficiency.tex")
