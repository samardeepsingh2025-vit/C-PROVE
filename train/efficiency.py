"""E14 efficiency report (CPU, seconds). python -m train.efficiency
Reads the per-summary stage timings logged by build_features (cache/frozen/stage_timing.jsonl), reports mean/median seconds per summary
and per claim for each stage, plus hardware and software versions. -> outputs/table_efficiency.tex"""
import argparse, json, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--minicheck", type=int, default=0, help="time MiniCheck on N labelled claims and record peak GPU memory (GPU)")
ap.add_argument("--features", default="cache/frozen/features.jsonl"); ap.add_argument("--config", default="configs/frozen.yaml"); a = ap.parse_args()
f = Path("cache/frozen/stage_timing.jsonl")
T = [json.loads(l) for l in open(f)] if f.exists() else []
print(f"{len(T)} logged summaries (note: the log is appended across runs; delete duplicates if you rebuilt)")
claims = np.array([t["claims"] for t in T]); rows = []
for k, nm in (("frames_s", "Claim decomposition (LLM)"), ("graph_s", "Edge construction"), ("evidence_nli_dims_s", "Retrieval, NLI and features")):
    v = np.array([t[k] for t in T]); rows.append((nm, v.mean(), np.median(v), (v / np.maximum(claims, 1)).mean()))
tot = np.array([t["frames_s"] + t["graph_s"] + t["evidence_nli_dims_s"] for t in T]); rows.append(("Total (without MiniCheck)", tot.mean(), np.median(tot), (tot / np.maximum(claims, 1)).mean()))
print(f"{'stage':32}{'mean s/summary':>16}{'median':>9}{'s/claim':>9}")
for r in rows: print(f"{r[0]:32}{r[1]:>16.2f}{r[2]:>9.2f}{r[3]:>9.2f}")
print(f"claims per summary: mean {claims.mean():.1f}")
env = {"python": sys.version.split()[0], "platform": platform.platform()}
try:
    import torch, transformers
    env.update(torch=torch.__version__, transformers=transformers.__version__)
    if torch.cuda.is_available(): env.update(gpu=torch.cuda.get_device_name(0), vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
except Exception as e: env["torch"] = f"n/a ({e})"
print(env)
mcrow = None
if a.minicheck:
    import torch
    from cprove.utils import load_config
    from cprove.data import load_corpus
    from baselines.run_baselines import transcript_text
    from minicheck.minicheck import MiniCheck
    meet = {m.meeting_id: m for m in load_corpus(load_config(a.config))[0]}
    cl, seen = [], set()
    for l in open(a.features, encoding="utf-8"):
        r = json.loads(l)
        if r["cid"] not in seen and r.get("author_label") in ("SUPPORTED", "UNSUPPORTED"): seen.add(r["cid"]); cl.append(r)
    cl = cl[: a.minicheck]; mc = MiniCheck(model_name="flan-t5-large", cache_dir="models/hf")
    torch.cuda.reset_peak_memory_stats(); t0 = time.perf_counter(); n = 0
    for r in cl:
        mc.score(docs=[transcript_text(meet[r["meeting_id"]])], claims=[r["text"]]); n += 1
    dt = (time.perf_counter() - t0) / n; peak = torch.cuda.max_memory_allocated() / 2**20
    mcrow = ("MiniCheck (flan-t5-large, full transcript, chunked)", dt, peak)
    print(f"MiniCheck: {dt:.2f} s/claim over {n} claims, peak GPU memory allocated by this process {peak:.0f} MB")
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
except Exception: pass
L = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule", r"Stage & Mean s/summary & Median & s/claim \\", r"\midrule"]
L += [f"{r[0]} & {r[1]:.2f} & {r[2]:.2f} & {r[3]:.2f} \\\\" for r in rows] + ([f"{mcrow[0]} & -- & -- & {mcrow[1]:.2f} \\\\"] if mcrow else []) + [r"\bottomrule", r"\end{tabular}"]
Path("outputs").mkdir(exist_ok=True); open("outputs/table_efficiency.tex", "w").write("\n".join(L)); print("wrote outputs/table_efficiency.tex")
