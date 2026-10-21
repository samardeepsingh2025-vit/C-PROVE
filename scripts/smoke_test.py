"""End-to-end CPU smoke test of the whole research pipeline on a synthetic toy corpus.
No downloads, no GPU: heuristic frames, lexical NLI, simulated judge.

  python scripts/smoke_test.py
It exercises: corpus I/O -> MeetVerify-Synth injection -> stages 1-4 -> calibration ->
lambda learning -> CRC -> MTI conformal -> RQ1/RQ2/ablations/RQ3/RQ4/RQ5/RQ6 -> baselines.
Numbers are NOT meaningful (toy data, fallback scorers); it only proves the code path works.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tests.fixtures import toy_corpus  # noqa: E402
from cprove.data import save_meetings_jsonl, meeting_split  # noqa: E402

CFG = "configs/smoke.yaml"
B = Path("data/smoke")
for d in ("cache/smoke", "models/smoke", "outputs/smoke"):
    shutil.rmtree(d, ignore_errors=True)
B.mkdir(parents=True, exist_ok=True)

meetings, sums = toy_corpus(12)
save_meetings_jsonl(meetings, "data/toy_meetings.jsonl")
split = meeting_split([m.meeting_id for m in meetings], (0.5, 0.25, 0.25), 42, by_series=False)
for sp, ids in split.items():
    with open(B / f"summaries_{sp}.jsonl", "w") as f:
        for s in sums:
            if s["meeting_id"] in ids:
                f.write(json.dumps(s) + "\n")


def run(*args):
    cmd = [sys.executable, *args]
    print("\n$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=str(ROOT)))
    if r.returncode:
        sys.exit(f"FAILED: {' '.join(args)}")


for sp in ("train", "dev", "test"):
    run("-m", "bench.inject", "--summaries", f"{B}/summaries_{sp}.jsonl", "--meetings", "data/toy_meetings.jsonl",
        "--out", f"{B}/synth_{sp}.jsonl", "--per_type", "2", "--seed", str(len(sp)))
run("-m", "train.fit_cprove", "--config", CFG, "--train", f"{B}/synth_train.jsonl", "--dev", f"{B}/synth_dev.jsonl", "--epochs", "6")
for sp in ("dev", "test"):
    run("-m", "baselines.run_baselines", "--config", CFG, "--bench", f"{B}/synth_{sp}.jsonl",
        "--systems", "nli_max", "factscore", "notebook_v5")
common = ["--config", CFG, "--bench", f"{B}/synth_test.jsonl", "--dev", f"{B}/synth_dev.jsonl", "--simulate_judge",
          "--bootstrap", "300", "--max_dose", "6", "--max_rq5", "6"]
for what in ("rq1", "rq2", "ablate", "rq3", "rq4", "rq5", "rq6"):
    run("-m", "experiments.run", what, *common)
print("\nSMOKE TEST PASSED")
