# PROJECT STATUS — C-PROVE (meeting-summary claim verification)
Written 2026-10-20. Everything below was checked against files on disk in /home/claude/cprove unless marked ASSUMPTION.

## Objective
Verify AI-generated meeting summaries against the transcript: atomic claims with slot frames -> typed Claim Dependency Graph
-> six confidence dimensions -> neuro-symbolic hinge-loss MRF (rules + learned weights) -> VoI-adaptive local-LLM judge ->
labels (conformal abstention) + Meeting Trust Index (MTI) with a conformal interval. Target: Q1 journal.
Owner: Hritik (VIT Vellore). Direction fixed by the owner: stay faithful to the original plan (claims, CDG, 6 dimensions,
neuro-symbolic rules + LLM, propagation, MTI); the new algorithm is the unified HL-MRF + VoI scheduler, plus a labelled benchmark (MeetVerify).
Design blueprint: Claude Doc "C-PROVE: Project Blueprint" https://claude.ai/code/artifact/36101e52-d902-488e-8f26-c7c6870be9fd

## Hardware / environment
Target: Windows or Linux, 16 GB RAM, RTX 5060 Laptop 8 GB. This session ran in a Linux cloud container, CPU only, Python 3.13, torch 2.14.1+cu130 (CPU use),
no transformers / sentence-transformers / llama-cpp / gradio installed, no internet to model hubs, NO real data, NO GGUF models.

## Verified facts (tests actually run)
- `pytest tests` -> 10 passed (solver bounds + energy decrease, evidence vs no-evidence ordering, dependency propagation, judge sensitivity,
  lambda learning lowers loss, MTI formula, CRC risk control, time/sentence/transcript parsing, pipeline faithful>corrupted, injection sanity).
- `python scripts/smoke_test.py` -> SMOKE TEST PASSED (full path on a synthetic 12-meeting toy corpus with fallback scorers:
  injection -> fit (calibrators, lambda, CRC, MTI conformal) -> baselines -> RQ1,RQ2,ablate,RQ3,RQ4,RQ5,RQ6).
- `python -m cprove.cli verify ...` works on the toy meeting.

## NOT verified (important, do not over-claim)
- Smoke numbers (e.g. C-PROVE AUROC 0.90 vs baselines 0.60-0.75) come from a TOY corpus whose text matches the author's own lexicons and from
  fallback components (heuristic frames, lexical NLI, status cues, SIMULATED judge with 15% noise). They show the code runs, NOT that the method works.
  Ablations on toy data were inconsistent with the paper's hypotheses (e.g. removing graph rules did not hurt). Expect to re-tune on real data.
- Never executed: llama.cpp/Ollama backends, HFNLI (DeBERTa), MiniCheck, BGE dense retrieval, StatusModel/EdgeClassifier fine-tuned checkpoints,
  all of train/train_classifier.py, train/label_silver.py, train/qlora_frames.py, AMI/QMSum/TofuEval loaders, bench/generate.py,
  annotation app/prepare/agreement/export, baselines summac/alignscore/minicheck, demo/app.py, GPU paths, Whisper.
- Real AMI data has not been loaded; loaders were written from the AMI NXT/QMSum formats from memory (ASSUMPTION about XML details).

## Known issues / design caveats
1. Heuristic fallbacks are weak: status cues, attribution, contextual dimension are rule-based unless the fine-tuned classifiers exist.
2. `ground()` keeps `R1b` contradiction hinge and R1 `ent->fact`; rule family weights are shared across all groundings of a family.
3. Dose-response (RQ4) re-injects errors with `inject_one` and may yield fewer errors than requested; `true_frac` records the actual fraction.
4. RQ5 deletion test is approximate (matches proof groundings by `kind`).
5. `experiments/run.py` thresholds: dev threshold is looked up from `outputs/preds/<dev>__<system>.jsonl`; baselines need dev preds too (smoke test does this).
6. Human-labelled MeetVerify-Human does not exist yet (needs annotation, ~1,500 claims, kappa>=0.6).
7. pyproject.toml now has dependencies; `pip install -e .` untested.
8. A tar backup exists at /home/claude/cprove_backup_20261020_1645.tar.gz (taken before the audit changes).

## Differences from the original proposal (disclosed to the owner earlier)
Propagation and rules merged into one HL-MRF solve; contextual/semantic/reasoning dimensions redefined (latest-state, status, graph-posterior);
headline MTI is a weighted mean of claim posteriors with the original five components kept as diagnostics (+ optional learned MTI-5).
