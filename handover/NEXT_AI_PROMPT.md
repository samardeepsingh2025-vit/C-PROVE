# PROMPT FOR THE NEXT AI (copy everything below)

You are continuing a research codebase called **C-PROVE** in the directory `cprove/` (the user is Hritik, M.Tech AI/ML, VIT Vellore). It verifies AI-generated **meeting summaries** against transcripts for a Q1-journal paper titled "A Graph-Based Neuro-Symbolic Adaptive Claim Verification Framework for Trustworthy Meeting Summarization". Stay on this topic and keep the original plan: atomic claims -> Claim Dependency Graph (before/causes/same_agent/task_depends/coref/revises/conflicts) -> six confidence dimensions (factual, attribution, temporal, contextual, semantic, reasoning) -> neuro-symbolic rules + local LLM -> graph propagation -> Meeting Trust Index (MTI).

## Read first (before changing anything)
`handover/PROJECT_STATUS.md`, `FILE_INVENTORY.md`, `ARCHITECTURE.md`, `REMAINING_WORK.md`, `SETUP_AND_RUN.md`, then open the actual source files you will touch. Inspect, don't assume. A backup tarball of the pre-audit state is at `../cprove_backup_20261020_1645.tar.gz`.

## State
Implemented (~5,700 lines): full library `cprove/` (frames, graph, evidence, nli, dimensions, rules, hlmrf, adaptive, judge, mti, conformal, explain, pipeline, cli), benchmark tools `bench/`, training `train/`, `baselines/`, `experiments/` (RQ1-RQ6 + ablations + bootstrap/McNemar/Holm), `demo/app.py`, tests.
VERIFIED: `python -m pytest -q tests` (10 pass) and `python scripts/smoke_test.py` (end-to-end on a TOY corpus with CPU fallbacks and a simulated judge).
NOT VERIFIED: everything involving real data or models (see PROJECT_STATUS "NOT verified"). Smoke-test numbers are NOT evidence the method works; do not cite them. Hardware target: RTX 5060 8 GB + 16 GB RAM, Windows; one heavy model resident at a time (utils.SLOT).

## Rules
- Reuse existing files; modify minimally; never delete data/outputs; save every change to disk; keep tests green (`pytest`, then smoke test) after each stage.
- Do not report results from heuristic/lexical/simulated fallbacks as real results. If llama-cpp/GGUF is missing the code silently falls back to heuristics — check `cp.llm.available`.
- Do not fabricate data, labels, kappa, or results. Annotation must be done by humans (tool in bench/annotate).
- No secrets in files.

## Order of work
1. Set up environment + models and run the sanity checks in SETUP_AND_RUN.md (GPU, llama.cpp, DeBERTa NLI, BGE).
2. Load real AMI (142 meetings); fix loaders against real files; add a fixture-based loader test.
3. Run Stage 1 with Qwen on real reference summaries; inspect and tune prompts/frames.
4. Generate summaries (`bench.generate`), build MeetVerify-Synth (`bench.inject`); eyeball injected errors, improve the entity injector.
5. Silver-label then train the status and edge classifiers (`train/`); evaluate against gold when available.
6. Human annotation -> agreement -> export human_dev/test.
7. `train.fit_cprove`; check calibration and learned lambda.
8. Baselines then `experiments.run all`; if ablations contradict the claims, fix the algorithm (rules, dimension definitions) rather than the narrative.
9. Run the real LLM judge in the adaptive loop; replace simulated RQ3.
10. TofuEval external test; collect 50-100 human 1-5 summary ratings for MTI validation; hallucination-report export and paper tables/figures.
Update `handover/REMAINING_WORK.md` and `PROJECT_STATUS.md` as you finish items.
