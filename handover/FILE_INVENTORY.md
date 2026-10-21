# FILE INVENTORY (status: V = verified by a test that ran, U = code exists, never executed on real inputs/models, P = partial)
## Core library `cprove/`
| file | purpose | status |
|---|---|---|
| schema.py | dataclasses Utterance, Meeting, Claim, Frame, Edge, Evidence, Observation, Verdict; jsonl helpers | V |
| utils.py | config loading (paths made absolute), seeding, ModelSlot (one heavy model at a time), JsonCache | V (config/cache), U (GPU free) |
| llm.py | LLM backends llamacpp/ollama/openai_compat/heuristic, JSON-schema output, cache | V heuristic only; U others |
| prompts.py | all prompts+JSON schemas (frames, edges, judge, summary, fabrication), PROMPT_VERSION | U |
| text.py | sentence split, status cues (negation-aware), quantities, time normalisation/compat | V partly (tests) |
| frames.py | Stage 1 claim decomposition (LLM or heuristic) | V heuristic; U LLM |
| graph.py | Stage 2 typed CDG rules + optional DeBERTa EdgeClassifier | V rules; U classifier |
| evidence.py | Stage 3 slot-aware hybrid retrieval (BGE+TFIDF) | V TF-IDF fallback; U BGE |
| nli.py | Stage 3b windows->3-way NLI mixture; LexicalNLI, HFNLI, MiniCheck | V lexical; U others |
| dimensions.py | Stage 4 six dimensions, StatusModel, Calibrators | V with fallbacks; U fine-tuned status model |
| rules.py | Stage 5a grounding R1..R10b,RJ | V |
| hlmrf.py | Stage 5b solver, lambda learning, judge sensitivity | V (tests) |
| adaptive.py | Stage 5c VoI/entropy/random/all schedulers | V with simulated judge; U with real LLM |
| judge.py | LLMJudge + SimulatedJudge | U (LLMJudge); V (simulated) |
| mti.py | MTI, 5 components, MTI5 NNLS, ConformalInterval | V formula; U validity vs humans |
| conformal.py | CRC abstention labeler | V |
| explain.py | proof traces | V (smoke) |
| pipeline.py | CProve end-to-end class, Ablation switches | V (toy) |
| io.py, cli.py | transcript parsing, `cprove verify` CLI | V (toy) |
| data/loaders.py, data/splits.py | AMI NXT / QMSum / TofuEval / csv / jsonl loaders; series-aware splits | U (splits V via smoke) |
## Benchmark `bench/`
inject.py (MeetVerify-Synth, V on toy), generate.py (summaries with 3 local LLMs, U), prepare_tofueval.py (U),
annotate/{prepare,app,agreement,export}.py + guidelines.md + label_studio.xml (U; guidelines complete).
## Training `train/`
label_silver.py, train_classifier.py (status+edge DeBERTa), qlora_frames.py (optional) — all U. fit_cprove.py (calibrators, lambda, CRC, MTI) — V on toy.
## Experiments / baselines
experiments/{common,stats,run}.py — V on toy (all RQs). baselines/run_baselines.py — nli_max, factscore, notebook_v5 V on toy; summac/alignscore/minicheck/llm_judge U.
## Other
configs/default.yaml (real run), configs/smoke.yaml (CPU), scripts/smoke_test.py, tests/ (fixtures + 10 tests), demo/app.py (U), requirements.txt, pyproject.toml.
Generated (do not hand-edit): data/smoke/*, data/toy_meetings.jsonl, cache/smoke/*, models/smoke/*, outputs/smoke/*.
Not present: any real dataset, GGUF/HF model weights, fine-tuned checkpoints, human annotations.
