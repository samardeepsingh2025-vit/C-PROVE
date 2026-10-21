# SETUP AND RUN
## Environment (Windows 11 + RTX 5060 8 GB, or Linux)
```
python -m venv .venv && .venv\Scripts\activate          # source .venv/bin/activate on Linux
pip install torch --index-url https://download.pytorch.org/whl/cu128    # RTX 50-series needs CUDA 12.8+ builds (ASSUMPTION: check pytorch.org)
pip install -r requirements.txt && pip install -e .
```
llama-cpp-python with CUDA on Windows: use a prebuilt CUDA wheel matching your CUDA version (or build with `CMAKE_ARGS="-DGGML_CUDA=on"`); alternative with no compile: install Ollama and set `llm.backend: ollama`, or run `llama-server` and set `openai_compat`.
Models (not downloaded; place under models/): qwen2.5-7b-instruct-q4_k_m.gguf (~4.7 GB); optional Llama-3.1-8B and Mistral-7B-v0.3 Q4_K_M. HF models download on first use: BAAI/bge-base-en-v1.5, MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli, microsoft/deberta-v3-base.
Data under data/: ami_public_manual_1.6.2/ (AMI NXT) or QMSum/; TofuEval annotations for the external test.
## Checks that work today (CPU, no downloads)
```
python -m pytest -q tests
python scripts/smoke_test.py            # ~5-8 min on CPU
python -m cprove.cli verify --transcript t.txt --summary s.txt --config configs/smoke.yaml --scheduler none
```
## Real pipeline order
```
python -c "from cprove.utils import load_config; from cprove.data import load_corpus; print(len(load_corpus(load_config())[0]))"   # expect 142 for AMI
python -m bench.generate --split train --refs_only --out data/meetverify/summaries_train.jsonl   # also dev/test
python -m bench.generate --split test            # 3 systems, one model at a time
for sp in train dev test: python -m bench.inject --summaries data/meetverify/summaries_$sp.jsonl --meetings data/ami_meetings.jsonl --out data/meetverify/synth_$sp.jsonl
python -m train.label_silver --task status && python -m train.train_classifier --task status --data data/silver/status.jsonl --out models/status_clf
python -m train.label_silver --task edges --summaries data/meetverify/summaries_train.jsonl && python -m train.train_classifier --task edges --data data/silver/edges.jsonl --out models/edge_clf
python -m bench.annotate.prepare --summaries data/meetverify/summaries_test.jsonl ; python -m bench.annotate.app --annotator A1 ; python -m bench.annotate.agreement --files ... ; python -m bench.annotate.export --summaries ... --gold ...
python -m train.fit_cprove --train data/meetverify/synth_train.jsonl --dev data/meetverify/synth_dev.jsonl [--human_dev data/meetverify/human_dev.jsonl --ratings ...]
python -m baselines.run_baselines --bench data/meetverify/human_test.jsonl --systems nli_max factscore summac_zs alignscore minicheck llm_judge notebook_v5
python -m experiments.run all --bench data/meetverify/human_test.jsonl --dev data/meetverify/human_dev.jsonl
python -m demo.app
```
(Paths of the AMI meetings jsonl: load_corpus writes data/ami_meetings.jsonl on first load.)
## Troubleshooting
- CUDA OOM: `llm.n_gpu_layers: 24`, `n_ctx: 4096`, nli.batch_size 4; models are loaded one at a time via ModelSlot.
- Windows: keep dataloader_num_workers=0 (already set). Paths with double spaces: set CPROVE_ROOT.
- llama-cpp missing/GGUF missing -> silently falls back to heuristics (a message is printed); don't report such runs as LLM results.
- Stale caches: delete cache/items_*.pkl after changing frames/graph/dimension code.
