# C-PROVE
Graph-based neuro-symbolic claim verification for trustworthy meeting summarization (AMI).

## Setup
```bash
conda create -n cprove python=3.10 && conda activate cprove
pip install -r requirements.txt && pip install -e .
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
# claim decomposition LLM
ollama serve   # separate terminal
ollama pull qwen2.5:7b
```
Download the AMI corpus (manual annotations 1.6.2) from https://groups.inf.ed.ac.uk/ami/download/ into `data/ami_public_manual_1.6.2/`.

## Reproduce results (run from repo root)
```bash
# 1. fit fusion on dev, evaluate on held-out test (n=379)
python -m train.final_eval --ann data\meetverify\annotations_A1.jsonl --dev cache\items_qwen_dev20.pkl --test data\meetverify\qwen_test.jsonl
# 2. external baselines (MiniCheck, AlignScore)
python -m train.baselines_real
# 3. MiniCheck-feature fusion + leave-one-meeting-out
python -m train.fuse_mc
# 4. external check on TofuEval (git clone https://github.com/amazon-science/tofueval ..\tofueval)
python -m train.tofueval
# 5. adaptive LLM-judge budget experiment
python -m train.adaptive_real
```
Results are summarised in `results_real.md`.

## Demo
```bash
ollama serve
python -m demo.app
```
Example input/output: `demo_example.txt`.

## Labels
`data/meetverify/` holds claim labels (LLM-drafted, fully human-verified by one annotator; see paper for protocol and limitations).

## Not included
Raw AMI corpus, model weights/caches, and `cache/` (regenerated on first run).
