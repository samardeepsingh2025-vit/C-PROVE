# C-PROVE — claim-level verification of meeting summaries (revision 2, tag `v2-frozen`)

Code, frozen configuration, labels and reproduction scripts for the paper. Everything reported in the paper was produced from
`configs/frozen.yaml` and the feature table `cache/frozen/features.jsonl` at git tag **`v2-frozen`**.

## Protocol
Nested leave-one-meeting-out cross-validation over 523 author-reviewed claims from 40 AMI meetings. Feature set, regularisation, Platt calibration and the
decision threshold are fitted inside each outer fold (`train/nested_cv.py`). No fresh test set of unseen meetings was collected, so results are
**development-informed**. Hand-weight rows use `hlmrf.init_weights` only; legacy dev-fitted artefacts (`calibrators.pkl`, `crc.json`, `rule_weights.json`) are not used
by any reported number (`python repro/check_leakage.py` verifies this).

## Reproduce
```
conda activate cprove ; pip install -r requirements-frozen.txt
python repro/make.py --list            # one target per table / figure
python repro/make.py check             # leakage check + unit tests
python repro/make.py build_features    # GPU + Ollama (qwen2.5:7b-instruct-q4_K_M), once; MiniCheck is resumable
python repro/make.py table_main        # Table 2, App. A ECE table, reliability figure (CPU, seconds)
```
| Paper item | Target | Output |
|---|---|---|
| Main table, ECE variants, reliability figure | `table_main` | `outputs/table_main.tex`, `table_ece_variants.tex`, `reliability.pdf` |
| Unverifiable-claim sensitivity | `table_unver` | `outputs/nested_cv_unver.json` |
| Independent gold-50, four-class check | `table_gold` | `outputs/gold_fusion_table.tex`, `four_class_gold.json` |
| NLI aggregation | `table_aggr` (after `nli_windows`) | `outputs/table_aggr.tex` |
| Retrieval | `retrieval_sweep`, `table_retrieval` | `outputs/` |
| Error injection | `robust_build` then `table_robust` | `outputs/table_robust.tex` |
| Component ladder | `table_ladder` | `outputs/table_ladder.tex` |
| Vetoes, graph, review/summary index, conformal | `veto`, `graph`, `review_mti`, `conformal` | `outputs/` |
| Decomposition | `decomposition` | `outputs/decomp_examples.tex`, `table_sentlevel.tex` |
| Efficiency | `efficiency` | `outputs/table_efficiency.tex` |

## Data
- `data/meetverify_v2/agreement/`: independent annotations `ann_B1.jsonl`, `ann_B2.jsonl`, adjudicated `gold_claims.jsonl`, agreement outputs; `bench/fresh/guidelines_v2.md`: annotation guidelines.
- Author-reviewed claim labels: `data/meetverify/annotations_A1.jsonl`. These were LLM-drafted and author-confirmed; 28% differed from the independent gold on the 50-claim overlap. They are **not** independent labels.
- AMI transcripts are not redistributed: obtain the corpus from the AMI project and convert with `cprove/data`.

## Not included / limitations
No fresh test set; AlignScore and a stronger LLM judge were not run; TofuEval was not rerun under the frozen protocol; graph-edge precision was not annotated;
no human summary-level ratings. See the paper's Threats to Validity.
