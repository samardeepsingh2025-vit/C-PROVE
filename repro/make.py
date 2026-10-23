"""One entry point per table / figure of the paper, all reading configs/frozen.yaml and cache/frozen/features.jsonl.
  python repro/make.py --list
  python repro/make.py table_main            # CPU targets run in seconds
GPU/heavy steps (build once): see 'build' targets. Windows: run from the repo root inside the conda env."""
import subprocess, sys
T = {
 # --- GPU / Ollama steps, run once (outputs are cached) ---
 "build_features":  (["python -m train.build_features --config configs/frozen.yaml --sets dev=data/meetverify/qwen_dev20.jsonl test=data/meetverify/qwen_test.jsonl --ann data/meetverify/annotations_A1.jsonl --out cache/frozen/features.jsonl"], "cache/frozen/features.jsonl"),
 "baselines":       (["python -m train.baselines_extra --which summac", "python -m train.llm_judge"], "cache/frozen/baseline_*.json"),
 "nli_windows":     (["python -m train.nli_windows"], "cache/frozen/nli_windows.json"),
 "robust_build":    (["python -m train.robustness --stage build"], "cache/frozen/robust.jsonl"),
 "retrieval_sweep": (["python -m train.retrieval_sweep"], "outputs/retrieval_sweep.*"),
 # --- CPU targets: tables and figures ---
 "table_main":      (["python -m train.nested_cv", "python -m train.full_metrics"], "outputs/table_main.tex  (Table 2), table_ece_variants.tex (App. A), reliability.pdf (Fig.)"),
 "table_unver":     (["python -m train.nested_cv --unverifiable_as_unsupported --out outputs/nested_cv_unver.json"], "outputs/nested_cv_unver.json (unverifiable sensitivity)"),
 "table_gold":      (["python -m train.gold_eval", "python -m train.gold_fusion_eval", "python -m train.four_class_gold"], "outputs/gold_fusion_table.tex, four_class_gold.json"),
 "table_aggr":      (["python -m train.nli_aggregation"], "outputs/table_aggr.tex"),
 "table_retrieval": (["python -m train.eval_retrieval"], "outputs/retrieval metrics"),
 "table_robust":    (["python -m train.robustness --stage eval"], "outputs/table_robust.tex  (injection table)"),
 "table_ladder":    (["python -m train.ladder"], "outputs/table_ladder.tex"),
 "veto":            (["python -m train.veto_eval", "python -m train.veto_sweep"], "outputs/veto_eval.json, table_vetosweep.tex"),
 "graph":           (["python -m train.graph_eval", "python -m train.graph_diag"], "outputs/graph_eval.json, table_edges.tex, table_graph_sens.tex"),
 "review_mti":      (["python -m train.mti_eval"], "outputs/mti_eval.json (review prioritisation + summary index)"),
 "conformal":       (["python -m train.conformal_eval"], "outputs/conformal_eval.json"),
 "decomposition":   (["python -m train.decomp_stats", "python -m train.sentence_baseline --stage eval", "python -m train.decomp_match"], "outputs/decomp_examples.tex, table_sentlevel.tex"),
 "efficiency":      (["python -m train.efficiency_stages --n 12 --mc_s 4.42"], "outputs/table_efficiency.tex"),
 "check":           (["python repro/check_leakage.py", "python -m pytest -q tests"], "leakage check + unit tests"),
}
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        for k, (_, o) in T.items(): print(f"{k:16} -> {o}")
        sys.exit(0)
    for t in sys.argv[1:]:
        for c in T[t][0]:
            print(">>", c, flush=True)
            if subprocess.call(c, shell=True): sys.exit(f"failed: {c}")
