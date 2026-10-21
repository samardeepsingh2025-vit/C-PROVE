# REMAINING WORK (priority order; E = essential, O = optional)
1. E — Environment: install real deps + GPU torch, GGUF + HF models. Verify `llm.available`, HFNLI, BGE load on the 8 GB GPU (peak VRAM per stage). Files: configs/default.yaml. Test: tiny run on 2 AMI meetings.
2. E — Real data: get AMI NXT (or QMSum), run load_corpus, check 142 meetings, utterance counts, merge_short effect, series-aware split 99/21/22. Fix loader bugs (U). Test: print stats; unit test on a tiny fixture XML.
3. E — Stage-1 quality on real summaries: inspect 50 LLM-decomposed sentences; tune prompt/few-shots; measure claim-count and slot fill. Files: prompts.py, frames.py.
4. E — Generate summaries (3 systems) + MeetVerify-Synth injection on AMI reference abstracts; check injected-error sanity by eye (inject.py entity injector is crude: picks random nouns).
5. E — Train status classifier (silver->gold) and edge classifier; report macro-F1 vs gold. Then wire (paths already in config) and re-run.
6. E — Annotate MeetVerify-Human (~1,500 claims, 15% double), kappa>=0.6, adjudicate, export dev/test. 
7. E — fit_cprove on real data; check calibration (ECE), learned lambda sanity, CRC coverage.
8. E — Run baselines (SummaC, AlignScore, MiniCheck need installs) and `experiments.run all`; examine whether each ablation hypothesis holds; if not, fix the rules (not the reporting).
9. E — Real adaptive-judge run with Qwen (LLMJudge never executed): check JSON validity rate, calibration of faithful_prob, latency; replace --simulate_judge results.
10. E — TofuEval external test (bench/prepare_tofueval.py untested), human MTI ratings (50-100 summaries) for RQ4.
11. O — QLoRA frame distillation (train/qlora_frames.py), Whisper demo, Label Studio path, ICSI.
12. O — More unit tests for loaders/dimensions with real-format fixtures; CI; logging (currently print-based).
13. E for paper — hallucination report generation (per-summary report from Result; CLI prints verdicts and proofs but no saved report), paper tables/figures export (CSV/MD/PDF exist for RQ3 only).
Dependencies: 1->2->3->4->5,6->7->8->9->10.
