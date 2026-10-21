# C-PROVE: real-data results (human-verified; dev n=144, held-out test n=379, 20 meetings)

Labels: LLM-drafted, fully human-verified by the author (all items checked; no label changed on review).
Fusion fitted on dev only; test never used for fitting. 95% bootstrap CIs (1000 resamples).

| Method | AUROC [95% CI] | AUPRC (unsupported) | ECE |
|---|---|---|---|
| C-PROVE, hand weights + graph | 0.743 [0.687, 0.797] | 0.588 | 0.241 |
| Entailment only (DeBERTa NLI, retrieved windows) | 0.740 [0.686, 0.792] | 0.518 | 0.181 |
| MiniCheck (Flan-T5-L, full transcript) | 0.758 [0.704, 0.809] | 0.582 | n/a |
| C-PROVE fusion: ent+con | 0.738 [0.683, 0.790] | 0.537 | 0.107 |
| C-PROVE fusion: ent+con+attr | 0.767 [0.713, 0.821] | 0.621 | 0.102 |
| C-PROVE fusion: ent+con+attr+sem | 0.772 [0.720, 0.825] | 0.621 | 0.098 |
| **C-PROVE fusion + MiniCheck feature** | **0.784 [0.730, 0.836]** | **0.656** | n/a (not yet measured) |

Claims for the paper (honest framing)
- Discrimination comparable to MiniCheck (CIs overlap); best AUPRC on unsupported claims.
- Calibration clearly better than hand-weighted and entailment-only scores (ECE 0.098 vs 0.181-0.241).
- Attribution feature is the main gain over entailment (+0.03 AUROC); the claim dependency graph added nothing on real data.
- Hard vetoes (unseen number, strong contradiction) add auditable proof traces.
- SummaC-ZS could not be run (incompatible with current transformers); entailment-only is the closest sentence-NLI analogue.
Limitations: single annotator, LLM-assisted labels, one summarizer (Qwen2.5-7B), AMI only, CI overlap.

Leave-one-meeting-out over all 523 labelled claims: MiniCheck 0.739; C-PROVE features 0.794; C-PROVE + MiniCheck 0.809.

External check (TofuEval-MeetingBank test, n=777 sentences, no speakers so attribution unavailable, fusion trained on AMI only):
| Method | AUROC [95% CI] | AUPRC (inconsistent) |
|---|---|---|
| Entailment only | 0.766 [0.723, 0.807] | 0.465 |
| C-PROVE fusion (ent, con, sem) | 0.771 [0.725, 0.809] | 0.478 |
| MiniCheck | 0.849 [0.813, 0.885] | 0.668 |
Honest reading: on speaker-free monologue-style transcripts a strong fact-checker (MiniCheck) wins on raw discrimination; C-PROVE's advantage is on multi-party meetings where who-said/decided-what matters, plus calibration and auditability.
