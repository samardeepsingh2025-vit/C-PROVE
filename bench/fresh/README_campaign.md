# Annotation campaign — run order

**Once (you, Hritik):**
```bat
python -m bench.fresh.select_meetings --n 22                         :: picks unused AMI meetings (seeded, logged)
ollama pull llama3.1:8b                                               :: second summariser (qwen2.5 is already there)
python -m bench.fresh.make_pack --systems qwen2.5:7b-instruct-q4_K_M llama3.1:8b
python -m bench.fresh.make_old_pack --n 150                           :: adds 150 old claims, blind, for agreement vs your labels
```
Give each friend ONLY: `data/meetverify_v2/pack.json`, `bench/fresh/guidelines_v2.md` (and the code to run the tool).
**Never** share `key_private.json`, `summaries.jsonl`, `claims.jsonl`, `annotations_A1.jsonl`, drafts or any results.
You must NOT annotate or look at the labels for these items before the friends finish.

**Friends (each, own computer or taking turns, ~15 h each):**
```bat
python -m bench.fresh.annotate_app --pack data/meetverify_v2/pack.json --annotator B1 --port 7861   (the other one: B2)
```
Browser opens at http://127.0.0.1:7861 . Output: `data/meetverify_v2/ann_B1.jsonl` (send it back when done).

**After both finish (you):**
```bat
python -m bench.fresh.agreement --a B1 --b B2 --author data\meetverify\annotations_A1.jsonl   :: kappa etc. -> paste output to me
python -m bench.fresh.agreement --adjudicate        :: someone neutral, or both friends together; one line reason each
python -m bench.fresh.agreement --export-gold       :: gold_claims.jsonl + gold_summaries.jsonl
```
