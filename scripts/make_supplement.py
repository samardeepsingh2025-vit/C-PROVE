"""Builds supplement.md: every prompt, the annotation guidelines and the adaptive-judge result, straight from the repo files.
  python scripts/make_supplement.py          -> supplement.md   (paste into the journal's supplementary file / convert with pandoc)"""
import ast, glob, json, pathlib, re
R = pathlib.Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(R))
out = ["# Supplementary material\n"]
def block(title, text): out.append(f"\n## {title}\n\n```text\n{text.strip()}\n```\n")
from cprove import prompts as P
out.append(f"Prompt version: `{P.PROMPT_VERSION}`. All prompts are used verbatim with Qwen2.5-7B-Instruct (Ollama, temperature 0).\n")
block("S1. Claim decomposition (system)", P.FRAME_SYSTEM); block("S1. Claim decomposition (user template)", P.FRAME_USER)
block("S2. Qwen zero-shot judge (system)", P.JUDGE_SYSTEM); block("S2. Qwen zero-shot judge (user template)", P.JUDGE_USER)
block("S3. Claim dependency edge prompt (system)", P.EDGE_SYSTEM)
block("S4. Summary generation (system)", P.SUMMARY_SYSTEM)
f = R / "train" / "status_check.py"
if f.exists():                                   # decision-status prompt: read the SYS string without running the script
    for n in ast.parse(f.read_text(encoding="utf-8")).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "SYS":
            block("S5. Decision-status check (system)", ast.literal_eval(n.value))
for g, t in (("bench/annotate/guidelines.md", "S6. Annotation guidelines (v1)"), ("bench/fresh/guidelines_v2.md", "S7. Annotation guidelines (v2)")):
    p = R / g
    if p.exists(): out.append(f"\n## {t}\n\n" + p.read_text(encoding="utf-8") + "\n")
for p in sorted(glob.glob(str(R / "outputs" / "adaptive*.json"))):
    out.append(f"\n## S8. Adaptive-judge result ({pathlib.Path(p).name})\n\n```json\n{json.dumps(json.load(open(p)), indent=1)[:6000]}\n```\n")
(R / "supplement.md").write_text("".join(out), encoding="utf-8"); print("wrote supplement.md,", sum(map(len, out)), "chars")
