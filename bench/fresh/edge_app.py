"""Blind edge-validation tool. For each pair of claims from the same summary decide whether the stated relation holds.

  python -m bench.fresh.edge_app --pack data/meetverify_v2/edges/edge_pack.json --annotator B1
"""
import argparse, json, time
from pathlib import Path

DEFS = {"before": "A happened BEFORE B in the meeting (a real temporal order, not just the order of the summary sentences).",
        "causes": "A is a REASON / cause for B (B happened or was decided because of A).",
        "same_agent": "A and B are about the SAME person or role doing something.",
        "task_depends": "B (a task/decision) can only be done or makes sense AFTER / given A.",
        "coref": "A and B state the SAME thing (duplicates / paraphrases).",
        "revises": "B REVISES, changes or replaces what A said (a later decision updating an earlier one).",
        "conflicts": "A and B CANNOT both be true at the same time."}
OPTS = ["HOLDS (real relation)", "ONLY SIMILAR WORDS / NEIGHBOURING SENTENCES (no real relation)", "DOES NOT HOLD", "UNCLEAR"]


def main():
    import gradio as gr
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/meetverify_v2/edges/edge_pack.json"); ap.add_argument("--annotator", required=True)
    ap.add_argument("--port", type=int, default=7871)
    a = ap.parse_args()
    pack = json.load(open(a.pack, encoding="utf-8")); out = Path(a.pack).parent / f"edges_{a.annotator}.jsonl"
    done = {json.loads(l)["edge_id"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    q = [e for e in pack if e["edge_id"] not in done]

    def view(msg=""):
        if not q:
            return "## All done - thank you!", "", "", msg, None
        e = q[0]
        return (f"### {e['edge_id']}  ·  relation: **{e['type']}**\n\n*{DEFS[e['type']]}*", f"**Claim A:** {e['a']}", f"**Claim B:** {e['b']}",
                f"{len(done)}/{len(pack)} done {msg}", None)

    def save(v):
        if not q or not v:
            return view("⚠ choose an option" if q else "")
        e = q.pop(0); done.add(e["edge_id"])
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"edge_id": e["edge_id"], "annotator": a.annotator, "type": e["type"], "verdict": v.split(" ")[0], "ts": time.strftime("%F %T")}) + "\n")
        return view()

    with gr.Blocks(title="Edge validation") as demo:
        head = gr.Markdown(); A = gr.Markdown(); B = gr.Markdown(); prog = gr.Markdown()
        v = gr.Radio(OPTS, label="Does the relation hold between A and B? (A comes first in the relation 'A -> B')")
        btn = gr.Button("Save & next", variant="primary")
        outs = [head, A, B, prog, v]
        demo.load(view, None, outs); btn.click(save, [v], outs)
    demo.launch(server_port=a.port)


if __name__ == "__main__":
    main()
