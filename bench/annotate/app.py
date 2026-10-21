"""MeetVerify-Human annotation tool (Gradio, runs locally, no internet needed).

Each item = one atomic claim + its retrieved transcript window. Annotators give:
  label: SUPPORTED / UNSUPPORTED / UNVERIFIABLE
  error_type (if UNSUPPORTED): one of the 8 taxonomy types
  evidence utterance ids (comma-separated)
  for 20% of items: gold CDG edges to other claims of the same summary (for ablation 7)

  python -m bench.annotate.prepare --summaries data/meetverify/summaries_test.jsonl   # builds items.jsonl
  python -m bench.annotate.app --annotator A1                                           # opens http://127.0.0.1:7861
Annotations are appended to data/meetverify/annotations_<annotator>.jsonl (resumable).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ERRORS = ["entity", "negation", "attribution_swap", "time_shift", "revision_ignored", "status_flip",
          "dependency_cascade", "fabrication", "other"]


def main():
    import gradio as gr
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/meetverify/items.jsonl")
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--port", type=int, default=7861)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")]
    out = Path(a.items).parent / f"annotations_{a.annotator}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["cid"] for l in open(out, encoding="utf-8")}
    queue = [it for it in items if it["cid"] not in done]
    state = {"i": 0}

    def render():
        if state["i"] >= len(queue):
            return "**All done — thank you!**", "", "", f"{len(done)}/{len(items)}"
        it = queue[state["i"]]
        ctx = "\n".join(f"- {c}" for c in it.get("summary_context", []))
        return (f"### Claim\n**{it['text']}**\n\nFrame: `{it.get('frame')}`\n\nSummary context:\n{ctx}",
                it["evidence"], it.get("full_window", ""), f"{len(done) + state['i']}/{len(items)}  ·  {it['cid']}")

    def save(label, err, ev_ids, edges, note):
        if state["i"] >= len(queue):
            return render()
        it = queue[state["i"]]
        rec = {"cid": it["cid"], "summary_id": it["summary_id"], "annotator": a.annotator, "label": label,
               "error_type": err if label == "UNSUPPORTED" else "none",
               "evidence_uids": [x.strip() for x in (ev_ids or "").split(",") if x.strip()],
               "edges": edges or "", "note": note or ""}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        state["i"] += 1
        return render()

    with gr.Blocks(title="MeetVerify annotation") as demo:
        gr.Markdown("Read **guidelines.md** first. Judge the claim against the transcript only.")
        prog = gr.Markdown()
        claim = gr.Markdown()
        with gr.Row():
            ev = gr.Textbox(label="Retrieved evidence (ids in brackets)", lines=14)
            full = gr.Textbox(label="Wider transcript window", lines=14)
        label = gr.Radio(["SUPPORTED", "UNSUPPORTED", "UNVERIFIABLE"], label="Label")
        err = gr.Dropdown(ERRORS, label="Error type (if UNSUPPORTED)")
        ev_ids = gr.Textbox(label="Evidence utterance ids (comma separated)")
        edges = gr.Textbox(label="(optional) edges: e.g. 'revises:S03c0; task_depends:S05c1'")
        note = gr.Textbox(label="Note")
        btn = gr.Button("Save & next", variant="primary")
        btn.click(save, [label, err, ev_ids, edges, note], [claim, ev, full, prog])
        demo.load(render, None, [claim, ev, full, prog])
    demo.launch(server_port=a.port)


if __name__ == "__main__":
    main()
