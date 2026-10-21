"""Review pre-filled draft labels (LLM-assisted annotation). Same output format as app.py, so
export/agreement work unchanged. You CONFIRM or CORRECT each draft; the draft is only a starting point.

  python -m bench.annotate.review --items data\\meetverify\\items_pilot.jsonl --drafts data\\meetverify\\pilot_drafts.jsonl --annotator A1

Low-confidence drafts are shown first (those are the ones most likely to be wrong).
Output: annotations_<annotator>.jsonl next to the items, with extra fields draft_label / draft_conf / changed
so you can report how often the human changed the AI draft.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ERRORS = ["entity", "negation", "attribution_swap", "time_shift", "revision_ignored", "status_flip",
          "dependency_cascade", "fabrication", "other"]
CONF_ORDER = {"low": 0, "med": 1, "high": 2}


def build_queue(items, drafts, done):
    by = {d["cid"]: d for d in drafts}
    q = [it for it in items if it["cid"] not in done]
    q.sort(key=lambda it: CONF_ORDER.get(by.get(it["cid"], {}).get("conf", "low"), 0))
    return q, by


def main():
    import gradio as gr
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--drafts", required=True)
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--port", type=int, default=7861)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items, encoding="utf-8")]
    drafts = [json.loads(l) for l in open(a.drafts, encoding="utf-8")]
    out = Path(a.items).parent / f"annotations_{a.annotator}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["cid"] for l in open(out, encoding="utf-8")}
    queue, by = build_queue(items, drafts, done)
    state = {"i": 0}

    def render():
        if state["i"] >= len(queue):
            return "**All done — thank you!**", "", "", f"{len(done) + state['i']}/{len(items)}", None, None, "", ""
        it = queue[state["i"]]
        d = by.get(it["cid"], {})
        ctx = "\n".join(f"- {c}" for c in it.get("summary_context", []))
        head = (f"### Claim\n**{it['text']}**\n\nFrame: `{it.get('frame')}`\n\n"
                f"**AI draft: {d.get('label', '?')}"
                f"{' / ' + d['error_type'] if d.get('error_type', 'none') != 'none' else ''}"
                f"  (confidence: {d.get('conf', '?')})** — {d.get('note', '')}\n\nSummary context:\n{ctx}")
        et = d.get("error_type") if d.get("label") == "UNSUPPORTED" else None
        return (head, it["evidence"], it.get("full_window", ""),
                f"{len(done) + state['i']}/{len(items)}  ·  {it['cid']}",
                d.get("label"), et, ", ".join(d.get("evidence_uids", [])), "")

    def save(label, err, ev_ids, edges, note):
        if state["i"] >= len(queue):
            return render()
        it = queue[state["i"]]
        d = by.get(it["cid"], {})
        rec = {"cid": it["cid"], "summary_id": it["summary_id"], "annotator": a.annotator, "label": label,
               "error_type": err if label == "UNSUPPORTED" else "none",
               "evidence_uids": [x.strip() for x in (ev_ids or "").split(",") if x.strip()],
               "edges": edges or "", "note": note or "",
               "draft_label": d.get("label"), "draft_error_type": d.get("error_type"), "draft_conf": d.get("conf"),
               "changed": label != d.get("label") or (label == "UNSUPPORTED" and err != d.get("error_type"))}
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        state["i"] += 1
        return render()

    with gr.Blocks(title="MeetVerify review") as demo:
        gr.Markdown("**Review mode.** A draft label is pre-filled. Check it against the evidence; change it if it is wrong. "
                    "Guidelines: `bench/annotate/guidelines.md`. Low-confidence drafts come first.")
        prog = gr.Markdown()
        claim = gr.Markdown()
        with gr.Row():
            ev = gr.Textbox(label="Retrieved evidence (ids in brackets)", lines=14)
            full = gr.Textbox(label="Wider transcript window", lines=14)
        label = gr.Radio(["SUPPORTED", "UNSUPPORTED", "UNVERIFIABLE"], label="Label")
        err = gr.Dropdown(ERRORS, label="Error type (if UNSUPPORTED)")
        ev_ids = gr.Textbox(label="Evidence utterance ids (comma separated)")
        edges = gr.Textbox(label="(optional) edges")
        note = gr.Textbox(label="Note")
        btn = gr.Button("Save & next", variant="primary")
        outs = [claim, ev, full, prog, label, err, ev_ids, note]
        btn.click(save, [label, err, ev_ids, edges, note], outs)
        demo.load(render, None, outs)
    demo.launch(server_port=a.port, inbrowser=True)


if __name__ == "__main__":
    main()
