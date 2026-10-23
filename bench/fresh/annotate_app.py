"""BLIND annotation tool v2 (Gradio, local).  Annotators see the FULL transcript, the summary and the claims.
They do NOT see model scores, LLM drafts, retrieved evidence or the summariser's name.

  python -m bench.fresh.annotate_app --pack data/meetverify_v2/pack.json --annotator B1 --port 7861
Output (resumable, append-only): data/meetverify_v2/ann_<annotator>.jsonl
"""
from __future__ import annotations
import argparse, hashlib, html, json, random, re, time
from pathlib import Path

LABELS = ["SUPPORTED", "CONTRADICTED", "CONFLICTING", "INSUFFICIENT"]
ERRORS = ["none", "number/entity", "negation", "speaker_attribution", "time_shift", "revision_ignored",
          "status_flip (e.g. proposed vs decided)", "fabrication (not in meeting)", "dependency", "other"]
ATOM = ["atomic (one checkable fact)", "compound (should be split)", "fragment (cannot be checked alone)"]


def parse_lines(txt: str, n_max: int) -> list[int]:
    out = []
    for part in re.split(r"[,\s;]+", (txt or "").strip()):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)-(\d+)", part)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            out += [x for x in range(min(a, b), max(a, b) + 1) if 1 <= x <= n_max]
        elif part.isdigit() and 1 <= int(part) <= n_max:
            out.append(int(part))
    return sorted(set(out))


def main():
    import gradio as gr
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="data/meetverify_v2/pack.json")
    ap.add_argument("--annotator", required=True)
    ap.add_argument("--port", type=int, default=7861)
    a = ap.parse_args()
    pack = json.load(open(a.pack, encoding="utf-8"))
    T, S = pack["transcripts"], pack["summaries"]
    seed = int(hashlib.md5(a.annotator.encode()).hexdigest()[:8], 16)
    random.Random(seed).shuffle(S)                       # each annotator sees a different order
    out = Path(a.pack).parent / f"ann_{a.annotator}.jsonl"
    recs = [json.loads(l) for l in open(out, encoding="utf-8")] if out.exists() else []
    done_c = {r["public_id"] for r in recs if r["type"] == "claim"}
    done_s = {r["sid"] for r in recs if r["type"] == "summary"}

    def queue():
        q = []
        for s in S:
            for c in s["claims"]:
                if c["public_id"] not in done_c:
                    q.append(("claim", s, c))
            if s.get("rate_summary", True) and s["sid"] not in done_s and all(c["public_id"] in done_c for c in s["claims"]):
                q.append(("summary", s, None))
        return q

    def transcript_html(mid, q=""):
        rows = []
        for u in T[mid]:
            line = f"{u['speaker']}: {u['text']}"
            if q and q.lower() not in line.lower():
                continue
            rows.append(f"<div style='margin:2px 0'><b>[{u['n']}]</b> <span style='color:#2b6cb0'>{html.escape(u['speaker'])}</span>: "
                        f"{html.escape(u['text'])}</div>")
        return "<div style='height:72vh;overflow-y:auto;font-size:14px;padding:6px;border:1px solid #888'>" + "".join(rows) + "</div>"

    def summary_md(s, cur=None):
        lines = []
        for i, sent in enumerate(s["sentences"]):
            mark = any(c["text"] and c.get("sent_idx") == i and c["public_id"] == cur for c in s["claims"]) if cur else False
            lines.append(f"{'**▶ ' if mark else ''}{i + 1}. {sent}{'**' if mark else ''}")
        return "\n\n".join(lines) if s["sentences"] else ""

    def view():
        q = queue()
        total = sum(len(s["claims"]) for s in S)
        prog = f"claims done {len(done_c)}/{total}"
        if not q:
            return dict(kind="done", s=None, head="## All done — thank you!", summ="", tr="", prog=prog)
        kind, s, c = q[0]
        if kind == "claim":
            head = f"### Claim {c['public_id']}\n\n> **{c['text']}**"
            return dict(kind="claim", s=s, c=c, head=head, summ=summary_md(s, c["public_id"]), tr=transcript_html(s["meeting_id"]), prog=prog)
        return dict(kind="summary", s=s, head=f"### Rate the whole summary {s['sid']}", summ=summary_md(s), tr=transcript_html(s["meeting_id"]), prog=prog)

    with gr.Blocks(title="Claim annotation") as demo:
        gr.Markdown("**Read guidelines first.** Judge ONLY against the transcript on the right (line numbers in [brackets]).")
        prog = gr.Markdown()
        with gr.Row():
            with gr.Column(scale=5):
                head = gr.Markdown()
                summ = gr.Markdown()
                with gr.Group(visible=True) as g_claim:
                    label = gr.Radio(LABELS, label="Label")
                    err = gr.Dropdown(ERRORS, value="none", label="Error type (if not SUPPORTED)")
                    ev = gr.Textbox(label="Evidence lines, e.g. 12, 14-16, 40  (lines that support OR contradict the claim)")
                    atom = gr.Radio(ATOM, label="Is the claim atomic?")
                    note = gr.Textbox(label="Note (optional)")
                with gr.Group(visible=False) as g_sum:
                    rating = gr.Radio(["1", "2", "3", "4", "5"], label="Overall factual accuracy of the whole summary (1 = many errors, 5 = fully accurate)")
                    review = gr.Radio(["yes", "no"], label="Would you want a human to check this summary before using it?")
                    missed = gr.Textbox(label="Important facts in the meeting that the summary missed / statements not covered by any claim (optional)", lines=3)
                msg = gr.Markdown()
                with gr.Row():
                    save = gr.Button("Save & next", variant="primary")
                    undo = gr.Button("Undo last")
            with gr.Column(scale=6):
                search = gr.Textbox(label="Search transcript (plain text filter; empty = show all)")
                tr = gr.HTML()
        state = gr.State({})

        def paint(v, m=""):
            is_c = v["kind"] == "claim"; is_s = v["kind"] == "summary"
            return (v["prog"], v["head"], v["summ"], v["tr"], gr.update(visible=is_c), gr.update(visible=is_s), m,
                    None, "none", "", None, "", None, None, "", v)

        outs = [prog, head, summ, tr, g_claim, g_sum, msg, label, err, ev, atom, note, rating, review, missed, state]

        def on_load():
            return paint(view())

        def on_save(label_, err_, ev_, atom_, note_, rating_, review_, missed_, st):
            v = view()
            if v["kind"] == "done":
                return paint(v)
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            if v["kind"] == "claim":
                mid = v["s"]["meeting_id"]; n = len(T[mid])
                lines = parse_lines(ev_, n)
                if not label_ or not atom_:
                    return (*paint(v, "⚠ choose a label and an atomicity option")[:7], label_, err_, ev_, atom_, note_, rating_, review_, missed_, v)
                if label_ in ("SUPPORTED", "CONTRADICTED", "CONFLICTING") and not lines:
                    return (*paint(v, "⚠ give the evidence line numbers (at least one)")[:7], label_, err_, ev_, atom_, note_, rating_, review_, missed_, v)
                rec = {"type": "claim", "public_id": v["c"]["public_id"], "sid": v["s"]["sid"], "annotator": a.annotator, "label": label_,
                       "error_type": err_ if label_ != "SUPPORTED" else "none", "evidence_lines": lines,
                       "evidence_uids": [T[mid][i - 1]["uid"] for i in lines], "atomicity": atom_.split(" ")[0], "note": note_ or "", "ts": now}
                done_c.add(rec["public_id"])
            else:
                if not rating_ or not review_:
                    return (*paint(v, "⚠ give a rating and the yes/no")[:7], label_, err_, ev_, atom_, note_, rating_, review_, missed_, v)
                rec = {"type": "summary", "sid": v["s"]["sid"], "annotator": a.annotator, "rating": int(rating_),
                       "needs_review": review_ == "yes", "missed": missed_ or "", "ts": now}
                done_s.add(rec["sid"])
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            recs.append(rec)
            return paint(view())

        def on_undo(st):
            if not recs:
                return paint(view(), "nothing to undo")
            r = recs.pop()
            (done_c.discard(r["public_id"]) if r["type"] == "claim" else done_s.discard(r["sid"]))
            with open(out, "w", encoding="utf-8") as f:
                for x in recs:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
            return paint(view(), "last annotation removed — redo it")

        def on_search(q, st):
            return transcript_html(st["s"]["meeting_id"], q) if st and st.get("s") else ""

        demo.load(on_load, None, outs)
        save.click(on_save, [label, err, ev, atom, note, rating, review, missed, state], outs)
        undo.click(on_undo, [state], outs)
        search.change(on_search, [search, state], [tr])
    demo.launch(server_port=a.port)


if __name__ == "__main__":
    main()
