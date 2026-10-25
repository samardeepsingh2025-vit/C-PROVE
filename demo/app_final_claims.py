"""C-PROVE demo with the PAPER's method: learned, Platt-calibrated fusion (no graph, no MTI).
  python -m demo.app_final            (opens http://127.0.0.1:7861)
Tab 1  Verify your own: type/paste a transcript (or upload audio -> Whisper transcript) + a summary. Live pipeline = frozen config + MiniCheck-free fusion (~30 s per summary).
Tab 2  Real AMI summaries: out-of-fold fusion probabilities vs the author labels (precomputed, instant, no GPU).
Needs: cache/frozen/features.jsonl, outputs/nested_cv.preds.npy; Ollama running for Tab 2."""
import html, json, re, sys
from pathlib import Path
import numpy as np

SEL = "fusion (selected in inner CV)"
BLOCKS = {"ent+con+attr+sem": ["ent", "con", "attr", "sem"], "ent+con+attr+sem+mc": ["ent", "con", "attr", "sem", "mc"]}
_S = {}


def colour(p, thr):
    return "#f6b8b8" if p < thr else ("#fff2b3" if p < thr + 0.15 else "#c8f0c8")


def table(rows):
    """rows: list of (p, thr, text, extra_html). Least-trusted first."""
    rows = sorted(rows, key=lambda r: r[0]); out = []
    for p, thr, text, extra in rows:
        note = p >= thr and ("number not found" in extra or "transcript contradicts" in extra)   # score says ok, but an automatic check raised a note
        label = "FLAG: likely unsupported" if p < thr else ("CHECK: an automatic check raised a note" if note else "ok")
        bg = "#fff2b3" if note else colour(p, thr)
        out.append(f"<div style='background:{bg};padding:6px;margin:4px;border-radius:6px;color:#111'>"
                   f"<b>P(supported) = {p:.2f}</b> &nbsp; {label}<br>{html.escape(text)}{extra}</div>")
    return "".join(out)


# ------------------------------------------------------------------ Tab 1: precomputed
def load_precomputed(features="cache/frozen/features.jsonl", preds="outputs/nested_cv.preds.npy", cv="outputs/nested_cv.json"):
    P = np.load(preds, allow_pickle=True).item(); cids = P["cids"]; pr = dict(zip(cids, P[SEL]))
    rows = [json.loads(l) for l in open(features, encoding="utf-8") if l.strip()]
    try:
        thr = float(np.median([c["threshold"] for c in json.load(open(cv))["choices"]]))
    except Exception:
        thr = 0.5
    by = {}
    for r in rows:
        if r["cid"] in pr:
            by.setdefault(r["summary_id"], []).append(r)
    return by, pr, thr


def show_summary(sid):
    by, pr, thr = _S["pre"]
    rows = by[sid]
    items = []
    for r in rows:
        lab = r.get("author_label"); p = float(pr[r["cid"]])
        wrong = (lab == "UNSUPPORTED" and p >= thr) or (lab == "SUPPORTED" and p < thr)
        extra = f"<br><small>author label: <b>{lab}</b>{' &nbsp; (the model disagrees)' if wrong else ''}</small>"
        items.append((p, thr, r["text"], extra))
    n_flag = sum(i[0] < thr for i in items)
    head = (f"<h3>{len(items)} claims, {n_flag} flagged as likely unsupported (threshold {thr:.2f})</h3>"
            "<p>Scores are out-of-fold: each claim was scored by a model that never saw its meeting. "
            "P(supported) is the model's estimate of transcript support, not a guarantee of truth.</p>")
    return head + table(items)


# ------------------------------------------------------------------ Tab 2: live
def fit_models(features="cache/frozen/features.jsonl"):
    from train.nested_cv import load_rows, fit_select, CANDIDATES
    rows = load_rows(features)
    out = {}
    for name in BLOCKS:
        pred, ch = fit_select(rows, {name: BLOCKS[name]}); out[name] = (pred, ch["threshold"])
    return out


def live_rows(cp, transcript, summary):
    from cprove.io import parse_transcript
    from cprove.dimensions import numeric_check
    m = parse_transcript(transcript)
    claims = cp.decompose(m, summary, "demo")
    evid, nli, obs, grounding = cp.observe(m, claims)
    rows = []
    for c in claims:
        raw = obs[c.cid].raw; nc = numeric_check(m, c)
        fr = {k: v for k, v in getattr(c.frame, "__dict__", {}).items() if v not in (None, "", [], {})}
        row = {"cid": c.cid, "text": c.text, "frame": fr, "ctype": getattr(c, "claim_type", ""), "sent": getattr(c, "sent_idx", 0),
               "ent": float(raw.get("ent") or 0), "con": float(raw.get("con") or 0),
               "num_unseen": float(bool(nc["unseen"])), "num_values": nc["unseen"], "mc": float("nan"),
               "evidence": "\n".join(evid[c.cid].text.split("\n")[:4])}
        row["con_high"] = float(row["con"] >= 0.9)
        for d in ("attr", "temp", "ctx", "sem"):
            v = raw.get(d); row[d] = 0.0 if v is None else float(v) - 0.5; row[d + "_flag"] = 0.0 if v is None else 1.0
        rows.append(row)
    return m, rows


def minicheck(m, rows):
    from minicheck.minicheck import MiniCheck
    from baselines.run_baselines import transcript_text
    mc = MiniCheck(model_name="flan-t5-large", cache_dir="models/hf"); doc = transcript_text(m)
    for r in rows:
        _, p, _, _ = mc.score(docs=[doc], claims=[r["text"]]); r["mc"] = float(p[0])


def clean_transcript(text):
    """Accept pasted Zoom/Teams-style lines: strips leading timestamps like [00:01:02] or 00:01."""
    return "\n".join(re.sub(r"^\s*[\[(]?\d{1,2}:\d{2}(?::\d{2})?[\])]?\s*[-:]?\s*", "", ln) for ln in text.splitlines())


def transcribe(audio, size, name, current):
    """Audio file -> 'Speaker: text' lines (Whisper, no speaker identification). Model is freed afterwards to save GPU memory."""
    if not audio:
        return current
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return "faster-whisper is not installed. In the Anaconda prompt run:  pip install faster-whisper   and restart the app.\n" + (current or "")
    try:
        import torch
        gpu = torch.cuda.is_available()
    except Exception:
        gpu = False
    model = WhisperModel(size, device="cuda" if gpu else "cpu", compute_type="int8_float16" if gpu else "int8")
    segs, _ = model.transcribe(audio, vad_filter=True)
    who = (name or "").strip() or "Speaker"
    text = "\n".join(f"{who}: {x.text.strip()}" for x in segs if x.text.strip())
    del model
    try:
        import gc; gc.collect(); torch.cuda.empty_cache()
    except Exception:
        pass
    if not text:
        return current
    return ((current.rstrip() + "\n") if (current or "").strip() else "") + text


def details_html(r):
    """Extracted frame slots + every signal the fusion used, for one claim."""
    def sig(name, key, note):
        v = f"{r[key] + 0.5:.2f}" if r.get(key + "_flag") else "n/a (could not be computed for this claim)"
        return f"<tr><td>{name}</td><td>{v}</td><td><small>{note}</small></td></tr>"
    slots = "".join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k, v in r["frame"].items()) or "<tr><td colspan=2>none</td></tr>"
    return ("<details><summary>extracted details</summary>"
            f"<p><small>claim type: <b>{html.escape(str(r['ctype']))}</b> &nbsp; source summary sentence: <b>{int(r['sent']) + 1}</b></small></p>"
            f"<b>Extracted frame</b><table style='font-size:90%'>{slots}</table>"
            "<b>Signals used by the model</b><table style='font-size:90%'>"
            f"<tr><td>entailment</td><td>{r['ent']:.2f}</td><td><small>NLI: transcript lines entail the claim</small></td></tr>"
            f"<tr><td>contradiction</td><td>{r['con']:.2f}</td><td><small>NLI: transcript lines contradict the claim</small></td></tr>"
            + sig("attribution", "attr", "is the named speaker/role linked to this act in the transcript")
            + sig("status", "sem", "does the evidence match the claimed status (proposed / decided / deferred ...)")
            + sig("temporal", "temp", "time expressions and ordering agree with the transcript")
            + sig("context", "ctx", "the claim still holds after later revisions in the meeting")
            + f"<tr><td>unseen number</td><td>{'yes' if r['num_unseen'] else 'no'}</td><td><small>a number in the claim does not occur in the transcript</small></td></tr>"
            "</table><small>The deployed model uses entailment, contradiction, attribution and status; temporal and context are shown for inspection "
            "(they were only used in the full-feature variant).</small></details>")


def verify(transcript, summary, use_mc):
    transcript = clean_transcript(transcript)
    if not transcript.strip() or not summary.strip():
        return "Paste a transcript ('Speaker: text' per line) and a summary."
    if "cp" not in _S:
        from cprove.utils import load_config, seed_everything
        from cprove.pipeline import CProve, Ablation
        from cprove.dimensions import Calibrators
        cfg = load_config("configs/frozen.yaml"); seed_everything(cfg["seed"])
        cp = CProve(cfg, Ablation(hand_weights=True)); cp.dims.cal = Calibrators("none")
        _S["cp"] = cp; _S["fit"] = fit_models()
    m, rows = live_rows(_S["cp"], transcript, summary)
    name, note = "ent+con+attr+sem", ""
    if use_mc:
        try:
            minicheck(m, rows); name = "ent+con+attr+sem+mc"
        except Exception as e:                       # 8 GB GPU: fall back instead of failing in front of an audience
            note = f"<p><i>MiniCheck unavailable ({type(e).__name__}); showing the MiniCheck-free model.</i></p>"
    pred, thr = _S["fit"][name]
    ps = pred(rows); items = []
    for r, p in zip(rows, ps):
        flags = []
        if r["num_unseen"]: flags.append("number not found in transcript: " + ", ".join(f"{v:g}" for v in r["num_values"]))
        if r["con_high"]: flags.append(f"transcript contradicts the claim (p_con={r['con']:.2f})")
        extra = ("<br><small>" + " | ".join(flags) + "</small>" if flags else "") + \
                details_html(r) + f"<details><summary>evidence</summary><pre style='white-space:pre-wrap'>{html.escape(r['evidence'])}</pre></details>"
        items.append((float(p), thr, r["text"], extra))
    n_sent = len({r["sent"] for r in rows})
    head = (f"<h3>{len(items)} atomic claims extracted from {n_sent} summary sentence(s), {sum(i[0] < thr for i in items)} flagged (model: {name}, threshold {thr:.2f})</h3>{note}"
            "<p>P(supported) is the calibrated probability that the transcript supports the claim. "
            "It ranks claims for human review; it does not prove a claim true or false. "
            "Decision-status errors (decided vs only discussed) are not reliably detected.</p>")
    return head + table(items)


def main():
    import gradio as gr
    _S["pre"] = load_precomputed()
    ids = sorted(_S["pre"][0])
    with gr.Blocks(title="C-PROVE") as ui:
        gr.Markdown("# C-PROVE: claim-level verification of meeting summaries\n"
                    "Each summary is split into atomic claims; each claim is checked against retrieved transcript evidence "
                    "(NLI, speaker attribution, decision status, optionally MiniCheck) and given a calibrated P(supported).")
        with gr.Tab("Verify your own"):
            gr.Markdown("**Option A:** type or paste a transcript below ('Name: text' per line; timestamps are removed automatically).  \n"
                        "**Option B:** upload or record audio, type the speaker's name, and click *Transcribe audio and add to transcript*. "
                        "Each clip is added under that name, so record one speaker at a time (change the name between clips). "
                        "Whisper cannot tell speakers apart inside one recording, so for a single multi-speaker recording leave the name as 'Speaker' and edit the names by hand.")
            with gr.Row():
                audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Meeting audio (optional)")
                with gr.Column():
                    size = gr.Dropdown(["tiny", "base", "small", "medium"], value="small", label="Speech-to-text model (bigger = slower, more accurate)")
                    who = gr.Textbox(label="Speaker name for this recording", value="Project Manager", placeholder="e.g. Project Manager")
                    tbtn = gr.Button("Transcribe audio and add to transcript")
            t = gr.Textbox(lines=12, label="Transcript (editable)", placeholder="Project Manager: Let us start with the budget.\nMarketing: I think twenty euros is too high.")
            s = gr.Textbox(lines=6, label="Summary to verify", placeholder="Paste the meeting summary here.")
            mc = gr.Checkbox(False, label="Also use MiniCheck (slower, about 2 minutes per summary)")
            res = gr.HTML()
            tbtn.click(transcribe, [audio, size, who, t], t)
            gr.Button("Verify summary", variant="primary").click(verify, [t, s, mc], res)
        with gr.Tab("Real AMI summaries"):
            dd = gr.Dropdown(ids, value=ids[0], label="Summary"); out = gr.HTML()
            dd.change(show_summary, dd, out); ui.load(show_summary, dd, out)
    ui.launch(server_port=7861, inbrowser=True)


if __name__ == "__main__":
    main()
