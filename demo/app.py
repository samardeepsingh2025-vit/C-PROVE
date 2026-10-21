"""Gradio demo: paste transcript + summary -> per-claim verdicts, proof traces, MTI.
  python -m demo.app [--config configs/default.yaml]
Optional Whisper: if `faster-whisper` is installed, upload audio to get a transcript (speakers are not diarised)."""
import argparse, json
import gradio as gr
from cprove.utils import load_config
from cprove.io import parse_transcript
from cprove.pipeline import CProve
from cprove.cli import result_to_dict

_cp = {}


def run(transcript, summary, scheduler):
    cp = _cp.setdefault("cp", CProve(load_config(_cp.get("cfg"))))
    r = cp.verify(parse_transcript(transcript), summary, "demo", scheduler=scheduler)
    d = result_to_dict(r)
    rows = [[c["label"], round(c["y"], 2), c["text"], " | ".join(p["text"] for p in c["proof"])] for c in d["claims"]]
    head = f"**MTI {d['mti']:.3f}** (90% interval {d['mti_interval'][0]:.2f}-{d['mti_interval'][1]:.2f}), LLM calls: {d['llm_calls']}"
    return head, rows, json.dumps(d["components"], indent=2)


def transcribe(audio):
    from faster_whisper import WhisperModel
    segs, _ = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16").transcribe(audio)
    return "\n".join(f"Speaker: {s.text.strip()}" for s in segs)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default=None); a = ap.parse_args()
    _cp["cfg"] = a.config
    with gr.Blocks(title="C-PROVE") as ui:
        gr.Markdown("# C-PROVE meeting-summary verifier")
        audio = gr.Audio(type="filepath", label="(optional) meeting audio")
        t = gr.Textbox(lines=12, label="Transcript ('Speaker: text' per line)")
        s = gr.Textbox(lines=6, label="Summary")
        sch = gr.Dropdown(["voi", "none", "entropy", "all"], value="voi", label="LLM-judge scheduler")
        btn = gr.Button("Verify", variant="primary")
        head = gr.Markdown(); tab = gr.Dataframe(headers=["label", "y", "claim", "why"], wrap=True); comp = gr.Code(language="json")
        audio.change(transcribe, audio, t)
        btn.click(run, [t, s, sch], [head, tab, comp])
    ui.launch()


if __name__ == "__main__":
    main()
