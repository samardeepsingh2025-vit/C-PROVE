"""Command line: cprove verify --transcript t.txt --summary s.txt [--config configs/default.yaml] [--json out.json]"""
from __future__ import annotations
import argparse, json
from .utils import load_config
from .io import parse_transcript
from .pipeline import CProve


def result_to_dict(r):
    return {"mti": r.mti, "mti_interval": r.mti_interval, "components": r.components,
            "claims": [{"cid": c.cid, "text": c.text, "frame": c.frame.__dict__, "type": c.claim_type,
                        "y": r.post[c.cid]["y"], "dims": {k: v for k, v in r.post[c.cid].items() if k != "y"},
                        "label": r.labels[c.cid], "proof": r.proofs[c.cid]} for c in r.claims],
            "edges": [e.__dict__ for e in r.edges], "llm_calls": getattr(r.trace, "calls", 0)}


def main():
    ap = argparse.ArgumentParser(prog="cprove")
    ap.add_argument("cmd", choices=["verify"])
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scheduler", default=None, help="none|voi|entropy|random|all")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    m = parse_transcript(open(a.transcript, encoding="utf-8").read())
    r = CProve(cfg).verify(m, open(a.summary, encoding="utf-8").read(), "S0", scheduler=a.scheduler)
    d = result_to_dict(r)
    print(f"MTI = {d['mti']:.3f}  interval = [{d['mti_interval'][0]:.2f}, {d['mti_interval'][1]:.2f}]  LLM calls = {d['llm_calls']}")
    for c in d["claims"]:
        print(f"[{c['label']:<11}] y={c['y']:.2f}  {c['text']}")
        for p in c["proof"]:
            print("      -", p["text"])
    if a.json:
        json.dump(d, open(a.json, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
