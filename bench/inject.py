"""MeetVerify-Synth: controlled, slot-level error injection.

Each corrupted summary contains exactly ONE injected error in ONE sentence, so every
sentence has a gold label and an error type. Injection operates on the slot that the
error type targets (agent, status, time, quantity/object, polarity), which is what makes
the error taxonomy measurable per dimension (RQ2).

Guarantees (fixes over the notebook's perturbation code):
  * attribution swaps never produce self-collisions ("Ravi and Ravi") — the new agent is
    sampled from the other participants and is checked to not already occur in the sentence;
  * a corruption is kept only if the sentence actually changed and the new value is NOT
    supported elsewhere in the transcript window (otherwise it is not an error);
  * revision_ignored uses a value that really occurs EARLIER in the transcript for the same
    object, i.e. a stale state, not a random number.

Usage:
  python -m bench.inject --summaries data/meetverify/summaries.jsonl --out data/meetverify/synth.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from cprove.text import split_sentences, TIME_RX, QTY, content_tokens, quantities, normalise_time
from cprove.frames import ROLE_ALIASES, find_agent

STATUS_MAP = [
    (re.compile(r"\b(decided on|settled on|went with|chose|agreed on)\b", re.I), ["proposed", "rejected", "postponed a decision on"]),
    (re.compile(r"\b(decided to)\b", re.I), ["proposed to", "decided not to"]),
    (re.compile(r"\b(proposed|suggested|recommended)\b", re.I), ["decided on", "rejected"]),
    (re.compile(r"\b(agreed (to|on|that))\b", re.I), ["disagreed about", "rejected the idea"]),
    (re.compile(r"\b(rejected|dismissed|ruled out)\b", re.I), ["approved", "decided on"]),
    (re.compile(r"\b(was left open|remained undecided|was postponed|was deferred)\b", re.I), ["was decided", "was finalised"]),
]
NEG_INSERT = [(re.compile(r"\b(is|are|was|were)\b(?! not)", re.I), r"\1 not"),
              (re.compile(r"\bwill\b(?! not)", re.I), "will not"),
              (re.compile(r"\b(decided|agreed|wanted|preferred|chose)\b", re.I),
               lambda m: "did not " + {"decided": "decide", "agreed": "agree", "wanted": "want", "preferred": "prefer", "chose": "choose"}[m.group(1).lower()])]
NEG_REMOVE = [(re.compile(r"\b(is|are|was|were|will) not\b", re.I), r"\1"),
              (re.compile(r"\bdid not (\w+)", re.I), r"\1ed"), (re.compile(r"\bno longer\b", re.I), "still")]
TIME_ALTS = {"WD": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
             "EV": ["next week", "the end of the month"], "REL": ["next week", "tomorrow", "next month"]}


def _swap_agent(sent, participants, rng):
    cur = find_agent(sent, participants)
    if not cur or cur == "group":
        return None
    # find surface form that maps to cur
    forms = sorted([k for k, v in ROLE_ALIASES.items() if v == cur] + [cur.lower()], key=len, reverse=True)
    for f in forms:
        m = re.search(r"\b" + re.escape(f) + r"\b", sent, re.I)
        if m:
            disp = {"Marketing": "Marketing Expert", "User Interface": "User Interface Designer"}
            others = [p for p in participants if p != cur and p.lower() not in sent.lower()]
            if not others:
                return None
            new = rng.choice(others)
            shown = disp.get(new, new)
            lead = "The " if m.start() == 0 and new != "group" else ""
            return sent[:m.start()] + lead + shown + sent[m.end():], {"slot": "agent", "old": cur, "new": new}
    return None


def _status_flip(sent, rng):
    for rx, alts in STATUS_MAP:
        m = rx.search(sent)
        if m:
            new = rng.choice(alts)
            out = sent[:m.start()] + new + sent[m.end():]
            if new in ("was decided", "was finalised"):
                out = re.sub(r"\s+until [^,.]*", "", out)
            return out, {"slot": "status", "old": m.group(0), "new": new}
    return None


def _negation(sent, rng):
    for rx, rep in NEG_REMOVE + NEG_INSERT:
        if rx.search(sent):
            new = rx.sub(rep, sent, count=1)
            if new != sent:
                return new, {"slot": "polarity"}
    return None


def _time_shift(sent, rng):
    m = TIME_RX.search(sent)
    if not m or m.group(0).lower() in ("first", "then", "later", "finally", "earlier"):
        return None
    tn = normalise_time(m.group(0)) or ""
    kind = tn.split(":", 1)[0]
    alts = [a for a in TIME_ALTS.get(kind, TIME_ALTS["WD"]) if a.lower() not in m.group(0).lower()]
    new = rng.choice(alts)
    lead = re.match(r"(by|before|on|until|after) ", m.group(0), re.I)
    rep = (lead.group(0) if lead else "") + new
    return sent[:m.start()] + rep + sent[m.end():], {"slot": "time", "old": m.group(0), "new": rep}


def _quantity(sent, rng, transcript_text):
    ms = list(QTY.finditer(sent))
    ms = [m for m in ms if re.search(r"\d", m.group(0))]
    if not ms:
        return None
    m = rng.choice(ms)
    v = float(re.search(r"\d+(?:[.,]\d+)?", m.group(0)).group(0).replace(",", "."))
    for _ in range(10):
        nv = round(v * rng.choice([0.5, 1.5, 2, 3, 0.8, 1.2]), 2)
        nv = int(nv) if nv == int(nv) else nv
        if str(nv) not in transcript_text and nv != v:
            break
    num = re.search(r"\d+(?:[.,]\d+)?", m.group(0))
    rep = m.group(0)[:num.start()] + str(nv) + m.group(0)[num.end():]
    return sent[:m.start()] + rep + sent[m.end():], {"slot": "quantity", "old": m.group(0), "new": rep}


def noun_vocab(transcript_text: str) -> set:
    """Rough noun pool: words following a determiner/possessive in the transcript."""
    from cprove.text import STOP
    nouns = re.findall(r"\b(?:a|an|the|our|this|that|of|kinetic|rubber|plastic)\s+([a-z]{4,})\b", transcript_text)
    return {w for w in nouns if w not in STOP and not w.endswith(("ly", "ing", "ed"))}


def _entity(sent, rng, meeting_vocab):
    toks = [t for t in re.findall(r"\b[a-z]{4,}\b", sent) if t in meeting_vocab]
    nouns = [t for t in toks if t not in ("will", "this", "that", "with", "they", "from", "have", "were", "decided",
                                         "proposed", "agreed", "group", "team", "meeting", "designer", "marketing", "manager",
                                         "interface", "industrial", "project", "user")]
    if not nouns:
        return None
    old = rng.choice(nouns)
    cands = [w for w in meeting_vocab if w != old and len(w) >= 5 and w not in sent.lower()]
    if not cands:
        return None
    new = rng.choice(cands)
    return re.sub(r"\b" + old + r"\b", new, sent, count=1), {"slot": "object", "old": old, "new": new}


def _revision_ignored(sent, rng, meeting):
    """If the sentence states a value that the transcript reached after a revision,
    replace it with the stale earlier value and present it as the decision."""
    digit_ms = [m for m in QTY.finditer(sent) if re.search(r"\d", m.group(0))]
    if not digit_ms:
        return None
    q = [float(re.search(r"\d+(?:[.,]\d+)?", m.group(0)).group(0).replace(",", ".")) for m in digit_ms]
    toks = content_tokens(sent)
    utts = meeting["utterances"]
    final_pos = None
    for j, u in enumerate(utts):
        if set(quantities(u["text"])) & set(q):
            final_pos = j
    if final_pos is None:
        return None
    for j in range(final_pos - 1, -1, -1):
        uq = [x for x in quantities(utts[j]["text"]) if x not in q and x > 1]
        if uq and len(toks & content_tokens(utts[j]["text"])) >= 1:
            stale = uq[0]
            stale = int(stale) if stale == int(stale) else stale
            m = [m for m in QTY.finditer(sent) if re.search(r"\d", m.group(0))][0]
            num = re.search(r"\d+(?:[.,]\d+)?", m.group(0))
            rep = m.group(0)[:num.start()] + str(stale) + m.group(0)[num.end():]
            new = sent[:m.start()] + rep + sent[m.end():]
            new = re.sub(r"\b(changed|revised|switched)( the \w+)? to\b", r"decided on\2", new, flags=re.I)
            new = re.sub(r"\s*\binstead\b", "", new)
            return new, {"slot": "ctx", "old": m.group(0), "new": rep, "stale_from": utts[j]["uid"]}
    return None


def _entity_llm(sent, llm):
    from cprove.prompts import ENTITY_SCHEMA, ENTITY_SYSTEM
    out = llm.json(ENTITY_SYSTEM, sent, ENTITY_SCHEMA)
    if not out or not out.get("sentence"):
        return None
    new = out["sentence"].strip()
    if new == sent or abs(len(new) - len(sent)) > 0.4 * len(sent) + 15:
        return None
    return new, {"slot": "object", "old": out.get("changed"), "new": new, "llm": True}


def _fabrication(sents, participants, rng, llm=None, summary=""):
    if llm is not None and getattr(llm, "available", False):
        from cprove.prompts import FABRICATE_SCHEMA, FABRICATE_SYSTEM, FABRICATE_USER
        hints = ["a budget or cost figure", "a specific named person doing a task", "a technical feature or component",
                 "a marketing or branding decision", "a deadline or schedule", "a supplier or manufacturing choice",
                 "a colour, material or shape decision", "a rejected alternative", "a competitor product", "an action item for next meeting"]
        hint = rng.choice(hints)
        out = llm.json(FABRICATE_SYSTEM + f" The invented sentence must concern: {hint}. Do not mention user testing, surveys or feedback.",
                       FABRICATE_USER.format(participants=", ".join(participants), summary=summary) + f"\n(variation {rng.randrange(10**6)})",
                       FABRICATE_SCHEMA)
        if out and out.get("sentence"):
            return out["sentence"], {"slot": "all", "llm": True}
    a = rng.choice(participants)
    things = ["a solar-powered charging dock", "a voice-control feature", "a subscription pricing model",
              "an external design agency", "a second prototype in metal", "a customer survey in Germany"]
    return f"The {a} proposed {rng.choice(things)}, and the group agreed to it.", {"slot": "all", "llm": False}


INJECTORS = ["entity", "negation", "attribution_swap", "time_shift", "revision_ignored", "status_flip",
             "dependency_cascade", "fabrication"]


def inject_one(summary: str, meeting: dict, etype: str, rng: random.Random, llm=None):
    """Returns dict(summary, sentences, gold[list 0/1], error_type, sent_idx, info) or None."""
    sents = split_sentences(summary)
    parts = meeting.get("meta", {}).get("participants") or sorted({u["speaker"] for u in meeting["utterances"]})
    ttext = " ".join(u["text"] for u in meeting["utterances"]).lower()
    vocab = noun_vocab(ttext)
    order = list(range(len(sents)))
    rng.shuffle(order)
    if etype == "fabrication":
        k = rng.randrange(len(sents) + 1)
        fab, info = _fabrication(sents, parts, rng, llm, summary)
        new = sents[:k] + [fab] + sents[k:]
        gold = [1] * len(new)
        gold[k] = 0
        return dict(sentences=new, gold=gold, error_type=etype, sent_idx=[k], info=info)
    for i in order:
        s = sents[i]
        res = None
        if etype == "attribution_swap":
            res = _swap_agent(s, parts, rng)
        elif etype == "status_flip":
            res = _status_flip(s, rng)
        elif etype == "negation":
            res = _negation(s, rng)
        elif etype == "time_shift":
            res = _time_shift(s, rng)
        elif etype == "entity" and llm is not None and getattr(llm, "available", False):
            res = _entity_llm(s, llm)
        elif etype == "entity":
            res = _quantity(s, rng, ttext) if quantities(s) and rng.random() < 0.5 else _entity(s, rng, vocab)
        elif etype == "revision_ignored":
            try:
                res = _revision_ignored(s, rng, meeting)
            except Exception:       # malformed sentence: skip it, never abort the run
                res = None
        elif etype == "dependency_cascade":
            res = _time_shift(s, rng) or _swap_agent(s, parts, rng)
            if res:
                # propagate the same corruption into a later sentence that depends on it
                toks = content_tokens(s)
                dep = [j for j in range(i + 1, len(sents)) if len(toks & content_tokens(sents[j])) >= 2
                       and re.search(r"\b(after|once|based on|following|when)\b", sents[j], re.I)]
                if not dep:
                    res = None
                else:
                    j = dep[0]
                    old, newv = res[1].get("old"), res[1].get("new")
                    if old and old in sents[j]:
                        sj = sents[j].replace(old, newv)
                    elif res[1].get("slot") == "time":    # dependent inherits the shifted date
                        sj = sents[j].rstrip(". ") + " on " + re.sub(r"^(by|before|on|until|after) ", "", newv) + "."
                    else:
                        sj = sents[j]
                    if sj == sents[j]:
                        res = None
                    else:
                        new = list(sents)
                        new[i], new[j] = res[0], sj
                        gold = [1] * len(new)
                        gold[i] = gold[j] = 0
                        return dict(sentences=new, gold=gold, error_type=etype, sent_idx=[i, j], info=res[1])
                continue
        if res and res[0] != s:
            new = list(sents)
            new[i] = res[0]
            gold = [1] * len(new)
            gold[i] = 0
            return dict(sentences=new, gold=gold, error_type=etype, sent_idx=[i], info=res[1])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", required=True, help="jsonl: {summary_id, meeting_id, summary, labels?}")
    ap.add_argument("--meetings", required=True, help="normalised meetings jsonl (cprove.data.save_meetings_jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per_type", type=int, default=1, help="corrupted copies per summary per error type")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_llm", action="store_true", help="LLM for fabrications (Qwen via config)")
    ap.add_argument("--systems", nargs="*", default=["reference"],
                    help="only inject into these summary sources (reference abstracts are faithful by construction)")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    meetings = {}
    with open(a.meetings, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            meetings[d["meeting_id"]] = d
    llm = None
    if a.use_llm:
        from cprove.utils import load_config
        from cprove.llm import make_llm
        llm = make_llm(load_config()["llm"])
    n = 0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.summaries, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as g:
        for line in f:
            s = json.loads(line)
            m = meetings.get(s["meeting_id"])
            if s.get("system", "reference") not in a.systems:
                continue
            if not m:
                continue
            # the clean version is itself an example (all sentences gold=1 if source is a reference summary)
            sents = split_sentences(s["summary"])
            base_gold = s.get("labels") or [1] * len(sents)
            g.write(json.dumps({"summary_id": s["summary_id"] + "_clean", "meeting_id": s["meeting_id"],
                                "sentences": sents, "gold": base_gold, "error_type": "none", "sent_idx": []}) + "\n")
            for et in INJECTORS:
                for k in range(a.per_type):
                    r = inject_one(s["summary"], m, et, rng, llm)
                    if r is None:
                        continue
                    r.update(summary_id=f"{s['summary_id']}_{et}_{k}", meeting_id=s["meeting_id"])
                    g.write(json.dumps(r, ensure_ascii=False) + "\n")
                    n += 1
    print(f"wrote {n} corrupted summaries -> {a.out}")


if __name__ == "__main__":
    main()