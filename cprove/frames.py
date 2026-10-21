"""Stage 1 — Atomic claim decomposition into slot frames.

LLM path: Qwen2.5-7B (Q4_K_M, llama.cpp) with JSON-schema-constrained decoding.
Heuristic path: clause splitting + lexicons. Used for CPU smoke tests and as the
"frames off" ablation's decomposer (one claim per sentence, no slots).
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from .schema import Claim, Frame
from .text import (split_sentences, guess_status, polarity, extract_time, extract_quantity,
                   normalise_time)
from .prompts import FRAME_SCHEMA, frame_messages

# AMI role aliases (summaries and transcripts use different names for the same person)
ROLE_ALIASES = {
    "project manager": "Project Manager", "pm": "Project Manager", "manager": "Project Manager",
    "marketing": "Marketing", "marketing expert": "Marketing", "me": "Marketing", "marketer": "Marketing",
    "industrial designer": "Industrial Designer", "id": "Industrial Designer", "designer": "Industrial Designer",
    "user interface": "User Interface", "ui": "User Interface", "user interface designer": "User Interface",
    "interface designer": "User Interface",
    "group": "group", "team": "group", "everyone": "group", "they": "group", "the group": "group", "the team": "group",
}

_ROLE_RX = re.compile(r"\b(" + "|".join(sorted((re.escape(k) for k in ROLE_ALIASES if len(k) > 3), key=len, reverse=True)) + r")\b", re.I)
CLAUSE_SPLIT = re.compile(r"\s*(?:;|, but |, while |, whereas |, and then | and later | but later |, after which )\s*", re.I)


def canon_agent(name: Optional[str], participants: list[str] | None = None) -> Optional[str]:
    if not name:
        return None
    n = name.lower().strip().removeprefix("the ").strip()
    if n in ROLE_ALIASES:
        return ROLE_ALIASES[n]
    for p in participants or []:
        if p.lower() == n or p.lower() in n or n in p.lower():
            return p
    return name.strip()


def find_agent(text: str, participants: list[str]) -> Optional[str]:
    t = text.lower()
    best, pos = None, 10 ** 9
    keys = sorted(set(list(ROLE_ALIASES.keys()) + [p.lower() for p in participants]), key=len, reverse=True)
    for k in keys:
        m = re.search(r"\b" + re.escape(k) + r"\b", t)
        if m and m.start() < pos and k not in ("they", "me", "id"):   # avoid pronoun/word false hits
            best, pos = k, m.start()
    return canon_agent(best, participants) if best else None


def find_recipient(text: str, participants: list[str]) -> Optional[str]:
    m = re.search(r"(?:assigned to|tasked|asked) (?:the )?([A-Z][\w ]+?)(?: to | with |\.|,|$)", text)
    if m:
        return canon_agent(m.group(1), participants)
    return None


VERB = re.compile(r"\b(decid\w+|propos\w+|suggest\w+|agree\w*|reject\w+|present\w*|discuss\w*|will \w+|is to \w+|"
                  r"want\w*|prefer\w*|chose|choose|select\w*|design\w*|explain\w*|report\w*|said|stated|mention\w*|"
                  r"recommend\w*|show\w*|argu\w+|conclud\w+|defer\w*|postpon\w+|chang\w+|switch\w*|cost\w*|need\w*)\b", re.I)


def heuristic_decompose(sentence: str, participants: list[str], prev_agent: Optional[str] = None,
                        meeting_date: Optional[date] = None, split: bool = True) -> list[dict]:
    clauses = [c for c in CLAUSE_SPLIT.split(sentence) if len(c.split()) >= 3] if split else [sentence]
    if not clauses:
        clauses = [sentence]
    out = []
    for c in clauses:
        agent = find_agent(c, participants)
        if agent is None and re.match(r"\s*(he|she|they|it was (he|she))\b", c, re.I):
            agent = prev_agent
        prev_agent = agent
        status = guess_status(c)
        masked = _ROLE_RX.sub(lambda mm: "X" * len(mm.group(0)), c)   # keep offsets, hide "Designer" etc.
        vm = VERB.search(masked)
        act = vm.group(0).lower() if vm else None
        obj = c[vm.end():].strip(" .,") if vm else None
        recipient = find_recipient(c, participants)
        if status == "assigned" and not recipient:
            recipient, agent = agent, None
        ctype = {"decided": "decision", "agreed": "decision", "assigned": "action_item"}.get(status, "fact")
        t = extract_time(c)
        out.append(dict(text=c.strip().rstrip(".") + ".", claim_type=ctype, agent=agent, act=act,
                        object=(obj[:80] if obj else None), time=t, status=status, polarity=polarity(c),
                        quantity=extract_quantity(c), recipient=recipient))
    return out


class FrameExtractor:
    def __init__(self, llm=None, max_claims_per_sentence: int = 4, use_llm: bool = True):
        self.llm = llm
        self.max_claims = max_claims_per_sentence
        self.use_llm = use_llm and llm is not None and llm.available

    def decompose_summary(self, summary: str, summary_id: str, participants: list[str],
                          meeting_date: Optional[date] = None, frames_off: bool = False) -> list[Claim]:
        sents = split_sentences(summary)
        claims: list[Claim] = []
        prev_agent = None
        for si, s in enumerate(sents):
            if frames_off:
                items = [dict(text=s, claim_type="fact", agent=None, act=None, object=None, time=None,
                              status="stated", polarity=1, quantity=None, recipient=None)]
            else:
                items = None
                if self.use_llm:
                    sys_, usr = frame_messages(s, participants, sents[:si])
                    res = self.llm.json(sys_, usr, FRAME_SCHEMA)
                    if res and isinstance(res.get("claims"), list) and res["claims"]:
                        items = res["claims"][: self.max_claims]
                if items is None:
                    items = heuristic_decompose(s, participants, prev_agent, meeting_date)[: self.max_claims]
            for k, it in enumerate(items):
                agent = canon_agent(it.get("agent"), participants)
                prev_agent = agent or canon_agent(it.get("recipient"), participants) or prev_agent
                fr = Frame(agent=agent, act=it.get("act"), obj=it.get("object"), time=it.get("time"),
                           time_norm=normalise_time(it.get("time"), meeting_date), status=it.get("status", "stated"),
                           polarity=int(it.get("polarity", 1)), quantity=it.get("quantity"),
                           recipient=canon_agent(it.get("recipient"), participants))
                claims.append(Claim(cid=f"{summary_id}_s{si:02d}c{k}", text=it.get("text", s), frame=fr,
                                    summary_id=summary_id, sent_idx=si, claim_type=it.get("claim_type", "fact")))
        return claims
