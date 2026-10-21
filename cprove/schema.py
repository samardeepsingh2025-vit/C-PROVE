"""Core data structures for C-PROVE.

Everything that flows between stages is one of these dataclasses, so each
stage can be cached to disk (JSON / Parquet) and re-run independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Literal
import json

STATUSES = ("proposed", "agreed", "decided", "rejected", "assigned", "deferred", "revised", "stated")
EDGE_TYPES = ("before", "causes", "same_agent", "task_depends", "coref", "revises", "conflicts")
DIMENSIONS = ("fact", "attr", "temp", "ctx", "sem", "rsn")
LABELS = ("SUPPORTED", "UNSUPPORTED", "ABSTAIN")
ERROR_TYPES = (
    "none", "entity", "negation", "attribution_swap", "time_shift",
    "revision_ignored", "status_flip", "dependency_cascade", "fabrication",
)

Status = Literal["proposed", "agreed", "decided", "rejected", "assigned", "deferred", "revised", "stated"]


@dataclass
class Utterance:
    uid: str                  # e.g. "ES2002a_u0153"
    speaker: str              # normalised speaker id/name ("A", "Project Manager", ...)
    text: str
    start: float = 0.0        # seconds
    end: float = 0.0
    idx: int = 0              # position in meeting
    dialogue_act: Optional[str] = None   # inform / suggest / assess / elicit / backchannel ...


@dataclass
class Meeting:
    meeting_id: str
    utterances: list[Utterance]
    meta: dict = field(default_factory=dict)   # date, participants, roles, corpus

    @property
    def speakers(self) -> list[str]:
        seen = []
        for u in self.utterances:
            if u.speaker not in seen:
                seen.append(u.speaker)
        return seen

    def window(self, idx: int, k: int = 2) -> list[Utterance]:
        lo, hi = max(0, idx - k), min(len(self.utterances), idx + k + 1)
        return self.utterances[lo:hi]


@dataclass
class Frame:
    """Slot-structured representation of an atomic claim."""
    agent: Optional[str] = None       # who did/said/owns it
    act: Optional[str] = None         # verb lemma: decide, propose, design, send ...
    obj: Optional[str] = None         # object / topic
    time: Optional[str] = None        # surface time expression
    time_norm: Optional[str] = None   # ISO-ish normalised ("2026-03-14", "REL:+1W", "ORD:3")
    status: Status = "stated"
    polarity: int = 1                 # 1 affirmed, 0 negated
    quantity: Optional[str] = None    # numbers, prices, counts
    recipient: Optional[str] = None   # for assignments: "assigned to X"

    def slots_present(self) -> dict[str, bool]:
        return {
            "agent": bool(self.agent), "time": bool(self.time),
            "quantity": bool(self.quantity), "status": self.status != "stated",
            "recipient": bool(self.recipient),
        }


@dataclass
class Claim:
    cid: str
    text: str
    frame: Frame = field(default_factory=Frame)
    summary_id: str = ""
    sent_idx: int = 0                  # sentence index inside the summary
    claim_type: str = "fact"          # fact | decision | action_item | opinion | event
    # gold (bench only)
    gold_label: Optional[str] = None   # SUPPORTED / UNSUPPORTED
    gold_error: str = "none"
    source_cid: Optional[str] = None   # for injected claims: original claim

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Claim":
        d = dict(d)
        d["frame"] = Frame(**d.get("frame", {}))
        return Claim(**d)


@dataclass
class Edge:
    src: str
    dst: str
    etype: str
    weight: float = 1.0      # edge confidence in [0,1]
    source: str = "rule"     # rule | classifier | llm | gold

    def key(self):
        return (self.src, self.dst, self.etype)


@dataclass
class Evidence:
    cid: str
    uids: list[str]
    scores: list[float]
    text: str                       # concatenated window text fed to NLI
    slot_hits: dict = field(default_factory=dict)  # slot -> uid that grounds it


@dataclass
class Observation:
    """Calibrated per-dimension observations for one claim (each in [0,1] or None
    if the dimension is not applicable — inapplicable dimensions create no atom)."""
    cid: str
    ent: float = 0.0
    con: float = 0.0
    fact: Optional[float] = None
    attr: Optional[float] = None
    temp: Optional[float] = None
    ctx: Optional[float] = None
    sem: Optional[float] = None
    llm: Optional[float] = None      # filled by adaptive verifier (expensive judge)
    raw: dict = field(default_factory=dict)

    def applicable(self) -> list[str]:
        return [d for d in ("fact", "attr", "temp", "ctx", "sem") if getattr(self, d) is not None]


@dataclass
class Verdict:
    cid: str
    y: float                         # posterior faithfulness in [0,1]
    dims: dict                       # dimension posteriors
    label: str                       # SUPPORTED / UNSUPPORTED / ABSTAIN
    proof: list = field(default_factory=list)
    llm_called: bool = False


def dump_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            if hasattr(r, "__dataclass_fields__"):
                r = asdict(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
