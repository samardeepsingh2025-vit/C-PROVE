"""Lightweight text utilities: sentence splitting, lexicons, time / quantity
normalisation, and status cue detection. Pure python, no models."""
from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Optional

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def norm(s: Optional[str]) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


STOP = set("""a an the of to in on for with and or but is are was were be been being it its this that these those
he she they them their his her we our you your i me my at by from as about into than then there here which who whom
what when where why how will would shall should can could may might must do does did done have has had not no""".split())


def content_tokens(s: str) -> set[str]:
    return {t for t in norm(s).split() if t not in STOP and len(t) > 1}


def jaccard(a: str, b: str) -> float:
    A, B = content_tokens(a), content_tokens(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


# --------------------------------------------------------------------- status cues
STATUS_CUES = {
    "decided": r"\b(decid\w*|final(ly|ised|ized)?|settled on|went with|go with|chose|chosen|concluded|agreed to go|we'?ll go|let'?s go with)\b",
    "agreed": r"\b(agree\w*|consensus|everyone liked|all (liked|accepted)|accepted|approv\w*|sounds good|okay with|fine with)\b",
    "proposed": r"\b(propos\w*|suggest\w*|recommend\w*|rais\w*|put forward|floated|pitched|what about|how about|maybe we|we could|idea)\b",
    "rejected": r"\b(reject\w*|dismiss\w*|ruled out|turned down|dropp\w*|abandon\w*|against|discard\w*|no to)\b",
    "assigned": r"\b(will (look|send|prepare|work|draw|make|check|find|do|design|present|investigate|research)|is to|are to|tasked|assigned|responsible for|action item|has to|needs? to|should (look|send|prepare|work|make|find))\b",
    "deferred": r"\b(defer\w*|postpon\w*|later|next meeting|left open|leave it open|keep it open|undecided|not decide\w*|not yet decided|table[ds]?|revisit)\b",
    "revised": r"\b(chang\w* (their|his|her|the) mind|revis\w*|instead|switched|reconsider\w*|changed to|no longer|actually)\b",
}
STATUS_PRIORITY = ["revised", "rejected", "decided", "assigned", "agreed", "deferred", "proposed"]


def status_cues(text: str) -> dict[str, float]:
    t = text.lower()
    out = {k: float(bool(re.search(p, t))) for k, p in STATUS_CUES.items()}
    # "we will not decide X today" -> deferred, not decided
    if out["decided"] and re.search(r"\b(not|n't|won't|never)\s+(\w+\s+){0,2}(decid\w*|final\w*|settle)", t):
        out["decided"], out["deferred"] = 0.0, 1.0
    return out


def guess_status(text: str) -> str:
    cues = status_cues(text)
    for s in STATUS_PRIORITY:
        if cues[s]:
            return s
    return "stated"


NEG = re.compile(r"\b(not|no|never|n't|nobody|none|neither|nor|without|cannot)\b", re.I)


def polarity(text: str) -> int:
    return 0 if NEG.search(text) else 1


# compatibility of statuses: claim status (row) supported by evidence status (col)?
# 1 = compatible, 0 = incompatible, 0.5 = weakly compatible
STATUS_COMPAT = {
    "decided": {"decided": 1, "agreed": 0.8, "assigned": 0.6, "proposed": 0.2, "rejected": 0, "deferred": 0, "revised": 0.6, "stated": 0.5},
    "agreed": {"decided": 1, "agreed": 1, "assigned": 0.6, "proposed": 0.3, "rejected": 0, "deferred": 0.2, "revised": 0.6, "stated": 0.5},
    "proposed": {"decided": 0.7, "agreed": 0.7, "assigned": 0.5, "proposed": 1, "rejected": 0.6, "deferred": 0.7, "revised": 0.6, "stated": 0.6},
    "rejected": {"decided": 0.1, "agreed": 0, "assigned": 0, "proposed": 0.2, "rejected": 1, "deferred": 0.3, "revised": 0.6, "stated": 0.4},
    "assigned": {"decided": 0.6, "agreed": 0.6, "assigned": 1, "proposed": 0.3, "rejected": 0, "deferred": 0.2, "revised": 0.5, "stated": 0.5},
    "deferred": {"decided": 0, "agreed": 0.1, "assigned": 0.3, "proposed": 0.6, "rejected": 0.3, "deferred": 1, "revised": 0.5, "stated": 0.5},
    "revised": {"decided": 0.6, "agreed": 0.6, "assigned": 0.4, "proposed": 0.4, "rejected": 0.5, "deferred": 0.4, "revised": 1, "stated": 0.5},
    "stated": {k: 0.8 for k in ["decided", "agreed", "assigned", "proposed", "rejected", "deferred", "revised", "stated"]},
}

# --------------------------------------------------------------------- quantities
NUM_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
NUM_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "thousand": 1000, "half": 0.5})
QTY = re.compile(r"(?:[$€£]\s?)?\b\d+(?:[.,]\d+)?\s?(?:%|percent|euros?|dollars?|pounds?|cents?|k|million|minutes?|hours?|days?|weeks?|units?|buttons?|grams?)?\b", re.I)


def quantities(text: str) -> list[float]:
    vals = []
    for m in QTY.finditer(text):
        n = re.search(r"\d+(?:[.,]\d+)?", m.group(0))
        if n:
            try:
                vals.append(float(n.group(0).replace(",", ".")))
            except ValueError:
                pass
    for w in norm(text).split():
        if w in NUM_WORDS and w not in ("one",):   # "one" is too ambiguous
            vals.append(float(NUM_WORDS[w]))
    return vals


def extract_quantity(text: str) -> Optional[str]:
    m = QTY.search(text)
    return m.group(0).strip() if m else None


# --------------------------------------------------------------------- time
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
TIME_RX = re.compile(
    r"\b(today|tomorrow|yesterday|tonight|this (morning|afternoon|evening|week|month)|next (week|month|meeting|time|session|monday|tuesday|wednesday|thursday|friday)|"
    r"(by|before|after|on|until|till) (the )?(next meeting|end of (the )?(day|week|month)|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"in (\d+|a|an|one|two|three|four|five|ten|fifteen|twenty|thirty) (minutes?|hours?|days?|weeks?|months?)|"
    r"(\d{1,2})(st|nd|rd|th)? (of )?(january|february|march|april|may|june|july|august|september|october|november|december)|"
    r"(january|february|march|april|may|june|july|august|september|october|november|december) \d{1,2}(st|nd|rd|th)?|"
    r"\d{4}-\d{2}-\d{2}|\d{1,2}(:\d{2})? ?(am|pm)|end of (the )?(day|week|month|meeting)|first meeting|second meeting|third meeting|final meeting|"
    r"(at the )?(start|beginning|end) of the meeting|later|earlier|first|then|finally|afterwards|subsequently)\b",
    re.I,
)
ORDER_WORDS = {"first": 0, "earlier": 0, "beginning of the meeting": 0, "start of the meeting": 0, "then": 1,
               "later": 2, "afterwards": 2, "subsequently": 2, "finally": 3, "end of the meeting": 3}


def extract_time(text: str) -> Optional[str]:
    m = TIME_RX.search(text)
    return m.group(0) if m else None


def normalise_time(expr: Optional[str], ref: Optional[date] = None) -> Optional[str]:
    """Map a time expression to a comparable canonical form.
    ABS:YYYY-MM-DD   absolute date (needs meeting date `ref` for relative ones)
    REL:+<n><unit>   relative offset when no ref date
    WD:<weekday>     weekday deadline
    EV:next_meeting  event-anchored deadline
    ORD:<k>          discourse order (first/then/later/finally)
    CLK:HH:MM        clock time
    """
    if not expr:
        return None
    e = expr.lower().strip()
    e = re.sub(r"^(by|before|until|till|on|at) (the )?", "", e)
    if m := re.match(r"\d{4}-\d{2}-\d{2}", e):
        return "ABS:" + m.group(0)
    if "next meeting" in e or "next time" in e or "next session" in e:
        return "EV:next_meeting"
    for i, mname in enumerate(MONTHS):
        if mname in e:
            d = re.search(r"\d{1,2}", e)
            if d:
                yr = ref.year if ref else 2000
                try:
                    return "ABS:" + date(yr, i + 1, int(d.group(0))).isoformat()
                except ValueError:
                    pass
    rel = {"today": 0, "tonight": 0, "tomorrow": 1, "yesterday": -1}
    for k, v in rel.items():
        if k in e:
            return ("ABS:" + (ref + timedelta(days=v)).isoformat()) if ref else f"REL:+{v}d"
    if m := re.search(r"in (\d+|a|an|one|two|three|four|five|ten|fifteen|twenty|thirty) (minute|hour|day|week|month)", e):
        n = m.group(1)
        n = 1 if n in ("a", "an") else int(n) if n.isdigit() else NUM_WORDS.get(n, 1)
        return f"REL:+{n}{m.group(2)[0]}"
    if "next week" in e:
        return "REL:+1w"
    if "next month" in e:
        return "REL:+1m"
    if "end of the week" in e or "end of week" in e:
        return "WD:friday"
    if "end of the day" in e or "end of day" in e:
        return "REL:+0d"
    for wd in WEEKDAYS:
        if wd in e:
            return "WD:" + wd
    if m := re.search(r"(\d{1,2})(?::(\d{2}))? ?(am|pm)", e):
        h = int(m.group(1)) % 12 + (12 if m.group(3) == "pm" else 0)
        return f"CLK:{h:02d}:{m.group(2) or '00'}"
    for k, v in ORDER_WORDS.items():
        if k in e:
            return f"ORD:{v}"
    for k, v in {"first meeting": 1, "second meeting": 2, "third meeting": 3, "final meeting": 4}.items():
        if k in e:
            return f"MTG:{v}"
    return "RAW:" + e


def time_compatible(a: Optional[str], b: Optional[str]) -> Optional[float]:
    """1 = same, 0 = conflicting, None = not comparable."""
    if not a or not b:
        return None
    ka, kb = a.split(":", 1)[0], b.split(":", 1)[0]
    if ka != kb:
        # weekday vs absolute etc. -> not directly comparable
        return None
    if ka == "ORD":
        return None   # discourse order is checked by the temporal rule, not slot match
    return 1.0 if a == b else 0.0
