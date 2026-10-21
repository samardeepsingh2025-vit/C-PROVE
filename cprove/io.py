"""Plain-text transcript parsing for the CLI and demo."""
from __future__ import annotations
import re
from .schema import Meeting, Utterance


def parse_transcript(text: str, meeting_id: str = "M0") -> Meeting:
    """One utterance per line as 'Speaker: text'. Lines without a speaker continue the previous one."""
    utts = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z][\w .\-]{0,40}?):\s+(.*)$", line)
        if m:
            utts.append(Utterance(f"{meeting_id}_u{len(utts):04d}", m.group(1).strip(), m.group(2).strip(), 0, 0, len(utts)))
        elif utts:
            utts[-1].text += " " + line
    parts = sorted({u.speaker for u in utts})
    return Meeting(meeting_id, utts, {"participants": parts})
