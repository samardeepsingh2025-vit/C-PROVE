"""All LLM prompts and JSON schemas used by C-PROVE (versioned: bump PROMPT_VERSION
if you change any text — it is part of the cache key and must be reported in the paper)."""

PROMPT_VERSION = "cprove-v1.0"

STATUS_ENUM = ["proposed", "agreed", "decided", "rejected", "assigned", "deferred", "revised", "stated"]
TYPE_ENUM = ["fact", "decision", "action_item", "opinion", "event"]

# --------------------------------------------------------------------------- Stage 1
FRAME_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "claim_type": {"type": "string", "enum": TYPE_ENUM},
                    "agent": {"type": ["string", "null"]},
                    "act": {"type": ["string", "null"]},
                    "object": {"type": ["string", "null"]},
                    "time": {"type": ["string", "null"]},
                    "status": {"type": "string", "enum": STATUS_ENUM},
                    "polarity": {"type": "integer", "enum": [0, 1]},
                    "quantity": {"type": ["string", "null"]},
                    "recipient": {"type": ["string", "null"]},
                },
                "required": ["text", "claim_type", "agent", "act", "object", "time", "status", "polarity", "quantity", "recipient"],
            },
        }
    },
    "required": ["claims"],
}

FRAME_SYSTEM = """You decompose sentences from a MEETING SUMMARY into atomic claims and fill a frame for each.
Rules:
1. One claim = one checkable proposition (one agent, one act, one object). Split coordinations ("A and B decided X and Y" -> 4 claims only if each is independently checkable; otherwise 2).
2. Resolve pronouns using the participant list and previous sentences. Never invent names not in the sentence/context.
3. status: proposed (suggested/raised), agreed (group accepted), decided (final decision), rejected, assigned (action item given to someone), deferred (postponed/left open), revised (changed an earlier position), stated (plain fact/report).
4. polarity = 0 only if the claim is negated ("did not", "rejected the idea of" stays polarity 1 with status rejected).
5. agent = who performs/owns the act (speaker or role). For assignments put the owner in "recipient" and the assigner (if stated) in "agent".
6. Copy time expressions and numbers verbatim into time / quantity.
7. Keep claim text short and self-contained. Output JSON only."""

FRAME_USER = """Participants: {participants}
Previous summary sentences (context, do NOT decompose): {context}
Sentence to decompose: "{sentence}"
"""

FRAME_FEWSHOT = [
    ("Participants: Project Manager, Marketing, Industrial Designer, User Interface\nPrevious summary sentences (context, do NOT decompose): The team discussed the remote control.\nSentence to decompose: \"The Marketing expert proposed a 25 euro price, but the group later decided on 12.50 euros.\"",
     {"claims": [
         {"text": "Marketing proposed a price of 25 euros.", "claim_type": "fact", "agent": "Marketing", "act": "propose", "object": "price", "time": None, "status": "proposed", "polarity": 1, "quantity": "25 euros", "recipient": None},
         {"text": "The group decided on a price of 12.50 euros.", "claim_type": "decision", "agent": "group", "act": "decide", "object": "price", "time": "later", "status": "decided", "polarity": 1, "quantity": "12.50 euros", "recipient": None}]}),
    ("Participants: Project Manager, Marketing, Industrial Designer, User Interface\nPrevious summary sentences (context, do NOT decompose): \nSentence to decompose: \"The Industrial Designer will look into battery options before the next meeting.\"",
     {"claims": [
         {"text": "The Industrial Designer is to look into battery options before the next meeting.", "claim_type": "action_item", "agent": None, "act": "look into", "object": "battery options", "time": "before the next meeting", "status": "assigned", "polarity": 1, "quantity": None, "recipient": "Industrial Designer"}]}),
]

# --------------------------------------------------------------------------- Stage 2 (optional LLM edge typing)
EDGE_SCHEMA = {
    "type": "object",
    "properties": {"relation": {"type": "string", "enum": ["before", "causes", "task_depends", "revises", "conflicts", "coref", "none"]},
                   "confidence": {"type": "number"}},
    "required": ["relation", "confidence"],
}
EDGE_SYSTEM = """Given two claims from the same meeting summary, output the single strongest relation from claim A to claim B:
before (A happens earlier than B), causes (A is the reason for B), task_depends (B's task needs A done first),
revises (B changes/overrides A), conflicts (A and B cannot both be true), coref (same event described twice), none.
Output JSON only."""
EDGE_USER = "Claim A: {a}\nClaim B: {b}"

# --------------------------------------------------------------------------- Stage 5 (adaptive judge)
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["SUPPORTED", "UNSUPPORTED", "INSUFFICIENT"]},
        "faithful_prob": {"type": "number"},
        "error_type": {"type": "string", "enum": ["none", "entity", "negation", "attribution_swap", "time_shift",
                                                  "revision_ignored", "status_flip", "fabrication", "other"]},
        "evidence_uids": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "faithful_prob", "error_type", "evidence_uids", "rationale"],
}
JUDGE_SYSTEM = """You verify one claim from a meeting summary against transcript excerpts.
A claim is SUPPORTED only if every slot is supported: who (speaker/owner), what, when, the status (proposed vs decided vs rejected), and it is the LATEST position (not one later revised).
UNSUPPORTED if any slot is contradicted or if the excerpts show it was later changed.
INSUFFICIENT if the excerpts do not mention it at all.
Cite utterance ids. faithful_prob is your probability the claim is fully faithful. Output JSON only."""
JUDGE_USER = """Claim: "{claim}"
Frame: agent={agent}; act={act}; object={obj}; time={time}; status={status}; polarity={polarity}; quantity={quantity}; recipient={recipient}

Transcript excerpts (chronological):
{evidence}

Related claims in the same summary and their current belief:
{neighbours}"""

# --------------------------------------------------------------------------- Stage 0 (summaries for MeetVerify-Human)
SUMMARY_SYSTEM = """You write abstractive meeting summaries. Cover: main topics, decisions (with who decided), action items (with owners and deadlines), and any changes of plan. 8-15 sentences. Use participant roles as names."""
SUMMARY_USER = "Participants: {participants}\nTranscript:\n{transcript}\n\nSummary:"

# --------------------------------------------------------------------------- Bench (fabrication injection)
FABRICATE_SCHEMA = {"type": "object", "properties": {"sentence": {"type": "string"}}, "required": ["sentence"]}
FABRICATE_SYSTEM = """Write ONE plausible sentence that could appear in this meeting's summary but is NOT stated or implied in the transcript. Same style and participants. Output JSON."""
FABRICATE_USER = "Participants: {participants}\nSummary so far:\n{summary}\n"


def frame_messages(sentence, participants, context):
    """Returns (system, user) with few-shot examples folded into the system prompt
    (llama.cpp chat templates handle a single system turn more reliably)."""
    import json
    shots = "\n\n".join(f"EXAMPLE INPUT:\n{u}\nEXAMPLE OUTPUT:\n{json.dumps(o)}" for u, o in FRAME_FEWSHOT)
    system = FRAME_SYSTEM + "\n\n" + shots
    user = FRAME_USER.format(participants=", ".join(participants), context=" ".join(context[-3:]), sentence=sentence)
    return system, user


# --------------------------------------------------------------------------- Bench (LLM entity corruption)
ENTITY_SCHEMA = {"type": "object", "properties": {"sentence": {"type": "string"}, "changed": {"type": "string"}},
                 "required": ["sentence", "changed"]}
ENTITY_SYSTEM = """Rewrite the sentence from a meeting summary so that EXACTLY ONE concrete detail (an object, feature, material, place, number or name of a thing) is replaced by a different but plausible one from the same domain. Keep the grammar, tense, speaker and length. Do not negate, do not change who did it, do not change dates or the decision status. The result must read naturally. Output JSON with the new sentence and the replaced detail."""