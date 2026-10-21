"""A small AMI-style meeting used by unit tests and the smoke run (no data download needed)."""
from cprove.schema import Utterance, Meeting

PM, ME, ID, UI = "Project Manager", "Marketing", "Industrial Designer", "User Interface"

LINES = [
    (PM, "Okay, let's start. Today we need to decide on the price and the battery for the remote control."),
    (ME, "From the market research, I propose a selling price of 25 euros."),
    (ID, "Twenty five seems high for a remote."),
    (UI, "Yeah, I think customers would not pay that much."),
    (PM, "What about the battery? Any suggestions?"),
    (ID, "I suggest a kinetic battery, it is more sustainable."),
    (UI, "Kinetic sounds good to me."),
    (ME, "I agree, kinetic fits the trendy image."),
    (PM, "Okay, so we go with the kinetic battery. That's decided."),
    (ME, "Back to the price, maybe we should go lower."),
    (PM, "Let's change the price to 12.50 euros instead, that is our final decision."),
    (ID, "Okay, 12.50 is fine with me."),
    (UI, "Yes, agreed."),
    (PM, "Now the case material. Should it be plastic or rubber?"),
    (ID, "Rubber is better for the double curved case."),
    (UI, "But rubber is more expensive."),
    (PM, "We will not decide the case material today, let's leave it open until the next meeting."),
    (PM, "Action items. The Industrial Designer will prepare a prototype of the kinetic battery design by Friday."),
    (ID, "Okay, I'll do the prototype by Friday."),
    (PM, "The User Interface designer will draw the button layout after the prototype is ready."),
    (UI, "Sure, I will draw the button layout once the prototype is done."),
    (PM, "And Marketing will send the trend report on Monday."),
    (ME, "Yes, I'll send the trend report on Monday."),
    (PM, "Good. The meeting is closed. See you next time."),
]


def toy_meeting() -> Meeting:
    utts = [Utterance(uid=f"TOY_u{i:03d}", speaker=s, text=t, start=6.0 * i, end=6.0 * i + 5, idx=i)
            for i, (s, t) in enumerate(LINES)]
    return Meeting("TOY001", utts, {"participants": [PM, ME, ID, UI], "corpus": "toy"})


FAITHFUL = (
    "Marketing proposed a selling price of 25 euros. "
    "The group decided on a kinetic battery. "
    "The Project Manager changed the price to 12.50 euros instead. "
    "The case material was left open until the next meeting. "
    "The Industrial Designer will prepare a prototype by Friday. "
    "The User Interface designer will draw the button layout after the prototype is ready. "
    "Marketing will send the trend report on Monday."
)

# one error per sentence (sentence index -> error type)
CORRUPTED = (
    "The Industrial Designer proposed a selling price of 25 euros. "         # attribution swap
    "The group rejected the kinetic battery. "                               # status flip
    "The group decided on a price of 25 euros. "                             # revision ignored
    "The case material was decided as rubber. "                              # status flip / fabrication
    "The Industrial Designer will prepare a prototype by Monday. "           # time shift
    "The User Interface designer will draw the button layout after the prototype is ready. "  # ok (but depends on wrong claim)
    "Marketing will send the trend report on Monday."                        # ok
)
CORRUPTED_GOLD = [0, 0, 0, 0, 0, 1, 1]


# ---------------------------------------------------------------- parametrised toy corpus (smoke tests)
def toy_corpus(n: int = 8, seed: int = 0):
    """n AMI-style meetings with different values, plus a faithful reference summary for each.
    Returns (meetings, summaries[{summary_id, meeting_id, system, summary}])."""
    import random
    rng = random.Random(seed)
    meetings, sums = [], []
    for k in range(n):
        p1, p2 = rng.choice([(25, "12.50"), (30, "18"), (40, "22"), (35, "15")])
        bat = rng.choice(["kinetic", "solar", "rechargeable", "lithium"])
        mat, alt = rng.sample(["rubber", "plastic", "wood", "titanium"], 2)
        d1, d2 = rng.sample(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], 2)
        roles = [PM, ME, ID, UI]
        proposer = rng.choice([ME, ID])
        lines = [
            (PM, f"Okay, let's start. Today we need to decide on the price and the battery for the remote control."),
            (proposer, f"From the market research, I propose a selling price of {p1} euros."),
            (UI if proposer != UI else ID, f"{p1} seems high for a remote."),
            (UI, "Yeah, I think customers would not pay that much."),
            (PM, "What about the battery? Any suggestions?"),
            (ID, f"I suggest a {bat} battery, it is more sustainable."),
            (UI, f"{bat.capitalize()} sounds good to me."),
            (ME, f"I agree, {bat} fits the trendy image."),
            (PM, f"Okay, so we go with the {bat} battery. That's decided."),
            (ME, "Back to the price, maybe we should go lower."),
            (PM, f"Let's change the price to {p2} euros instead, that is our final decision."),
            (ID, f"Okay, {p2} is fine with me."),
            (UI, "Yes, agreed."),
            (PM, f"Now the case material. Should it be {mat} or {alt}?"),
            (ID, f"{mat.capitalize()} is better for the double curved case."),
            (UI, f"But {mat} is more expensive."),
            (PM, "We will not decide the case material today, let's leave it open until the next meeting."),
            (PM, f"Action items. The Industrial Designer will prepare a prototype of the battery design by {d1}."),
            (ID, f"Okay, I'll do the prototype by {d1}."),
            (PM, "The User Interface designer will draw the button layout after the prototype is ready."),
            (UI, "Sure, I will draw the button layout once the prototype is done."),
            (PM, f"And Marketing will send the trend report on {d2}."),
            (ME, f"Yes, I'll send the trend report on {d2}."),
            (PM, "Good. The meeting is closed. See you next time."),
        ]
        mid = f"TS{1000 + k}a"
        utts = [Utterance(uid=f"{mid}_u{i:03d}", speaker=s, text=t, start=6.0 * i, end=6.0 * i + 5, idx=i)
                for i, (s, t) in enumerate(lines)]
        meetings.append(Meeting(mid, utts, {"participants": roles, "corpus": "toy"}))
        summ = (f"{proposer} proposed a selling price of {p1} euros. "
                f"The group decided on a {bat} battery. "
                f"The Project Manager changed the price to {p2} euros instead. "
                f"The case material was left open until the next meeting. "
                f"The Industrial Designer will prepare a prototype by {d1}. "
                f"The User Interface designer will draw the button layout after the prototype is ready. "
                f"Marketing will send the trend report on {d2}.")
        sums.append({"summary_id": f"{mid}__ref", "meeting_id": mid, "system": "reference", "summary": summ})
    return meetings, sums
