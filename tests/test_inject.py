import random
from dataclasses import asdict
from tests.fixtures import toy_meeting, FAITHFUL
from bench.inject import inject_one, INJECTORS


def test_each_injection_changes_exactly_flagged_sentences():
    m = toy_meeting(); md = {"meeting_id": m.meeting_id, "utterances": [asdict(u) for u in m.utterances], "meta": m.meta}
    rng = random.Random(0); seen = set()
    for et in INJECTORS:
        for _ in range(3):
            r = inject_one(FAITHFUL, md, et, rng)
            if r:
                seen.add(et); assert r["gold"].count(0) == len(r["sent_idx"]) and len(r["sentences"]) == len(r["gold"])
    assert len(seen) >= 7
    r = inject_one(FAITHFUL, md, "attribution_swap", rng); s = r["sentences"][r["sent_idx"][0]]
    assert "Industrial Designer and Industrial Designer" not in s
