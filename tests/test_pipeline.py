from tests.fixtures import toy_meeting, FAITHFUL, CORRUPTED
from cprove.utils import load_config
from cprove.pipeline import CProve


def test_corrupted_scores_lower_than_faithful():
    cp = CProve(load_config("configs/smoke.yaml"))
    a = cp.verify(toy_meeting(), FAITHFUL, "f", scheduler="none")
    b = cp.verify(toy_meeting(), CORRUPTED, "c", scheduler="none")
    assert a.mti > b.mti
    assert all(0 <= p["y"] <= 1 for p in a.post.values())
    assert all(a.proofs[c.cid] for c in a.claims)
