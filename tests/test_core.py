import numpy as np, torch
from cprove.schema import Claim, Frame, Edge, Observation
from cprove.rules import ground
from cprove.hlmrf import HLMRF, Weights, Compiled
from cprove.mti import mti
from cprove.conformal import CRCLabeler
from cprove.text import normalise_time, time_compatible, split_sentences
from cprove.io import parse_transcript
from cprove.utils import load_config

INIT = load_config("configs/smoke.yaml")["hlmrf"]["init_weights"]


def _claims(n=3):
    return [Claim(cid=f"c{i}", text=f"claim {i}", summary_id="s", sent_idx=i) for i in range(n)]


def _obs(c, ent, con, fact):
    return Observation(cid=c.cid, ent=ent, con=con, fact=fact)


def test_solver_bounds_and_monotone_energy():
    cl = _claims(); obs = {c.cid: _obs(c, .8, .1, .8) for c in cl}
    p = ground(cl, obs, [Edge("c0", "c1", "task_depends", .9)])
    comp = Compiled.from_program(p); lam = Weights(INIT)().detach()
    y0 = torch.full((comp.n,), .5)
    y = comp.solve(lam, y0, 200)
    assert float(y.min()) >= 0 and float(y.max()) <= 1
    assert float(comp.energy(y, lam)) <= float(comp.energy(y0, lam)) + 1e-6


def test_good_evidence_beats_bad_evidence():
    cl = _claims(2)
    obs = {"c0": _obs(cl[0], .95, .02, .95), "c1": _obs(cl[1], .05, .9, .05)}
    m = HLMRF(Weights(INIT)); p = ground(cl, obs, [])
    y, _ = m.infer(p); post = m.posteriors(p, y)
    assert post["c0"]["y"] > .6 > .4 > post["c1"]["y"] or post["c0"]["y"] - post["c1"]["y"] > .3


def test_dependency_propagates_failure():
    cl = _claims(2)
    obs = {c.cid: _obs(c, .9, .02, .9) for c in cl}; obs["c0"] = _obs(cl[0], .05, .9, .05)
    m = HLMRF(Weights(INIT))
    y_dep, _ = m.infer(ground(cl, obs, [Edge("c0", "c1", "task_depends", 1.0)]))
    p_ind = ground(cl, obs, []); y_ind, _ = m.infer(p_ind)
    p_dep = ground(cl, obs, [Edge("c0", "c1", "task_depends", 1.0)])
    assert m.posteriors(p_dep, y_dep)["c1"]["y"] < m.posteriors(p_ind, y_ind)["c1"]["y"]


def test_judge_sensitivity_shape():
    cl = _claims(3); obs = {c.cid: _obs(c, .5, .2, .5) for c in cl}
    p = ground(cl, obs, [Edge("c0", "c1", "task_depends", .9)]); m = HLMRF(Weights(INIT))
    y, _ = m.infer(p); J, rows = m.judge_sensitivity(p, y, ["c0", "c1"])
    assert J.shape == (3, 2) and np.isfinite(J).all()
    assert len(p.groundings) == len(ground(cl, obs, [Edge("c0", "c1", "task_depends", .9)]).groundings)  # judge atoms removed


def test_weight_learning_reduces_loss():
    progs, gold = [], []
    for k in range(20):
        cl = _claims(2); good = k % 2 == 0
        obs = {"c0": _obs(cl[0], .7 if good else .3, .1 if good else .5, .7 if good else .3), "c1": _obs(cl[1], .8, .05, .8)}
        progs.append(ground(cl, obs, [])); gold.append({"c0": int(good), "c1": 1})
    m = HLMRF(Weights(INIT)); before = m.eval_loss(progs, gold)
    m.fit(progs, gold, epochs=8, lr=.1, verbose=False)
    assert m.eval_loss(progs, gold) < before


def test_mti_weighted_mean():
    cl = _claims(2); cl[0].claim_type = "decision"
    post = {"c0": {"y": 1.0}, "c1": {"y": 0.0}}
    assert abs(mti(cl, [], post, {"decision": 2.0}) - 2 / 3) < 1e-9


def test_crc_risk_controlled():
    rng = np.random.default_rng(0); g = rng.integers(0, 2, 2000)
    y = np.clip(g * .6 + .2 + rng.normal(0, .2, 2000), 0, 1)
    c = CRCLabeler(.1).fit(y[:1000], g[:1000]); r = c.report(y[1000:], g[1000:])
    assert r["risk"] <= .12


def test_time_and_split_and_parse():
    assert normalise_time("by Friday") == "WD:friday" and time_compatible("WD:friday", "WD:monday") == 0.0
    assert len(split_sentences("A did x. B did y.")) == 2
    m = parse_transcript("PM: hello there\nME: hi\ncontinued"); assert len(m.utterances) == 2 and "continued" in m.utterances[1].text
