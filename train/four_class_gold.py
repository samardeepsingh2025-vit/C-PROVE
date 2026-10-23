"""4-class check against the adjudicated gold labels (CPU, seconds, no refit).

  python -m train.four_class_gold --preds outputs/nested_cv_552.preds.npy
Uses OUT-OF-FOLD fusion trust. Rule: CONFLICTING if p_ent>=0.5 and p_con>=0.5; else SUPPORTED if trust>=tau; else CONTRADICTED if p_con>=theta;
else INSUFFICIENT.  tau and theta are chosen ONLY on the claims that are not in the gold set:
  tau   = threshold maximising binary macro-F1 of the trust score (supported vs not);
  theta = p_con cut-off maximising 3-class macro-F1 against PROXY labels derived from the author's labels
          (SUPPORTED; UNVERIFIABLE -> INSUFFICIENT; UNSUPPORTED -> CONTRADICTED unless error type is 'fabrication' -> INSUFFICIENT).
The proxy mapping is an assumption of ours and is stated in the paper. Compared with the same rule built on entailment-only and MiniCheck trust."""
import argparse, json
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from cprove import metrics as M
from train.nested_cv import best_threshold

CL = ["SUPPORTED", "CONTRADICTED", "CONFLICTING", "INSUFFICIENT"]
ap = argparse.ArgumentParser()
ap.add_argument("--preds", default="outputs/nested_cv_552.preds.npy"); ap.add_argument("--features", default="cache/frozen/features.jsonl")
ap.add_argument("--gold", default="data/meetverify_v2/agreement/gold_claims.jsonl"); ap.add_argument("--boots", type=int, default=2000)
a = ap.parse_args()
P = np.load(a.preds, allow_pickle=True).item(); cids = P.pop("cids"); pos = {c: i for i, c in enumerate(cids)}
F = {}
for l in open(a.features, encoding="utf-8"):
    r = json.loads(l); F.setdefault(r["cid"], r)
gold = {}
for l in open(a.gold, encoding="utf-8"):
    r = json.loads(l)
    if r["set"] == "old" and r["cid"] in pos:
        gold[r["cid"]] = r["label"]
gid = sorted(gold); oid = [c for c in cids if c not in gold and F[c].get("author_label") in ("SUPPORTED", "UNSUPPORTED", "UNVERIFIABLE")]
print(f"gold claims {len(gid)} (4-class counts {dict(zip(*np.unique([gold[c] for c in gid], return_counts=True)))}); tuning claims {len(oid)}")


def proxy(c):
    r = F[c]; lab = r["author_label"]
    if lab == "SUPPORTED": return "SUPPORTED"
    if lab == "UNVERIFIABLE": return "INSUFFICIENT"
    return "INSUFFICIENT" if (r.get("author_error") or "") == "fabrication" else "CONTRADICTED"


def rule(trust, ent, con, tau, theta):
    out = []
    for t, e, c in zip(trust, ent, con):
        out.append("CONFLICTING" if (e >= 0.5 and c >= 0.5) else "SUPPORTED" if t >= tau else "CONTRADICTED" if c >= theta else "INSUFFICIENT")
    return np.array(out)


def macro(y, p, labels):
    return float(f1_score(y, p, labels=labels, average="macro", zero_division=0))


grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
sources = {"Fusion (selected)": np.array(P["fusion (selected in inner CV)"], float), "Entailment only": np.array([F[c]["ent"] for c in cids]), "MiniCheck": np.array([F[c]["mc"] for c in cids])}
yg = np.array([gold[c] for c in gid]); grp = np.array([F[c]["meeting_id"] for c in gid]); present = [c for c in CL if (yg == c).any()]
ent_all, con_all = np.array([F[c]["ent"] for c in cids]), np.array([F[c]["con"] for c in cids])
res = {}
print(f"\n{'trust source':20} tau   theta  macro-F1 (classes present in gold: {present})  [95% meeting-clustered CI]   accuracy")
for name, sc in sources.items():
    ok = [pos[c] for c in oid if not np.isnan(sc[pos[c]])]
    yb = np.array([int(F[cids[i]]["author_label"] == "SUPPORTED") for i in ok])
    tau = best_threshold(yb, sc[ok])
    yo = np.array([proxy(cids[i]) for i in ok])
    theta = max(grid, key=lambda th: macro(yo, rule(sc[ok], ent_all[ok], con_all[ok], tau, th), ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"]))
    gi = [pos[c] for c in gid]; pred = rule(sc[gi], ent_all[gi], con_all[gi], tau, theta)
    mf = macro(yg, pred, present); acc = float((pred == yg).mean())
    rng = np.random.default_rng(0); ug = np.unique(grp); bs = []
    for _ in range(a.boots):
        pick = rng.choice(ug, len(ug)); idx = np.concatenate([np.where(grp == g)[0] for g in pick]); bs.append(macro(yg[idx], pred[idx], present))
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    res[name] = dict(tau=float(tau), theta=float(theta), macro_f1=mf, ci=ci, accuracy=acc, pred_counts={c: int((pred == c).sum()) for c in CL},
                     confusion=confusion_matrix(yg, pred, labels=CL).tolist(), per_class_f1=dict(zip(CL, f1_score(yg, pred, labels=CL, average=None, zero_division=0).tolist())))
    print(f"{name:20} {tau:.2f}  {theta:.2f}   {mf:.3f}   [{ci[0]:.3f}, {ci[1]:.3f}]   {acc:.3f}")
maj = max(CL, key=lambda c: (yg == c).sum()); print(f"{'majority class':20} (always {maj})  macro-F1 {macro(yg, np.array([maj] * len(yg)), present):.3f}  accuracy {(yg == maj).mean():.3f}")
r = res["Fusion (selected)"]
print(f"\nFusion confusion matrix (rows = gold, cols = predicted) {CL}:")
for c, row in zip(CL, r["confusion"]):
    print(f"  {c:13} {row}")
print("per-class F1:", {k: round(v, 3) for k, v in r["per_class_f1"].items()}, "| predicted counts:", r["pred_counts"])
import pathlib; pathlib.Path("outputs").mkdir(exist_ok=True); json.dump(res, open("outputs/four_class_gold.json", "w"), indent=1)
