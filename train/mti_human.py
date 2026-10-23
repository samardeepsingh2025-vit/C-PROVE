"""E11: summary-level trust vs INDEPENDENT human summary ratings (CPU, ~1 min). Needs the two rating CSVs (bench.fresh.make_summary_sheet).
  python -m train.mti_human --ratings data/meetverify_v2/summary_rating/summary_ratings_B1.csv data/meetverify_v2/summary_rating/summary_ratings_B2.csv
Out-of-fold fusion trust per claim -> summary-level indices: omega-weighted mean trust (omega: decision/action_item 2, else 1; this is the
claim-level MTI of the paper, the five pipeline components are NOT recomputed), mean trust, min trust, mean MiniCheck.
Reports: inter-annotator agreement; Spearman / Kendall vs mean human factuality; AUROC for needs-review (any annotator = yes);
Dirichlet weight sensitivity (200 random weight vectors over claim types); claim-type weights fitted by NNLS inside leave-one-meeting-out folds;
missed-proposition counts. -> outputs/table_mti_human.tex"""
import argparse, csv, json
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, kendalltau
from scipy.optimize import nnls
from cprove import metrics as M
from train.nested_cv import load_rows, run


def corr(v, h):
    return spearmanr(v, h)[0], kendalltau(v, h)[0]


def wkappa(a, b, k=5):
    a, b = np.asarray(a, int) - 1, np.asarray(b, int) - 1
    O = np.zeros((k, k))
    for x, y in zip(a, b): O[x, y] += 1
    O /= O.sum(); E = np.outer(O.sum(1), O.sum(0)); W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    return 1 - (W * O).sum() / (W * E).sum()


def ckappa(a, b):
    a, b = np.asarray(a), np.asarray(b); po = (a == b).mean(); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def analyse(S, trust_by, claim_trust, types, claim_types, hum, rev, mid, rng, n_dir=200):
    """S: summary keys; trust_by: dict name->{summary: array of claim trust}; types: dict summary->array of claim types;
    hum: mean factuality per summary; rev: needs-review (0/1) per summary."""
    res = {}
    for nm, d in trust_by.items():
        v = np.array([d[s] for s in S]); rho, tau = corr(v, hum)
        auc = M.auroc(1 - rev, v) if rev.min() != rev.max() else float("nan")
        res[nm] = dict(spearman=rho, kendall=tau, auroc_review=auc)
    # Dirichlet over claim types
    base = claim_trust; ds = []
    for _ in range(n_dir):
        w = dict(zip(claim_types, rng.dirichlet(np.ones(len(claim_types)))))
        v = [float(np.dot([w[t] for t in types[s]], base[s]) / sum(w[t] for t in types[s])) for s in S]
        ds.append(spearmanr(v, hum)[0])
    res["dirichlet"] = dict(min=float(np.min(ds)), q05=float(np.quantile(ds, .05)), median=float(np.median(ds)), q95=float(np.quantile(ds, .95)), max=float(np.max(ds)))
    # NNLS inside leave-one-meeting-out folds: summary score = sum_t w_t * mean trust of type t
    X = np.array([[np.mean([x for x, t in zip(base[s], types[s]) if t == ct]) if ct in types[s] else np.mean(base[s]) for ct in claim_types] for s in S])
    pred = np.zeros(len(S)); mids = np.array([mid[s] for s in S])
    for m in set(mids):
        tr = mids != m; w, _ = nnls(np.c_[X[tr], np.ones(tr.sum())], hum[tr]); pred[~tr] = np.c_[X[~tr], np.ones((~tr).sum())] @ w
    rho, tau = corr(pred, hum); res["fitted_in_folds"] = dict(spearman=rho, kendall=tau, auroc_review=M.auroc(1 - rev, pred) if rev.min() != rev.max() else float("nan"))
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ratings", nargs=2); ap.add_argument("--gold", default="data/meetverify_v2/gold_summaries.jsonl", help="existing campaign ratings (used when --ratings is not given)"); ap.add_argument("--features", default="cache/frozen/features.jsonl")
    a = ap.parse_args()
    rows = load_rows(a.features); out, _ = run(rows); p = np.asarray(out["selected"], float)
    sid = np.array([r["summary_id"] for r in rows])
    if a.ratings:
        R = [{r["key"]: r for r in csv.DictReader(open(f, encoding="utf-8"))} for f in a.ratings]
        S = [s for s in sorted(set(sid)) if all(s in r and r[s]["factuality"] and r[s]["needs_review"] for r in R)]
        f1, f2 = [[int(r[s]["factuality"]) for s in S] for r in R]; n1, n2 = [[int(r[s]["needs_review"] == "yes") for s in S] for r in R]
        miss = [[int(r[s]["n_missed"]) for s in S] for r in R]
    else:
        G = {}
        for l in open(a.gold, encoding="utf-8"):
            g = json.loads(l); G[g["summary_id"]] = g
        S = [s for s in sorted(set(sid)) if s in G]
        f1 = [G[s]["rating_a"] for s in S]; f2 = [G[s]["rating_b"] for s in S]
        n1 = [int(G[s]["needs_review_a"]) for s in S]; n2 = [int(G[s]["needs_review_b"]) for s in S]; miss = None
        print(f"using the existing campaign ratings: {len(S)} of the {len(set(sid))} labelled summaries were rated by both annotators")
    if len(S) < 10: print("WARNING: fewer than 10 summaries - correlations are not interpretable; report descriptively only")
    print(f"{len(S)} summaries rated by both annotators (of {len(set(sid))})")
    print(f"agreement: factuality Spearman {spearmanr(f1, f2)[0]:.2f}, quadratic-weighted kappa {wkappa(f1, f2):.2f}; needs-review Cohen kappa {ckappa(n1, n2):.2f}")
    hum = (np.array(f1) + np.array(f2)) / 2; rev = ((np.array(n1) + np.array(n2)) >= 1).astype(int)
    print(f"mean factuality {hum.mean():.2f}; needs review (either annotator) {rev.sum()}/{len(S)}")
    for i, ms in enumerate(miss or []):
        print(f"annotator {i + 1}: missed propositions mean {np.mean(ms):.2f} per summary; summaries with >=1 missed {sum(m > 0 for m in ms)}/{len(S)}")
    claim_types = sorted(set(r["claim_type"] for r in rows)); omega = {t: 2.0 if t in ("decision", "action_item") else 1.0 for t in claim_types}
    tr, ty, mc, mid = {}, {}, {}, {}
    for s in S:
        ix = np.where(sid == s)[0]; tr[s] = p[ix]; ty[s] = [rows[i]["claim_type"] for i in ix]; mc[s] = np.array([rows[i].get("mc", np.nan) for i in ix]); mid[s] = rows[ix[0]]["meeting_id"]
    tb = {"MTI (omega-weighted mean trust)": {s: float(np.dot([omega[t] for t in ty[s]], tr[s]) / sum(omega[t] for t in ty[s])) for s in S},
          "mean trust (per-claim)": {s: float(tr[s].mean()) for s in S}, "min trust": {s: float(tr[s].min()) for s in S}}
    if not any(np.isnan(mc[s]).any() for s in S): tb["mean MiniCheck"] = {s: float(mc[s].mean()) for s in S}
    res = analyse(S, tb, tr, ty, claim_types, hum, rev, mid, np.random.default_rng(0))
    print(f"\n{'index':34}{'Spearman':>10}{'Kendall':>9}{'AUROC(needs review)':>21}")
    for nm in tb: r = res[nm]; print(f"{nm:34}{r['spearman']:>10.3f}{r['kendall']:>9.3f}{r['auroc_review']:>21.3f}")
    d = res["dirichlet"]; print(f"Dirichlet claim-type weights x200: Spearman min {d['min']:.3f}, 5% {d['q05']:.3f}, median {d['median']:.3f}, 95% {d['q95']:.3f}, max {d['max']:.3f}")
    f = res["fitted_in_folds"]; print(f"claim-type weights fitted by NNLS in leave-one-meeting-out folds: Spearman {f['spearman']:.3f}, Kendall {f['kendall']:.3f}, AUROC(needs review) {f['auroc_review']:.3f}")
    L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Summary-level index & Spearman & Kendall & AUROC (needs review) \\", r"\midrule"]
    for nm in tb: r = res[nm]; L.append(f"{nm} & {r['spearman']:.2f} & {r['kendall']:.2f} & {r['auroc_review']:.2f} \\\\")
    L.append(f"Type weights fitted in folds & {f['spearman']:.2f} & {f['kendall']:.2f} & {f['auroc_review']:.2f} \\\\"); L += [r"\bottomrule", r"\end{tabular}"]
    open("outputs/table_mti_human.tex", "w").write("\n".join(L)); json.dump(res, open("outputs/mti_human.json", "w"), indent=1, default=float)
    print("wrote outputs/table_mti_human.tex")


if __name__ == "__main__":
    main()
