"""E9 graph diagnostics (CPU, ~2 min, no GPU, no training).
  python -m train.graph_diag
(1) edge inventory: count per type / source, per summary, share of edges between claims of the SAME sentence, share created by
    order or lexical overlap only (types before / coref / same_agent) -> outputs/table_edges.tex (precision column filled from
    outputs/edge_precision.json if present, else '--').
(2) sensitivity of the hand-weight HL-MRF AUROC (523 labelled claims): no graph, full graph, each edge type removed,
    top-50% edges per summary by weight, graph-rule strength x{0.5,1,2} (multiplies the potential of every graph grounding).
    Paired meeting-clustered bootstrap against the no-graph posterior -> outputs/table_graph_sens.tex."""
import argparse, glob, json, pickle
from collections import Counter, defaultdict
import numpy as np
from cprove.utils import load_config
from cprove import metrics as M
from cprove.hlmrf import HLMRF, Weights
from cprove.rules import ground
from train.nested_cv import load_rows

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/frozen.yaml"); ap.add_argument("--features", default="cache/frozen/features.jsonl")
ap.add_argument("--items", nargs="*", default=None)
a = ap.parse_args()
cfg = load_config(a.config); h = cfg["hlmrf"]
model = HLMRF(Weights(h["init_weights"]), h["iters"], h["tol"])
files = a.items or sorted(glob.glob(str(cfg["paths"]["cache"]) + "/items_*.pkl"))
items = [it for f in files for it in pickle.load(open(f, "rb"))]
rows = load_rows(a.features); lab = {r["cid"]: int(r["author_label"] == "SUPPORTED") for r in rows}
mid = {r["cid"]: r["meeting_id"] for r in rows}
print(f"{len(items)} summaries from {files}; {len(lab)} labelled claims")

# ---------------- (1) inventory
cnt, src, same, per = Counter(), defaultdict(Counter), Counter(), defaultdict(list)
for it in items:
    sent = {c.cid: c.sent_idx for c in it.claims}; n = Counter()
    for e in it.edges:
        cnt[e.etype] += 1; src[e.etype][e.source] += 1; n[e.etype] += 1
        if e.src in sent and e.dst in sent and sent[e.src] == sent[e.dst]:
            same[e.etype] += 1
    for t in cnt: per[t].append(n[t])
N = len(items); tot = sum(cnt.values())
print(f"\n{tot} edges over {N} summaries ({tot / N:.1f} per summary)")
print(f"{'type':13}{'count':>7}{'per summ':>10}{'same-sentence':>15}  sources")
for t, c in cnt.most_common():
    print(f"{t:13}{c:>7}{c / N:>10.2f}{same[t] / c:>15.0%}  {dict(src[t])}")
prec = {}
try:
    prec = json.load(open("outputs/edge_precision.json"))
except Exception:
    pass
L = [r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule", r"Edge type & Count & Per summary & Same sentence & Precision & Flagged \\", r"\midrule"]
FLAG = {"before": "order only", "coref": "lexical overlap", "same_agent": "shared speaker", "similar": "similarity"}
for t, c in cnt.most_common():
    pr = prec.get(t); ps = f"{pr:.2f}" if isinstance(pr, (int, float)) else "--"
    L.append(f"{t.replace('_', ' ')} & {c} & {c / N:.2f} & {same[t] / c:.0%} & {ps} & {FLAG.get(t, '--')} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]; open("outputs/table_edges.tex", "w").write("\n".join(L))

# ---------------- (2) sensitivity
def run(variant):
    y = {}
    for it in items:
        edges = list(it.edges)
        if variant[0] == "none": edges = []
        elif variant[0] == "drop": edges = [e for e in edges if e.etype != variant[1]]
        elif variant[0] == "top50": edges = sorted(edges, key=lambda e: -e.weight)[: max(0, len(edges) // 2)] if edges else []
        pr = ground(it.claims, it.obs, edges, it.order_ok)
        if variant[0] == "scale":
            for g in pr.groundings:
                if "etype" in g.meta: g.mult *= variant[1]
        yy, _ = model.infer(pr)
        for c in it.claims:
            if c.cid in lab: y[c.cid] = float(yy[pr.y_index(c.cid)])
    return y

cids = sorted(lab); yt = np.array([lab[c] for c in cids]); gg = np.array([mid[c] for c in cids])
variants = [("none",), ("full",)] + [("drop", t) for t in cnt] + [("top50",)] + [("scale", f) for f in (0.5, 2.0)]
names = {("none",): "no graph", ("full",): "full graph (reference)", ("top50",): "top 50\\% edges per summary by weight",
         ("scale", 0.5): "graph strength $\\times 0.5$", ("scale", 2.0): "graph strength $\\times 2$"}
res = {v: run(v) for v in variants}
base = np.array([res[("none",)][c] for c in cids])
print(f"\n{'variant':36}{'AUROC':>7}{'d vs no graph [95% CI]':>28}{'p':>7}{'flips':>7}")
T = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule", r"Graph variant & AUROC & $\Delta$ vs no graph [95\% CI] & Verdict flips \\", r"\midrule"]
for v in variants:
    q = np.array([res[v][c] for c in cids]); nm = names.get(v) or f"drop {v[1].replace('_', ' ')} edges"
    flips = int(((q >= 0.5) != (base >= 0.5)).sum())
    if v == ("none",):
        print(f"{nm:36}{M.auroc(yt, q):>7.3f}"); T.append(f"{nm} & {M.auroc(yt, q):.3f} & -- & -- \\\\"); continue
    d, lo, hi, p = M.paired_bootstrap(M.auroc, yt, q, base, gg, 1000)
    print(f"{nm:36}{M.auroc(yt, q):>7.3f}{d:>+10.4f} [{lo:+.4f},{hi:+.4f}]{p:>7.3f}{flips:>7}")
    T.append(f"{nm} & {M.auroc(yt, q):.3f} & {d:+.3f} [{lo:+.3f}, {hi:+.3f}] & {flips} \\\\")
T += [r"\bottomrule", r"\end{tabular}"]; open("outputs/table_graph_sens.tex", "w").write("\n".join(T))
json.dump({"counts": dict(cnt), "per_summary": {t: c / N for t, c in cnt.items()}, "same_sentence": {t: same[t] / c for t, c in cnt.items()}},
          open("outputs/graph_diag.json", "w"), indent=1)
print("wrote outputs/table_edges.tex, outputs/table_graph_sens.tex")
