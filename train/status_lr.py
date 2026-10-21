"""Status classifier as logistic regression on BGE embeddings (stable fallback when DeBERTa fine-tuning won't learn).
Features = [embedding of the >>-marked centre utterance, embedding of the whole window].
  python -m train.status_lr --data data\\silver\\status.jsonl --out models\\status_clf_lr.pkl
"""
import argparse, json, pickle, random
from pathlib import Path
import numpy as np

LABELS = ["proposed", "agreed", "decided", "rejected", "assigned", "deferred", "revised", "stated"]


def split_center(text: str):
    lines = text.split("\n")
    c = [l[3:] for l in lines if l.startswith(">> ")]
    win = " ".join(l.strip().lstrip("> ").strip() for l in lines)
    return (c[0] if c else win), win


def featurize(embed, texts):
    cw = [split_center(t) for t in texts]
    return np.hstack([embed([c for c, _ in cw]), embed([w for _, w in cw])])


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    from cprove.utils import device
    enc = SentenceTransformer(a.encoder, device=device())
    embed = lambda xs: enc.encode(xs, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    rows = [r for r in rows if r["label"] in LABELS]
    mids = sorted({r["meeting_id"] for r in rows}); random.Random(0).shuffle(mids)
    dev_m = set(mids[: max(1, len(mids) // 8)])
    tr = [r for r in rows if r["meeting_id"] not in dev_m]; dv = [r for r in rows if r["meeting_id"] in dev_m]
    print("train", len(tr), "dev", len(dv))
    Xtr, Xdv = featurize(embed, [r["text"] for r in tr]), featurize(embed, [r["text"] for r in dv])
    ytr, ydv = [r["label"] for r in tr], [r["label"] for r in dv]
    best = None
    for C in (0.3, 1, 3, 10, 30):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(Xtr, ytr)
        f1 = f1_score(ydv, clf.predict(Xdv), average="macro")
        print(f"C={C}: dev macro-F1 {f1:.3f}")
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    f1, C, clf = best
    print(classification_report(ydv, clf.predict(Xdv), zero_division=0))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"clf": clf, "encoder": a.encoder, "labels": list(clf.classes_), "C": C, "dev_macro_f1": f1}, open(a.out, "wb"))
    print("saved", a.out, "dev macro-F1", round(f1, 3))


if __name__ == "__main__":
    main()
