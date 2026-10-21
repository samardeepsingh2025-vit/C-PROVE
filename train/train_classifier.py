"""Fine-tune DeBERTa-v3-base for (a) utterance-window status or (b) claim-pair edge type.

Fits an 8 GB RTX 5060 laptop GPU: bf16/fp16, batch 16, max_len <=192, ~1.2 GB weights+optim
with gradient checkpointing off; ~6-10 min per epoch on 6k examples.

  python -m train.train_classifier --task status --data data/silver/status.jsonl --out models/status_clf
  python -m train.train_classifier --task edges  --data data/silver/edges.jsonl  --out models/edge_clf
Optional --gold data/meetverify/status_gold.jsonl : extra human-labelled rows, upweighted x3.
Split by meeting/summary id (no leakage). Writes eval metrics to <out>/metrics.json.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

STATUS = ["proposed", "agreed", "decided", "rejected", "assigned", "deferred", "revised", "stated"]
EDGES = ["none", "before", "causes", "task_depends", "coref", "revises", "conflicts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["status", "edges"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--gold", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="microsoft/deberta-v3-base")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp32", action="store_true", help="disable bf16/fp16 mixed precision")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,
                              DataCollatorWithPadding)
    from sklearn.metrics import f1_score, accuracy_score, classification_report

    labels = STATUS if a.task == "status" else EDGES
    l2i = {l: i for i, l in enumerate(labels)}
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    if a.gold:
        g = [json.loads(l) for l in open(a.gold, encoding="utf-8")]
        rows += g * 3
    rows = [r for r in rows if r["label"] in l2i]
    grp_key = "meeting_id" if a.task == "status" else "summary_id"
    groups = sorted({r.get(grp_key, str(i)) for i, r in enumerate(rows)})
    rng = np.random.default_rng(a.seed)
    rng.shuffle(groups)
    dev_groups = set(groups[: max(1, len(groups) // 8)])
    tr = [r for i, r in enumerate(rows) if r.get(grp_key, str(i)) not in dev_groups]
    dv = [r for i, r in enumerate(rows) if r.get(grp_key, str(i)) in dev_groups]
    print("train", len(tr), Counter(r["label"] for r in tr))
    print("dev  ", len(dv))

    try:   # the fast-tokenizer conversion can silently fall back to a broken vocab; the slow SentencePiece one is safe
        tok = AutoTokenizer.from_pretrained(a.base, use_fast=False)
    except Exception as e:
        print("slow tokenizer unavailable:", e)
        tok = AutoTokenizer.from_pretrained(a.base)
    _t = tok("Marketing: I propose a price of 25 euros.")["input_ids"]
    print("tokenizer check:", type(tok).__name__, tok.convert_ids_to_tokens(_t))
    if len(set(_t)) < 5 or tok.unk_token_id in _t[1:-1]:
        raise SystemExit("tokenizer is broken (unknown tokens) - fix before training")
    ml = a.max_len or (192 if a.task == "status" else 160)

    def enc(batch):
        if a.task == "status":
            e = tok(batch["text"], truncation=True, max_length=ml)
        else:
            e = tok(batch["a"], batch["b"], truncation=True, max_length=ml)
        e["labels"] = [l2i[l] for l in batch["label"]]
        return e

    cols = ["text", "label"] if a.task == "status" else ["a", "b", "label"]
    dtr = Dataset.from_list([{k: r[k] for k in cols} for r in tr]).map(enc, batched=True, remove_columns=cols)
    ddv = Dataset.from_list([{k: r[k] for k in cols} for r in dv]).map(enc, batched=True, remove_columns=cols)

    # class-balanced loss (status is dominated by 'stated', edges by 'none')
    cnt = Counter(l2i[r["label"]] for r in tr)
    w = torch.tensor([len(tr) / (len(labels) * max(1, cnt[i])) for i in range(len(labels))], dtype=torch.float)
    w = w.clamp(max=10.0)

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            y = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(out.logits.float(), y, weight=w.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    # transformers>=5 loads checkpoint dtype by default (deberta-v3-base is stored in fp16) -> force fp32 master weights
    kw = dict(num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=l2i)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(a.base, dtype=torch.float32, **kw)
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(a.base, torch_dtype=torch.float32, **kw)
    model = model.float()
    bf16 = (not a.fp32) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    args = TrainingArguments(
        output_dir=a.out + "_ckpt", per_device_train_batch_size=a.bs, per_device_eval_batch_size=64,
        gradient_accumulation_steps=1, learning_rate=a.lr, num_train_epochs=a.epochs, warmup_steps=max(1, int(0.06 * a.epochs * -(-len(tr) // a.bs))),
        weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", save_total_limit=1, bf16=bf16, fp16=(not a.fp32) and torch.cuda.is_available() and not bf16,
        logging_steps=50, report_to=[], seed=a.seed, dataloader_num_workers=0)   # workers=0: Windows-safe

    def metrics(p):
        pred = p.predictions.argmax(-1)
        return {"macro_f1": f1_score(p.label_ids, pred, average="macro"), "acc": accuracy_score(p.label_ids, pred)}

    tr_ = WTrainer(model=model, args=args, train_dataset=dtr, eval_dataset=ddv, compute_metrics=metrics,
                   data_collator=DataCollatorWithPadding(tok))
    tr_.train()
    res = tr_.evaluate()
    pred = tr_.predict(ddv)
    rep = classification_report(pred.label_ids, pred.predictions.argmax(-1), labels=list(range(len(labels))),
                                target_names=labels, zero_division=0, output_dict=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tr_.save_model(a.out)
    tok.save_pretrained(a.out)
    json.dump({"eval": res, "report": rep}, open(Path(a.out) / "metrics.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
