"""OPTIONAL: distil Stage-1 frame extraction from Qwen2.5-7B into Qwen2.5-3B (or 1.5B) with QLoRA.

Why: frame extraction is the most frequent LLM call (every summary sentence). A 3B student
in 4-bit runs ~2.5x faster than the 7B teacher on an RTX 5060 and frees VRAM so the NLI model
can stay resident. This is an efficiency option (report it in RQ6), not required for the method.

Memory on 8 GB: 3B in 4-bit NF4 ~2.1 GB + LoRA r=16 + activations at len 1024, batch 2,
grad-accum 8, gradient checkpointing -> ~6.5 GB peak. (Windows: bitsandbytes >= 0.43 has native wheels.)

  1) python -m train.qlora_frames --make_data --summaries data/meetverify/summaries_train.jsonl   # teacher labels via cache
  2) python -m train.qlora_frames --train
  3) merge + export to GGUF with llama.cpp's convert_hf_to_gguf.py, then quantise Q4_K_M,
     and point configs/default.yaml llm.gguf_path at it (or keep the 7B for the judge).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cprove.utils import load_config


def make_data(cfg, summaries, out):
    from cprove.llm import make_llm
    from cprove.data import load_corpus
    from cprove.prompts import FRAME_SCHEMA, frame_messages
    from cprove.text import split_sentences
    llm = make_llm(cfg["llm"])
    assert llm.available, "teacher LLM (Qwen2.5-7B GGUF) required"
    meetings = {m.meeting_id: m for m in load_corpus(cfg)[0]}
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for line in open(summaries, encoding="utf-8"):
            s = json.loads(line)
            m = meetings[s["meeting_id"]]
            parts = m.meta.get("participants") or m.speakers
            sents = split_sentences(s["summary"])
            for i, sent in enumerate(sents):
                sys_, usr = frame_messages(sent, parts, sents[:i])
                res = llm.json(sys_, usr, FRAME_SCHEMA)
                if res and res.get("claims"):
                    f.write(json.dumps({"messages": [{"role": "system", "content": sys_}, {"role": "user", "content": usr},
                                                     {"role": "assistant", "content": json.dumps(res)}]}) + "\n")
                    n += 1
    print(n, "examples ->", out)


def train(cfg, data, out):
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    q = cfg["train"]["qlora_frames"]
    tok = AutoTokenizer.from_pretrained(q["base"])
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(q["base"], quantization_config=bnb, device_map="auto",
                                                 torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=q["r"], lora_alpha=q["alpha"], lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    ds = load_dataset("json", data_files=data)["train"].train_test_split(test_size=0.05, seed=42)
    args = SFTConfig(output_dir=out, per_device_train_batch_size=q["batch_size"], gradient_accumulation_steps=q["grad_accum"],
                     num_train_epochs=q["epochs"], learning_rate=q["lr"], lr_scheduler_type="cosine", warmup_ratio=0.03,
                     bf16=True, logging_steps=20, save_strategy="epoch", eval_strategy="epoch", report_to=[],
                     max_length=q["max_length"], assistant_only_loss=True, gradient_checkpointing=True,
                     dataloader_num_workers=0)
    t = SFTTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["test"], processing_class=tok)
    t.train()
    t.save_model(out)
    # merged fp16 copy for GGUF conversion
    from peft import AutoPeftModelForCausalLM
    merged = AutoPeftModelForCausalLM.from_pretrained(out, torch_dtype=torch.float16, device_map="cpu").merge_and_unload()
    merged.save_pretrained(out + "_merged")
    tok.save_pretrained(out + "_merged")
    print("merged ->", out + "_merged  (convert with llama.cpp/convert_hf_to_gguf.py, then llama-quantize Q4_K_M)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_data", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--summaries", default="data/meetverify/summaries_train.jsonl")
    ap.add_argument("--data", default="data/silver/frames_sft.jsonl")
    ap.add_argument("--out", default="models/qwen3b_frames_lora")
    a = ap.parse_args()
    cfg = load_config()
    Path(a.data).parent.mkdir(parents=True, exist_ok=True)
    if a.make_data:
        make_data(cfg, a.summaries, a.data)
    if a.train:
        train(cfg, a.data, a.out)


if __name__ == "__main__":
    main()
