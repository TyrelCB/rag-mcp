#!/usr/bin/env python3
"""LoRA SFT on exported session data. Run inside the NGC pytorch container
(see run_container.sh) — native aarch64 wheels for TRL/PEFT are installed there.

Smoke test:
  python train_lora.py --base Qwen/Qwen3-0.6B --data /path/train_chat.jsonl --max-steps 2
Real run:
  python train_lora.py --base Qwen/Qwen3.5-9B --data data/sft-YYYYMMDD/train_tools.jsonl \
      --val data/sft-YYYYMMDD/val_tools.jsonl --epochs 2 --run-name qwen95-sessions-v1
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.5-9B", help="HF base model id")
    ap.add_argument("--data", required=True, help="train JSONL ({'messages': [...]})")
    ap.add_argument("--val", default=None, help="optional val JSONL")
    ap.add_argument("--run-name", default="rag-sft")
    ap.add_argument("--out", default=None, help="output dir (default training/runs/<run-name>)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--target-preset", choices=["all", "attn"], default="all",
                    help="'all' = all-linear (dense models). 'attn' = attention/"
                         "linear-attn projections only — use for MoE bases like "
                         "Qwen3.6-35B-A3B (avoids per-expert LoRA modules) and for "
                         "faster runs on dense models.")
    ap.add_argument("--no-packing", action="store_true",
                    help="Disable sequence packing even when flash-attn is available")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    out_dir = Path(args.out or Path(__file__).parent / "runs" / args.run_name)

    data_files = {"train": args.data}
    if args.val:
        data_files["validation"] = args.val
    ds = load_dataset("json", data_files=data_files)

    # flash-attn enables packing (sdpa cross-contaminates packed samples)
    try:
        import flash_attn  # noqa: F401
        attn_impl, packing = "flash_attention_2", True
    except ImportError:
        attn_impl, packing = "sdpa", False
    if args.no_packing:
        packing = False
    print(f"attn={attn_impl} packing={packing} targets={args.target_preset}")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,   # plain bf16 LoRA; no bitsandbytes on aarch64
        device_map="auto",
        attn_implementation=attn_impl,
    )

    if args.target_preset == "attn":
        # full-attention + linear-attention (gated delta net) projections;
        # deliberately skips MoE expert MLPs
        target_modules = r".*(self_attn|linear_attn)\..*proj[a-z_]*"
    else:
        target_modules = "all-linear"

    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        run_name=args.run_name,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="epoch" if args.max_steps < 0 else "no",
        eval_strategy="epoch" if args.val else "no",
        max_length=args.max_seq_len,
        packing=packing,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation"),
        peft_config=peft_config,
        args=sft_config,
    )
    trainer.train()
    trainer.save_model(str(out_dir / "adapter"))
    tokenizer.save_pretrained(str(out_dir / "adapter"))
    print(f"adapter saved to {out_dir / 'adapter'}")


if __name__ == "__main__":
    main()
