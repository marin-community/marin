# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full SFT with FP32 master parameters, BF16 autocast and 8-bit Adam."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import transformers
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--domain", choices=["math", "code"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.output.exists() and args.resume is None:
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required; use gpu-borrow")
    tokenizer = AutoTokenizer.from_pretrained(args.parent, local_files_only=True, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(
        args.parent, torch_dtype=torch.float32, attn_implementation="sdpa", local_files_only=True
    )
    model.config.use_cache = False
    training_args = TrainingArguments(
        output_dir=str(args.output),
        max_steps=args.steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=True,
        liger_kernel_config={
            "rope": False,
            "rms_norm": False,
            "swiglu": False,
            "cross_entropy": False,
            "fused_linear_cross_entropy": True,
        },
        prediction_loss_only=True,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to="wandb",
        run_name=f"task290-{args.domain}-full-sft",
        seed=290,
        data_seed=290,
        dataloader_num_workers=0,
    )
    metadata = {
        "domain": args.domain,
        "parent_path": str(args.parent),
        "parameter_dtype": "float32",
        "autocast_dtype": "bfloat16",
        "saved_dtype": "float32",
        "training_args": training_args.to_dict(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "data_sha256": hashlib.sha256((args.data / "metadata.json").read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "training-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "train-source.py").write_bytes(Path(__file__).read_bytes())
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=load_from_disk(str(args.data / args.domain / "train")),
        eval_dataset=load_from_disk(str(args.data / args.domain / "validation")),
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    model.config.use_cache = True
    final = args.output / "final-fp32"
    trainer.save_model(str(final))
    metadata["best_checkpoint"] = trainer.state.best_model_checkpoint
    metadata["best_validation_loss"] = trainer.state.best_metric
    metadata["completed_steps"] = trainer.state.global_step
    (args.output / "training-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
