# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .config import DataConfig, ModelConfig, TrainConfig
from .data import build_dataset
from .diffusion import BlockDiffusionObjective
from .model import BlockDiffusionDenoiser


def parse_args() -> tuple[ModelConfig, DataConfig, TrainConfig]:
    parser = argparse.ArgumentParser(description="Train attention or Bi-GDN block diffusion models in PyTorch.")
    parser.add_argument("--variant", choices=("baseline", "bigdn"), default="baseline")
    parser.add_argument("--dataset", choices=("toy", "text", "hf", "fineweb", "fineweb_edu"), default="toy")
    parser.add_argument("--tokenizer", default="auto")
    parser.add_argument("--text-file")
    parser.add_argument("--hf-dataset")
    parser.add_argument("--hf-dataset-config")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-text-field", default="text")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--toy-size", type=int, default=4096)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/block_diffusion_cuda"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ddp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--vocab-size", type=int, default=50_257)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--window-blocks", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--attention-period", type=int, default=4)
    parser.add_argument("--gdn-heads", type=int, default=12)
    args = parser.parse_args()

    model = ModelConfig(
        vocab_size=args.vocab_size,
        mask_token_id=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        block_size=args.block_size,
        window_blocks=args.window_blocks,
        diffusion_steps=args.diffusion_steps,
        variant=args.variant,
        attention_period=args.attention_period,
        gdn_heads=args.gdn_heads,
    )
    data = DataConfig(
        dataset=args.dataset,
        tokenizer=args.tokenizer,
        text_file=args.text_file,
        hf_dataset=args.hf_dataset,
        hf_dataset_config=args.hf_dataset_config,
        hf_split=args.hf_split,
        hf_text_field=args.hf_text_field,
        max_examples=args.max_examples,
        streaming=args.streaming,
        shuffle_buffer=args.shuffle_buffer,
        toy_size=args.toy_size,
        seed=args.seed,
    )
    train = TrainConfig(
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        log_every=args.log_every,
        save_every=args.save_every,
        sample_every=args.sample_every,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=args.device,
        compile=args.compile,
        bf16=args.bf16,
        ddp=args.ddp,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    return model, data, train


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def ddp_state(enabled: bool) -> tuple[bool, int, int, int]:
    if not enabled or "RANK" not in os.environ:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def cleanup_ddp(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step: int, *, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * cosine


def maybe_init_wandb(train_config: TrainConfig, *, rank: int, config_payload: dict[str, object]):
    if rank != 0 or train_config.wandb_project is None:
        return None
    try:
        import wandb
    except ImportError:
        return None
    return wandb.init(
        project=train_config.wandb_project,
        name=train_config.wandb_run_name,
        config=config_payload,
    )


def save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    train_config: TrainConfig,
    model_config: ModelConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "train_config": train_config,
            "model_config": model_config,
        },
        path,
    )


def decode_sample(tokens: torch.Tensor, tokenizer: object | None) -> str:
    if tokenizer is None:
        return " ".join(map(str, tokens.tolist()))
    return tokenizer.decode(tokens.tolist())


def main() -> None:
    model_config, data_config, train_config = parse_args()
    distributed, rank, world_size, local_rank = ddp_state(train_config.ddp)
    seed_everything(train_config.seed + rank)

    device = resolve_device(train_config.device)
    if distributed:
        device = torch.device(f"cuda:{local_rank}")

    out_dir = train_config.out_dir / model_config.variant
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "config.json", "w", encoding="utf-8") as fp:
            json.dump(
                {
                    "model": model_config.__dict__,
                    "data": data_config.__dict__,
                    "train": {k: (str(v) if isinstance(v, Path) else v) for k, v in train_config.__dict__.items()},
                },
                fp,
                indent=2,
            )

    loaded = build_dataset(data_config, model_config, rank=rank, world_size=world_size)
    sampler = None
    if distributed and not loaded.is_streaming:
        sampler = DistributedSampler(loaded.dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        loaded.dataset,
        batch_size=train_config.batch_size,
        sampler=sampler,
        shuffle=sampler is None and not loaded.is_streaming,
        num_workers=train_config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = BlockDiffusionDenoiser(model_config).to(device)
    if train_config.compile and device.type == "cuda":
        model = torch.compile(model)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    objective = BlockDiffusionObjective(model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        betas=train_config.betas,
        weight_decay=train_config.weight_decay,
    )
    run = maybe_init_wandb(
        train_config,
        rank=rank,
        config_payload={
            "model": model_config.__dict__,
            "data": data_config.__dict__,
            "train": {k: (str(v) if isinstance(v, Path) else v) for k, v in train_config.__dict__.items()},
        },
    )

    autocast_enabled = device.type == "cuda" and train_config.bf16
    autocast_dtype = torch.bfloat16 if autocast_enabled else torch.float32
    data_iter = iter(dataloader)
    started = time.time()

    try:
        for step in range(train_config.steps):
            model.train()
            if sampler is not None and step % max(1, len(dataloader)) == 0:
                sampler.set_epoch(step)
            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            accuracy_accum = 0.0
            masked_accum = 0.0
            for _ in range(train_config.grad_accum_steps):
                try:
                    batch_tokens = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch_tokens = next(data_iter)
                batch_tokens = batch_tokens.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
                    stats = objective.loss(model, batch_tokens)
                    loss = stats["loss"] / train_config.grad_accum_steps
                loss.backward()
                loss_accum += float(stats["loss"].detach())
                accuracy_accum += float(stats["masked_accuracy"].detach())
                masked_accum += float(stats["num_masked"].detach())

            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            lr = cosine_lr(
                step,
                base_lr=train_config.lr,
                warmup_steps=train_config.warmup_steps,
                total_steps=train_config.steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()

            if rank == 0 and (step % train_config.log_every == 0 or step == train_config.steps - 1):
                elapsed = time.time() - started
                tokens_per_step = train_config.batch_size * world_size * model_config.max_seq_len
                tokens_per_second = tokens_per_step * (step + 1) / max(elapsed, 1e-6)
                log = {
                    "step": step,
                    "loss": loss_accum,
                    "masked_accuracy": accuracy_accum / train_config.grad_accum_steps,
                    "masked_tokens": masked_accum / train_config.grad_accum_steps,
                    "lr": lr,
                    "tokens_per_second": tokens_per_second,
                }
                print(json.dumps(log), flush=True)
                if run is not None:
                    run.log(log, step=step)

            if rank == 0 and train_config.save_every and (step + 1) % train_config.save_every == 0:
                save_checkpoint(
                    path=out_dir / f"step-{step + 1}.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step + 1,
                    train_config=train_config,
                    model_config=model_config,
                )

            if rank == 0 and train_config.sample_every and (step + 1) % train_config.sample_every == 0:
                raw_model = model.module if isinstance(model, DDP) else model
                raw_model.eval()
                sampled = objective.sample_sequence(raw_model, num_blocks=2)
                sample_text = decode_sample(sampled[0], loaded.tokenizer)
                print(f"sample step {step + 1}: {sample_text[:400]}", flush=True)
                if run is not None:
                    run.log({"sample_text": sample_text}, step=step)
    finally:
        if rank == 0 and run is not None:
            run.finish()
        cleanup_ddp(distributed)


if __name__ == "__main__":
    main()
