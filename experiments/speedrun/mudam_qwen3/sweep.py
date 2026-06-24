# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-130m reproduction sweep for Mudam (Shampoo-Muon), to compare against Muon.

Mudam (`levanter.optim.mudam`) is a one-sided SOAP / matrix-Adam: precondition the momentum by
the inverse-sqrt of the gradient second moment H=EMA(GGᵀ or GᵀG) via a coupled Newton-Schulz
(no explicit eigh). Defaults: input-side preconditioner, kimi update scaling, no extra Muon
re-orthogonalization (another_muon=None). Run as-is and sweep LR to see if it matches/beats the
Muon baseline (eval/paloma/c4_en/bpb = 1.1663; same-setup Muon control = 1.1673).

Because kimi scaling (scale = √max(d_out,d_in)) makes the update ~√d larger than Muon's, the
optimal LR is well below Muon's 0.016 — so LR is swept wide and low.

    MARIN_PREFIX=gs://marin-us-east5 .venv/bin/iris --config lib/iris/config/marin.yaml job run \\
      --no-wait --preemptible --region us-east5 --extra tpu \\
      -e WANDB_API_KEY "$WANDB_API_KEY" -e MARIN_PREFIX gs://marin-us-east5 \\
      -e REGION us-east5 -e TPU_VARIANT v6e-8 -e LRS -e RUN_TAG \\
      -- python -m experiments.speedrun.mudam_qwen3.sweep
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.mudam import MudamConfig
from marin.execution.executor import ExecutorStep, executor_main

from experiments.defaults import default_train
from experiments.llama import llama_150m
from experiments.prebuilt_caches import fineweb_edu_subcache_10B
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)

REGION = os.environ.get("REGION", "us-east5")
SEQ_LEN = 4096
WANDB_ENTITY = "marin-community"
WANDB_PROJECT = "speedrun"

# Absolute matrix LRs (kimi scaling → optimum well below Muon's 0.016).
LRS: tuple[float, ...] = tuple(float(x) for x in os.environ.get("LRS", "0.0002,0.0005,0.001,0.003,0.01").split(","))
USE_SCALING: str = os.environ.get("USE_SCALING", "kimi")
RUN_TAG: str = os.environ.get("RUN_TAG", "")


@dataclass(frozen=True)
class SizeSpec:
    label: str
    llama_cfg: LlamaConfig
    adam_lr: float
    train_batch_size: int
    num_train_steps: int
    tpu_variant: str
    decay: float


_TPU_VARIANT = os.environ.get("TPU_VARIANT", "v6e-8")
SIZE = SizeSpec("130m", llama_150m, 0.0032, 128, 4959, _TPU_VARIANT, 0.8)


def _to_qwen3_from_llama(llama_cfg: LlamaConfig) -> Qwen3Config:
    return Qwen3Config(
        max_seq_len=SEQ_LEN,
        hidden_dim=llama_cfg.hidden_dim,
        intermediate_dim=llama_cfg.intermediate_dim,
        num_layers=llama_cfg.num_layers,
        num_heads=llama_cfg.num_heads,
        num_kv_heads=llama_cfg.num_kv_heads,
        head_dim=getattr(llama_cfg, "head_dim", None),
        use_bias=getattr(llama_cfg, "use_bias", False),
        rope=llama_cfg.rope,
        activation_function=llama_cfg.activation_function,
        initializer_range=llama_cfg.initializer_range,
        layer_norm_epsilon=llama_cfg.layer_norm_epsilon,
        tie_word_embeddings=llama_cfg.tie_word_embeddings,
        upcast_attn=llama_cfg.upcast_attn,
        attn_backend=llama_cfg.attn_backend,
        flash_attention_block_size=llama_cfg.flash_attention_block_size,
        hybrid_norm=False,
        scan_layers=True,
        gradient_checkpointing=True,
    )


def _lr_label(lr: float) -> str:
    return f"lr{f'{lr:g}'.replace('.', '_').replace('-', 'm')}"


def _override_tracker(step: ExecutorStep) -> ExecutorStep:
    pod = step.config
    inner = pod.train_config
    trainer = dataclasses.replace(
        inner.trainer, tracker=dataclasses.replace(inner.trainer.tracker, entity=WANDB_ENTITY, project=WANDB_PROJECT)
    )
    inner = dataclasses.replace(inner, trainer=trainer)
    return dataclasses.replace(step, config=dataclasses.replace(pod, train_config=inner))


def _build_step(lr: float) -> ExecutorStep:
    optimizer = MudamConfig(
        momentum=0.95,
        shampoo_beta=0.95,
        beta1=0.95,
        beta2=0.95,
        epsilon=1e-15,
        weight_decay=0.1,
        max_grad_norm=1.0,
        adam_lr=SIZE.adam_lr,
        use_scaling=USE_SCALING,
        prefer_input_side=True,
        normalization="muon",
        learning_rate=lr,
        lr_schedule="linear",
        warmup=0,
        rewarmup=0,
        decay=SIZE.decay,
        min_lr_ratio=0,
    )
    train = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(SIZE.tpu_variant, regions=[REGION], preemptible=True),
        train_seq_len=SEQ_LEN,
        train_batch_size=SIZE.train_batch_size,
        num_train_steps=SIZE.num_train_steps,
        learning_rate=lr,
        optimizer_config=optimizer,
    )
    tag = f"_{RUN_TAG}" if RUN_TAG else ""
    run_id = f"qwen3_{SIZE.label}_mudam{tag}_{USE_SCALING}_{SEQ_LEN}_{_lr_label(lr)}"
    step = default_train(
        name=run_id,
        tokenized=fineweb_edu_subcache_10B,
        model_config=_to_qwen3_from_llama(SIZE.llama_cfg),
        train_config=train,
        tags=["speedrun", "mudam", USE_SCALING, f"qwen3_{SIZE.label}"],
        use_default_validation=True,
        eval_harness_tasks=(),
        wandb_name=run_id,
        wandb_group="mudam_qwen3_130m",
    )
    return _override_tracker(step)


def main() -> None:
    if os.getenv("CI") is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return
    steps = [_build_step(lr) for lr in LRS]
    logger.info("Mudam 130m repro: %d runs (scaling=%s lrs=%s)", len(steps), USE_SCALING, LRS)
    executor_main(steps=steps, description="Qwen3-130m Mudam (Shampoo-Muon) reproduction vs Muon: LR sweep.")


if __name__ == "__main__":
    main()
