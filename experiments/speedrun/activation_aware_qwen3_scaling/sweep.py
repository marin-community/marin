# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-130m sweep for Ali Jadbabaie's activation-aware Muon ("idea 3").

The activation-aware update whitens the Muon step by the layer's *input activation*
second moment Σ = A Aᵀ (see ``levanter.optim.activation_aware``):

    D = sign( M · Σ^{-1/2} ) · Σ^{-1/2}        (sign = matrix sign = NS5 polar factor)

Σ is captured from the forward pass (``levanter.optim.activation_capture``): the two
block-level input Grams — ``input_layernorm`` output (→ attention q/k/v) and
``post_attention_layernorm`` output (→ MLP gate/up). It cannot be recovered from the
gradient (the gradient G=δAᵀ conflates δ and A; a gradient-derived Σ is also degenerate
because M·(MᵀM)^{-1/2} is already the polar factor). o_proj/down_proj get plain Muon;
embeddings/lm_head/norms/biases get AdamW. Σ=I recovers plain Muon.

The swept knobs are the **learning rate** (Kaiyue's earlier torch attempt diverged → LR
matters) and the **damping** λ in (Σ + λ·mean(eig)·I)^{-1/2} (large λ → closer to plain
Muon, the anchor). Everything else mirrors the gain-gated / HybridMuon Qwen3 speedrun
(marin#4933): 130m = llama_150m, fineweb-edu-10B cache, paloma validation, W&B →
marin-community/speedrun. The Muon baseline (1.1663) is the reference.

Submit (preemptible v5p-8, one coordinator):

    MARIN_PREFIX=gs://marin-us-central1 \\
    .venv/bin/iris --config lib/iris/config/marin.yaml job run --no-wait \\
      --preemptible --region us-central1 --extra tpu \\
      -e WANDB_API_KEY "$WANDB_API_KEY" -e MARIN_PREFIX "gs://marin-us-central1" \\
      -e LR_MULTS -e DAMPINGS -e RUN_TAG \\
      -- python -m experiments.speedrun.activation_aware_qwen3_scaling.sweep

``--extra tpu`` is REQUIRED (workers need jax+libtpu). ``--region`` pins the coordinator
to the data region (no cross-region I/O).
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.activation_aware import ActivationAwareConfig
from marin.execution.executor import ExecutorStep, executor_main

from experiments.defaults import default_train
from experiments.llama import llama_150m
from experiments.prebuilt_caches import fineweb_edu_subcache_10B
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)

# Both us-central1 and us-east5 mirror the fineweb-edu-10B-ac65f6 cache + paloma; set
# REGION + MARIN_PREFIX + the coordinator's --region together to the same region.
REGION = os.environ.get("REGION", "us-central1")
SEQ_LEN = 4096
WANDB_ENTITY = "marin-community"
WANDB_PROJECT = "speedrun"


def _floats(env: str, default: str) -> tuple[float, ...]:
    return tuple(float(x) for x in os.environ.get(env, default).split(","))


# LR is primary (the earlier attempt diverged); damping λ trades off whitening strength
# vs. the plain-Muon anchor (large λ → ≈ Muon).
LR_MULTIPLIERS: tuple[float, ...] = _floats("LR_MULTS", "0.5,1.0,2.0")
DAMPINGS: tuple[float, ...] = _floats("DAMPINGS", "0.001,0.01,0.1")


@dataclass(frozen=True)
class SizeSpec:
    label: str
    llama_cfg: LlamaConfig
    muon_lr: float
    adam_lr: float
    train_batch_size: int
    num_train_steps: int
    tpu_variant: str
    momentum: float
    adam_epsilon: float
    max_grad_norm: float
    decay: float


# 130m only. Centers/steps/resources copied from the HybridMuon / gain-gated 130m row.
_TPU_VARIANT = os.environ.get("TPU_VARIANT", "v5p-8")
SIZE = SizeSpec("130m", llama_150m, 0.016, 0.0032, 128, 4959, _TPU_VARIANT, 0.95, 1e-15, 1.0, 0.8)


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


def _lr_label(multiplier: float) -> str:
    return f"lrx{f'{multiplier:g}'.replace('.', '_')}"


def _damp_label(damping: float) -> str:
    return f"d{f'{damping:g}'.replace('.', '_').replace('-', 'm')}"


def _override_train(step: ExecutorStep) -> ExecutorStep:
    """Point W&B at marin-community/speedrun AND enable activation-Gram capture."""
    pod = step.config
    inner = pod.train_config
    trainer = inner.trainer
    tracker = dataclasses.replace(trainer.tracker, entity=WANDB_ENTITY, project=WANDB_PROJECT)
    trainer = dataclasses.replace(trainer, tracker=tracker, compute_activation_grams=True)
    inner = dataclasses.replace(inner, trainer=trainer)
    pod = dataclasses.replace(pod, train_config=inner)
    return dataclasses.replace(step, config=pod)


# Optional run-name suffix to force FRESH run_ids (avoid resuming a stale checkpoint).
RUN_TAG: str = os.environ.get("RUN_TAG", "")


def _build_step(multiplier: float, damping: float) -> ExecutorStep:
    matrix_lr = SIZE.muon_lr * multiplier
    adam_lr = SIZE.adam_lr * multiplier

    optimizer = ActivationAwareConfig(
        damping=damping,
        sigma_beta=0.95,
        normalize_fro=True,
        ns_steps=5,
        lr=matrix_lr,
        learning_rate=matrix_lr,
        adam_lr=adam_lr,
        momentum=SIZE.momentum,
        nesterov=True,
        weight_decay=0.1,
        beta1=0.8,
        beta2=0.98,
        epsilon=SIZE.adam_epsilon,
        matrix_epsilon=1e-7,
        max_grad_norm=SIZE.max_grad_norm,
        use_kimi_scaling=False,
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
        learning_rate=matrix_lr,
        optimizer_config=optimizer,
    )

    tag = f"_{RUN_TAG}" if RUN_TAG else ""
    run_id = f"qwen3_{SIZE.label}_act_aware_{_damp_label(damping)}{tag}_{SEQ_LEN}_{_lr_label(multiplier)}"
    step = default_train(
        name=run_id,
        tokenized=fineweb_edu_subcache_10B,
        model_config=_to_qwen3_from_llama(SIZE.llama_cfg),
        train_config=train,
        tags=["speedrun", "activation_aware", _damp_label(damping), f"qwen3_{SIZE.label}"],
        use_default_validation=True,
        eval_harness_tasks=(),
        wandb_name=run_id,
        wandb_group="activation_aware_qwen3_130m",
    )
    return _override_train(step)


def main() -> None:
    if os.getenv("CI") is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    steps = [_build_step(multiplier, damping) for multiplier in LR_MULTIPLIERS for damping in DAMPINGS]
    logger.info(
        "Activation-aware 130m sweep: %d runs (lr_mults=%s dampings=%s)",
        len(steps),
        LR_MULTIPLIERS,
        DAMPINGS,
    )
    executor_main(
        steps=steps,
        description="Qwen3-130m activation-aware Muon sweep (Ali idea 3): LR x damping.",
    )


if __name__ == "__main__":
    main()
