# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-130m sweep for Ali Jadbabaie's left-preconditioned Muon ("idea 4").

    U = ρ · H^{-1/2} · polar( H^{-1/2} · M ) ,   H = EMA(G Gᵀ)

Left/output-side whitening by the gradient's own second moment H (Shampoo left factor),
with a truncated pseudo-inverse for H^{-1/2} (facebookresearch/optimizers#265). Pure
optimizer — gradient-only, no activation capture (cf. idea 3). H = I recovers plain Muon.

Swept knobs: learning rate (primary) × clamp_rel (truncated-pseudo-inverse threshold).
Everything else mirrors the gain-gated / activation-aware Qwen3 speedrun (marin#4933):
130m = llama_150m, fineweb-edu-10B, paloma, W&B marin-community/speedrun. Compare on
**eval/paloma/c4_en/bpb** (Muon baseline 1.1663; a same-setup Muon control = 1.1673).

Submit (v6e-8 preemptible, us-east5):

    MARIN_PREFIX=gs://marin-us-east5 \\
    .venv/bin/iris --config lib/iris/config/marin.yaml job run --no-wait \\
      --preemptible --region us-east5 --extra tpu \\
      -e WANDB_API_KEY "$WANDB_API_KEY" -e MARIN_PREFIX "gs://marin-us-east5" \\
      -e REGION us-east5 -e TPU_VARIANT v6e-8 -e LR_MULTS -e CLAMPS -e RUN_TAG \\
      -- python -m experiments.speedrun.left_precond_muon_qwen3.sweep
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.left_precond_muon import LeftPrecondMuonConfig
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


def _floats(env: str, default: str) -> tuple[float, ...]:
    return tuple(float(x) for x in os.environ.get(env, default).split(","))


# LR is the primary tuning axis (per the goal). clamp_rel is fixed near Shampoo's 1e-15 (only
# numerically-zero directions truncated); 1e-8 included once to confirm insensitivity to mild truncation.
LR_MULTIPLIERS: tuple[float, ...] = _floats("LR_MULTS", "0.5,1.0,2.0,4.0")
CLAMPS: tuple[float, ...] = _floats("CLAMPS", "1e-15,1e-8")

# Variant ablations (comma-separated). Each maps to optimizer flags:
#   full       — U = H^{-1/2} polar(H^{-1/2} M)            (the base idea 4)
#   inner_only — U = polar(H^{-1/2} M)                     (v1: drop the outer H^{-1/2})
#   real_inv   — damped real inverse instead of truncated  (v2)
#   fro_norm   — rescale U to Muon's Frobenius norm        (v3)
VARIANT_FLAGS = {
    "full": dict(outer_precond=True, real_inverse=False, normalize_fro=False),
    "inner_only": dict(outer_precond=False, real_inverse=False, normalize_fro=False),
    "real_inv": dict(outer_precond=True, real_inverse=True, normalize_fro=False),
    "fro_norm": dict(outer_precond=True, real_inverse=False, normalize_fro=True),
}
VARIANTS: tuple[str, ...] = tuple(v.strip() for v in os.environ.get("VARIANTS", "full").split(","))

# Damping λ on mean-normalized H eigenvalues: (w/mean + λ)^{-1/2}. λ=0 = un-damped (over-amplifies
# tiny eigenvalues); larger λ bounds it, λ→∞ ⟹ Muon. Swept (esp. for inner_only) to test whether a
# damped inner whitening beats Muon.
DAMPINGS: tuple[float, ...] = _floats("DAMPINGS", "0.0")


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


_TPU_VARIANT = os.environ.get("TPU_VARIANT", "v6e-8")
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


def _lr_label(m: float) -> str:
    return f"lrx{f'{m:g}'.replace('.', '_')}"


def _clamp_label(c: float) -> str:
    return f"c{f'{c:g}'.replace('.', '_').replace('-', 'm').replace('+', 'p')}"


def _override_tracker(step: ExecutorStep) -> ExecutorStep:
    pod = step.config
    inner = pod.train_config
    trainer = dataclasses.replace(
        inner.trainer, tracker=dataclasses.replace(inner.trainer.tracker, entity=WANDB_ENTITY, project=WANDB_PROJECT)
    )
    inner = dataclasses.replace(inner, trainer=trainer)
    return dataclasses.replace(step, config=dataclasses.replace(pod, train_config=inner))


RUN_TAG: str = os.environ.get("RUN_TAG", "")


def _build_step(multiplier: float, clamp_rel: float, variant: str, damping: float = 0.0) -> ExecutorStep:
    matrix_lr = SIZE.muon_lr * multiplier
    adam_lr = SIZE.adam_lr * multiplier
    optimizer = LeftPrecondMuonConfig(
        h_beta=0.95,
        clamp_rel=clamp_rel,
        ns_steps=5,
        damping=damping,
        **VARIANT_FLAGS[variant],
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
    vlabel = "" if variant == "full" else f"_{variant}"
    dlabel = "" if damping == 0.0 else f"_dmp{f'{damping:g}'.replace('.', '_')}"
    run_id = f"qwen3_{SIZE.label}_leftprec{vlabel}{dlabel}{tag}_{_clamp_label(clamp_rel)}_{SEQ_LEN}_{_lr_label(multiplier)}"
    step = default_train(
        name=run_id,
        tokenized=fineweb_edu_subcache_10B,
        model_config=_to_qwen3_from_llama(SIZE.llama_cfg),
        train_config=train,
        tags=["speedrun", "left_precond_muon", variant, _clamp_label(clamp_rel), f"qwen3_{SIZE.label}"],
        use_default_validation=True,
        eval_harness_tasks=(),
        wandb_name=run_id,
        wandb_group="left_precond_muon_qwen3_130m",
    )
    return _override_tracker(step)


def main() -> None:
    if os.getenv("CI") is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return
    steps = [_build_step(m, c, v, d) for v in VARIANTS for m in LR_MULTIPLIERS for c in CLAMPS for d in DAMPINGS]
    logger.info(
        "Left-precond Muon 130m sweep: %d runs (variants=%s lr=%s clamps=%s dampings=%s)",
        len(steps),
        VARIANTS,
        LR_MULTIPLIERS,
        CLAMPS,
        DAMPINGS,
    )
    executor_main(
        steps=steps, description="Qwen3-130m left-preconditioned Muon sweep (Ali idea 4): variant x LR x clamp."
    )


if __name__ == "__main__":
    main()
