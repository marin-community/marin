# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-130m sweep for Ali Jadbabaie's gain-gated Muon ("idea 2").

The gain-gated update interpolates between Muon and the square-root / HJB policy
via a gain ``g`` (see ``levanter.optim.gain_gated``):

    U = P · diag(min(ρ, √(2 g σᵢ))) · Qᵀ ,   from the SVD of the (Frobenius-normalized) gradient.

``ρ`` is folded into the learning rate (ρ=1), so the swept knobs are the **gain g**
(the important one — it sets how Muon-like vs. square-root-like the optimizer is)
and the **learning rate**. We run both variants Ali described:

  * ``orig`` — apply U as derived (its Frobenius norm varies with g);
  * ``norm`` — rescale U to ‖U‖_F = √(min(d1,d2)) (Muon's update norm), so g changes
    only the update *direction*, decoupled from the learning rate.

Everything mirrors the HybridMuon / PRISM-Berkeley Qwen3 speedrun setting
(marin#4933): 130m = llama_150m, fineweb-edu-10B cache, paloma validation, region-
pinned to us-central1, W&B → marin-community/speedrun. The Muon endpoint (large g)
recovers the recorded Muon baseline as a reference.

The grid is env-overridable (comma-separated) so the sweep can run in waves:
  GAINS       gain values g                 (default 0.5,1,2,4,8,16)
  LR_MULTS    LR multipliers on 0.016       (default 0.5,1.0,2.0)
  VERSIONS    orig / norm                   (default orig,norm)

Submit (preemptible v5p-8, one coordinator):

    MARIN_PREFIX=gs://marin-us-central1 \\
    .venv/bin/iris --config lib/iris/config/marin.yaml job run --no-wait \\
      --preemptible --region us-central1 --extra tpu \\
      -e WANDB_API_KEY "$WANDB_API_KEY" -e MARIN_PREFIX "gs://marin-us-central1" \\
      -e GAINS -e LR_MULTS -e VERSIONS \\
      -- python -m experiments.speedrun.gain_gated_qwen3_scaling.sweep

``--extra tpu`` is REQUIRED (workers need jax+libtpu via IRIS_JOB_EXTRAS, else "No
accelerator found"). ``--region us-central1`` pins the coordinator to the data region
to avoid cross-region lock-bookkeeping egress.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.gain_gated import GainGatedConfig
from marin.execution.executor import ExecutorStep, executor_main

from experiments.defaults import default_train
from experiments.llama import llama_150m
from experiments.prebuilt_caches import fineweb_edu_subcache_10B
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)

# Region for the training children + their data. Both us-central1 and us-east5 mirror
# the fineweb-edu-10B-ac65f6 cache + paloma, so either is in-region (no cross-region I/O)
# — set REGION + MARIN_PREFIX + the coordinator's --region together to the same region.
# Splitting the grid across both regions multiplies the available preemptible v5p-8 spot
# capacity (the per-region spot pool grants only a few slices at a time).
REGION = os.environ.get("REGION", "us-central1")
SEQ_LEN = 4096
WANDB_ENTITY = "marin-community"
WANDB_PROJECT = "speedrun"


def _floats(env: str, default: str) -> tuple[float, ...]:
    return tuple(float(x) for x in os.environ.get(env, default).split(","))


def _versions(default: str = "orig,norm") -> tuple[str, ...]:
    vs = tuple(v.strip() for v in os.environ.get("VERSIONS", default).split(","))
    for v in vs:
        if v not in ("orig", "norm"):
            raise ValueError(f"VERSIONS entries must be 'orig' or 'norm', got {v!r}")
    return vs


# g grid spans the full Muon↔square-root interpolation. Calibrated on 130m-shaped
# matrices (Frobenius-normalized σ): g≤4 ≈ pure square-root, g≈8–64 is the interesting
# interior (fraction of singular values saturated rises 0→1), g≥256 ≈ pure Muon.
GAINS: tuple[float, ...] = _floats("GAINS", "1,4,8,16,32,64,256")
LR_MULTIPLIERS: tuple[float, ...] = _floats("LR_MULTS", "0.5,1.0,2.0")
VERSIONS: tuple[str, ...] = _versions()


@dataclass(frozen=True)
class SizeSpec:
    label: str
    llama_cfg: LlamaConfig
    muon_lr: float  # center matrix LR (Muon's 130m center)
    adam_lr: float  # center AdamW LR (embeddings / lm_head / biases)
    train_batch_size: int
    num_train_steps: int
    tpu_variant: str
    momentum: float
    adam_epsilon: float
    max_grad_norm: float
    decay: float


# 130m only (per the sweep directive). Centers/steps/resources copied verbatim from
# the HybridMuon / PRISM-Berkeley 130m row (marin#4933). TPU is env-overridable
# (TPU_VARIANT) so part of the grid can run on the abundant v6e-preemptible pool in
# us-east5-b (REGION=us-east5) while the scarce v5p spot handles the rest.
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


def _g_label(g: float) -> str:
    return f"g{g:g}".replace(".", "_")


def _lr_label(multiplier: float) -> str:
    return f"lrx{f'{multiplier:g}'.replace('.', '_')}"


def _override_tracker(step: ExecutorStep) -> ExecutorStep:
    pod = step.config
    inner = pod.train_config
    trainer = inner.trainer
    tracker = dataclasses.replace(trainer.tracker, entity=WANDB_ENTITY, project=WANDB_PROJECT)
    trainer = dataclasses.replace(trainer, tracker=tracker)
    inner = dataclasses.replace(inner, trainer=trainer)
    pod = dataclasses.replace(pod, train_config=inner)
    return dataclasses.replace(step, config=pod)


def _build_step(version: str, gain: float, multiplier: float) -> ExecutorStep:
    matrix_lr = SIZE.muon_lr * multiplier
    adam_lr = SIZE.adam_lr * multiplier
    normalize_fro = version == "norm"

    optimizer = GainGatedConfig(
        gain=gain,
        rho=1.0,  # folded into the learning rate
        normalize_fro=normalize_fro,
        lr=matrix_lr,
        learning_rate=matrix_lr,  # lr_scheduler reads `learning_rate`
        adam_lr=adam_lr,
        momentum=SIZE.momentum,
        nesterov=True,
        weight_decay=0.1,
        beta1=0.8,
        beta2=0.98,
        epsilon=SIZE.adam_epsilon,
        svd_epsilon=1e-7,
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

    run_id = f"qwen3_{SIZE.label}_gain_gated_{version}_{_g_label(gain)}_{SEQ_LEN}_{_lr_label(multiplier)}"
    step = default_train(
        name=run_id,
        tokenized=fineweb_edu_subcache_10B,
        model_config=_to_qwen3_from_llama(SIZE.llama_cfg),
        train_config=train,
        tags=["speedrun", "gain_gated", f"version_{version}", _g_label(gain), f"qwen3_{SIZE.label}"],
        use_default_validation=True,
        eval_harness_tasks=(),
        wandb_name=run_id,
        wandb_group=f"gain_gated_{version}_qwen3_130m",
    )
    return _override_tracker(step)


def main() -> None:
    if os.getenv("CI") is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    steps = [
        _build_step(version, gain, multiplier)
        for version in VERSIONS
        for gain in GAINS
        for multiplier in LR_MULTIPLIERS
    ]
    logger.info("Gain-gated 130m sweep: %d runs (versions=%s gains=%s lr_mults=%s)", len(steps), VERSIONS, GAINS, LR_MULTIPLIERS)
    executor_main(
        steps=steps,
        description="Qwen3-130m gain-gated Muon sweep (Ali idea 2): gain g x LR x {orig,norm}.",
    )


if __name__ == "__main__":
    main()
