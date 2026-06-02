# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3 scaling LR sweep for the HybridMuon optimizer (variants "a" and "f").

Reproduces the experiment setting of the PRISM-Berkeley Qwen3 speedrun sweep
(marin-community/marin#4933) with the JAX ``HybridMuonConfig`` in place of the
PRISM-Berkeley optimizer. For each model size we sweep the per-size center
learning rate by multipliers {0.5, 0.75, 1.0, 1.25, 1.5}, scaling both the
HybridMuon (matrix) LR and the AdamW (embedding/lm_head/bias) LR together, for
each of the two HybridMuon variants.

Everything is region-pinned to ``us-central1`` (training cache + paloma validation +
checkpoints all live in ``gs://marin-us-central1``) so there is no cross-region I/O.
(The same ``fineweb-edu-10B-ac65f6`` cache + paloma also exist in ``gs://marin-us-east5``;
flip ``REGION``/``MARIN_PREFIX`` together to move regions without cross-region reads.)

W&B runs are logged to ``marin-community/speedrun``.

Submit (preemptible), one coordinator per command:

    MARIN_PREFIX=gs://marin-us-central1 \\
    .venv/bin/iris --config lib/iris/config/marin.yaml job run \\
      --no-wait \\
      --preemptible \\
      --region us-central1 \\
      --extra tpu \\
      -e WANDB_API_KEY "$WANDB_API_KEY" \\
      -e MARIN_PREFIX "gs://marin-us-central1" \\
      -- python -m experiments.speedrun.hybrid_muon_qwen3_scaling.sweep

``--extra tpu`` is REQUIRED: it makes the worker `uv sync --extra tpu` (jax 0.9.2 +
libtpu), which propagates to the child training jobs via IRIS_JOB_EXTRAS. Without it
the workers install the default jax 0.10 line with no libtpu, JAX falls back to CPU,
and every job dies with "No accelerator found" (which iris mislabels as a bad node).

``--region us-central1`` pins the COORDINATOR to the same region as MARIN_PREFIX/REGION/
data. Pinning only the child steps (REGION below) is not enough: the coordinator does
constant `.executor_status.lock` bookkeeping on the prefix, so a region mismatch turns
every lock op into cross-region egress. (``--region`` is mutually exclusive with
``--reserve``, so we drop the reservation and let children autoscale.) Keep REGION,
MARIN_PREFIX, --region, and the data bucket all in the SAME region.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.hybrid_muon import HybridMuonConfig, HybridMuonVariant
from marin.execution.executor import ExecutorStep, executor_main

from experiments.defaults import default_train
from experiments.llama import llama_150m, llama_300m, llama_600m
from experiments.prebuilt_caches import fineweb_edu_subcache_10B
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)

REGION = "us-central1"
SEQ_LEN = 4096
WANDB_ENTITY = "marin-community"
WANDB_PROJECT = "speedrun"
VARIANTS: tuple[HybridMuonVariant, ...] = ("a", "f")
# Trimmed 2026-06-01 (preemptible coordinator-livelock): cut from {0.5,0.75,1.0,1.25,1.5}
# to the 3 LRs bracketing the observed optimum (130m best=lrx0_75, 300m best~lrx0_5/lrx1).
LR_MULTIPLIERS: tuple[float, ...] = (0.5, 0.75, 1.0)


@dataclass(frozen=True)
class SizeSpec:
    """Per-size sweep settings, mirroring the PRISM-Berkeley Qwen3 sweep."""

    label: str
    llama_cfg: LlamaConfig
    muon_lr: float  # center HybridMuon (matrix) learning rate
    adam_lr: float  # center AdamW learning rate (embeddings / lm_head / biases)
    train_batch_size: int
    num_train_steps: int
    tpu_variant: str
    momentum: float
    adam_epsilon: float
    max_grad_norm: float
    decay: float


# Centers, batch sizes, step counts and resources copied verbatim from the
# PRISM-Berkeley selected-runs metadata (marin#4933). The two large sizes run on
# a single larger slice rather than the original multi-slice layout: the global
# train_batch_size is unchanged, so the optimization is identical (only fewer
# chips / longer wall-clock), and single-slice avoids DCN fragility.
# 1_2b dropped 2026-06-01: its 22888-step runs cannot finish under the ~15-min
# preemptible coordinator-restart cycle, and they were blocking variant f.
SIZES: tuple[SizeSpec, ...] = (
    SizeSpec("130m", llama_150m, 0.016, 0.0032, 128, 4959, "v5p-8", 0.95, 1e-15, 1.0, 0.8),
    SizeSpec("300m", llama_300m, 0.006, 0.0018, 128, 11444, "v5p-8", 0.98, 1e-15, 1.0, 0.8),
    SizeSpec("520m", llama_600m, 0.004, 0.0012, 256, 9918, "v5p-16", 0.98, 1e-25, 1.0, 1.0),
)


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
    # 0.5 -> "0_5", 0.75 -> "0_75", 1.0 -> "1", 1.25 -> "1_25", 1.5 -> "1_5"
    text = f"{multiplier:g}".replace(".", "_")
    return f"lrx{text}"


def _override_tracker(step: ExecutorStep) -> ExecutorStep:
    """Point the run's W&B tracker at marin-community/speedrun."""
    pod = step.config
    inner = pod.train_config
    trainer = inner.trainer
    tracker = dataclasses.replace(trainer.tracker, entity=WANDB_ENTITY, project=WANDB_PROJECT)
    trainer = dataclasses.replace(trainer, tracker=tracker)
    inner = dataclasses.replace(inner, trainer=trainer)
    pod = dataclasses.replace(pod, train_config=inner)
    return dataclasses.replace(step, config=pod)


def _build_step(size: SizeSpec, variant: HybridMuonVariant, multiplier: float) -> ExecutorStep:
    muon_lr = size.muon_lr * multiplier
    adam_lr = size.adam_lr * multiplier

    optimizer = HybridMuonConfig(
        variant=variant,
        lr=muon_lr,
        learning_rate=muon_lr,  # lr_scheduler reads `learning_rate` (the `lr` field is vestigial)
        adam_lr=adam_lr,
        momentum=size.momentum,
        nesterov=True,
        backend_steps=5,
        weight_decay=0.1,
        beta1=0.8,
        beta2=0.98,
        epsilon=size.adam_epsilon,
        muon_epsilon=1e-5,
        max_grad_norm=size.max_grad_norm,
        use_kimi_scaling=False,
        lr_schedule="linear",
        warmup=0,
        rewarmup=0,
        decay=size.decay,
        min_lr_ratio=0,
    )

    train = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(size.tpu_variant, regions=[REGION]),
        train_seq_len=SEQ_LEN,
        train_batch_size=size.train_batch_size,
        num_train_steps=size.num_train_steps,
        learning_rate=muon_lr,
        optimizer_config=optimizer,
    )

    run_id = f"qwen3_{size.label}_hybrid_muon_{variant}_{SEQ_LEN}_{_lr_label(multiplier)}"
    step = default_train(
        name=run_id,
        tokenized=fineweb_edu_subcache_10B,
        model_config=_to_qwen3_from_llama(size.llama_cfg),
        train_config=train,
        tags=["speedrun", "hybrid_muon", f"variant_{variant}", f"qwen3_{size.label}"],
        use_default_validation=True,
        eval_harness_tasks=(),
        wandb_name=run_id,
        wandb_group=f"hybrid_muon_{variant}_qwen3_scaling",
    )
    return _override_tracker(step)


def main() -> None:
    if os.getenv("CI") is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    steps = [
        _build_step(size, variant, multiplier) for variant in VARIANTS for size in SIZES for multiplier in LR_MULTIPLIERS
    ]
    executor_main(
        steps=steps,
        description=(
            "Qwen3 scaling LR sweep for HybridMuon variants a & f "
            "(reproduces marin#4933 PRISM-Berkeley setting), region-pinned to us-east5."
        ),
    )


if __name__ == "__main__":
    main()
