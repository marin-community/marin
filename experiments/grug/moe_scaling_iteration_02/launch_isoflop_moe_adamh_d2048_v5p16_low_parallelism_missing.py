# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retry the missing d2048 rows on v5p-16 with smaller train/eval parallelism.

This parent launches only the d2048 rows still missing from the comparison
table:

- gatedschema: 1e19 d2048
- gatednorm: 3e18 d2048
- gatednorm: 1e19 d2048

The runs use v5p-16 with a smaller train microbatch and smaller eval batch to
reduce per-device memory pressure while keeping the global train batch from the
original isoflop schedule.
"""

import dataclasses
import math

from fray.cluster import ResourceConfig
from levanter.optim import GrugAttentionMlpLmHeadAdamHConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe_scaling_iteration_02.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugEvalConfig,
    GrugMoeLaunchConfig,
    GrugTrainerConfig,
    _build_model_config,
    _compute_flops_per_token,
    _compute_tokens_and_batch,
    run_grug_moe_trial,
)

GATED_NORM_RANK = 16
TPU_VARIANT = "v5p-16"
HIDDEN_DIM = 2048
TRAIN_PER_DEVICE_PARALLELISM = 2
EVAL_BATCH_SIZE = 64
GROUP = "isoflop-moe-adamh-d2048-v5p16-lowparallelism"

ADAMH_REFERENCE_BUDGET = 1e18
ADAMH_REFERENCE_HIDDEN_DIM = 512
ADAMH_REFERENCE_ADAMH_LR = 0.007292038680986272

GATEDNORM_REFERENCE_BUDGET = 1e18
GATEDNORM_REFERENCE_HIDDEN_DIM = 512
GATEDNORM_REFERENCE_ADAMH_LR = 0.007292038680986272

PLAIN_REFERENCE_MODEL_CONFIG = _build_model_config(ADAMH_REFERENCE_HIDDEN_DIM)
PLAIN_REFERENCE_FLOPS_PER_TOKEN = _compute_flops_per_token(PLAIN_REFERENCE_MODEL_CONFIG)
_, PLAIN_REFERENCE_BATCH_SIZE, PLAIN_REFERENCE_TRAIN_STEPS = _compute_tokens_and_batch(
    ADAMH_REFERENCE_BUDGET, PLAIN_REFERENCE_FLOPS_PER_TOKEN
)
PLAIN_REFERENCE_ADAM_LR = min(0.01, (0.33 * math.sqrt(PLAIN_REFERENCE_BATCH_SIZE)) / ADAMH_REFERENCE_HIDDEN_DIM)

GATED_REFERENCE_MODEL_CONFIG = dataclasses.replace(
    _build_model_config(GATEDNORM_REFERENCE_HIDDEN_DIM),
    gated_norm_rank=GATED_NORM_RANK,
)
GATED_REFERENCE_FLOPS_PER_TOKEN = _compute_flops_per_token(GATED_REFERENCE_MODEL_CONFIG)
_, GATED_REFERENCE_BATCH_SIZE, GATED_REFERENCE_TRAIN_STEPS = _compute_tokens_and_batch(
    GATEDNORM_REFERENCE_BUDGET, GATED_REFERENCE_FLOPS_PER_TOKEN
)
GATED_REFERENCE_ADAM_LR = min(0.01, (0.33 * math.sqrt(GATED_REFERENCE_BATCH_SIZE)) / GATEDNORM_REFERENCE_HIDDEN_DIM)


def _compute_data_size(*, batch_size: int, train_steps: int) -> float:
    return float(batch_size * train_steps)


def _lr_score(*, batch_size: int, train_steps: int) -> float:
    data_size = _compute_data_size(batch_size=batch_size, train_steps=train_steps)
    return math.sqrt(batch_size) / (data_size**0.25)


PLAIN_REFERENCE_LR_SCORE = _lr_score(batch_size=PLAIN_REFERENCE_BATCH_SIZE, train_steps=PLAIN_REFERENCE_TRAIN_STEPS)
PLAIN_ADAMH_LR_SCALE = ADAMH_REFERENCE_ADAMH_LR / PLAIN_REFERENCE_LR_SCORE
PLAIN_ADAM_LR_SCALE = PLAIN_REFERENCE_ADAM_LR / PLAIN_REFERENCE_LR_SCORE

GATED_REFERENCE_LR_SCORE = _lr_score(batch_size=GATED_REFERENCE_BATCH_SIZE, train_steps=GATED_REFERENCE_TRAIN_STEPS)
GATED_ADAMH_LR_SCALE = GATEDNORM_REFERENCE_ADAMH_LR / GATED_REFERENCE_LR_SCORE
GATED_ADAM_LR_SCALE = GATED_REFERENCE_ADAM_LR / GATED_REFERENCE_LR_SCORE


def _compute_adamh_lr(*, batch_size: int, train_steps: int, gated_norm: bool) -> float:
    scale = GATED_ADAMH_LR_SCALE if gated_norm else PLAIN_ADAMH_LR_SCALE
    return scale * _lr_score(batch_size=batch_size, train_steps=train_steps)


def _compute_adam_lr(*, batch_size: int, train_steps: int, gated_norm: bool) -> float:
    scale = GATED_ADAM_LR_SCALE if gated_norm else PLAIN_ADAM_LR_SCALE
    return scale * _lr_score(batch_size=batch_size, train_steps=train_steps)


def _run_id(*, budget: float, gated_norm: bool) -> str:
    if gated_norm:
        prefix = "isoflop-moe-adamh-gatednorm-v5p16-lowparallelism"
    else:
        prefix = "isoflop-moe-adamh-r3-gatedschema-v5p16-lowparallelism"
    return f"{prefix}-{budget:.0e}-d{HIDDEN_DIM}"


def _build_step(*, budget: float, gated_norm: bool) -> ExecutorStep:
    model_cfg = _build_model_config(HIDDEN_DIM)
    if gated_norm:
        model_cfg = dataclasses.replace(model_cfg, gated_norm_rank=GATED_NORM_RANK)

    flops_per_token = _compute_flops_per_token(model_cfg)
    _tokens, batch_size, train_steps = _compute_tokens_and_batch(budget, flops_per_token)
    adamh_lr = _compute_adamh_lr(batch_size=batch_size, train_steps=train_steps, gated_norm=gated_norm)
    adam_lr = _compute_adam_lr(batch_size=batch_size, train_steps=train_steps, gated_norm=gated_norm)
    beta2 = max(0.95, 0.98 ** (batch_size / 128))
    run_id = _run_id(budget=budget, gated_norm=gated_norm)

    tags = [
        "grug",
        "moe-core",
        "iteration-02",
        "v3",
        "isoflop",
        "attn-mlp-lmh-adamh",
        "mesh-expert=1-on-v5p",
        "retry",
        "missing-table-rows",
        "d2048",
        "lowparallelism",
        "smaller-per-device-parallelism",
        f"budget={budget:.0e}",
        f"d={HIDDEN_DIM}",
        f"hardware={TPU_VARIANT}",
        f"adamh_lr={adamh_lr:g}",
        f"adam_lr={adam_lr:g}",
        f"batch_size={batch_size}",
        f"train_steps={train_steps}",
        f"per_device_parallelism={TRAIN_PER_DEVICE_PARALLELISM}",
        f"eval_batch_size={EVAL_BATCH_SIZE}",
    ]
    if gated_norm:
        tags.extend(["gatednorm", f"gated_norm_rank={GATED_NORM_RANK}"])
    else:
        tags.append("gatedschema")

    config = GrugMoeLaunchConfig(
        model=versioned(model_cfg),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=run_id,
        resources=versioned(ResourceConfig.with_tpu(TPU_VARIANT)),
        steps=versioned(train_steps),
        batch_size=versioned(batch_size),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=tags,
            group=GROUP,
            name=run_id,
        ),
        optimizer=versioned(
            GrugAttentionMlpLmHeadAdamHConfig(
                learning_rate=adamh_lr,
                adam_lr=adam_lr,
                beta1=0.96,
                beta2=beta2,
                epsilon=1e-15,
                weight_decay=0.1,
                lr_schedule="linear",
                decay=None,
                min_lr_ratio=0.0,
                warmup=0.1,
                max_grad_norm=1,
            )
        ),
        grug_trainer=versioned(
            GrugTrainerConfig(
                trainer=TrainerConfig(per_device_parallelism=TRAIN_PER_DEVICE_PARALLELISM),
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=EVAL_BATCH_SIZE,
                steps_per_eval=1000,
                max_eval_batches=8,
                eval_current=True,
                eval_ema=False,
            )
        ),
    )

    return ExecutorStep(
        name=f"grug/{run_id}",
        fn=run_grug_moe_trial,
        config=config,
    )


def create_d2048_steps() -> list[ExecutorStep]:
    return [
        _build_step(budget=1e19, gated_norm=False),
        _build_step(budget=3e18, gated_norm=True),
        _build_step(budget=1e19, gated_norm=True),
    ]


d2048_steps = create_d2048_steps()


if __name__ == "__main__":
    executor_main(
        steps=d2048_steps,
        description=(
            "Retry the missing d2048 AdamH and GatedNorm rows on v5p-16 with "
            "smaller train and eval parallelism"
        ),
    )
