# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retry gatednorm 1e19 d2048 on v5p-16 with a smaller eval batch.

The previous v5p-16 low-parallelism attempt progressed through training and
failed right as eval started. This retry keeps the same train microbatching,
but drops eval batch size further to reduce eval-side memory pressure.
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
BUDGET = 1e19
HIDDEN_DIM = 2048
TRAIN_PER_DEVICE_PARALLELISM = 2
EVAL_BATCH_SIZE = 16
RUN_ID = "isoflop-moe-adamh-gatednorm-v5p16-eval16-r1-1e19-d2048"
GROUP = "isoflop-moe-adamh-gatednorm-v5p16-eval16-r1"

REFERENCE_BUDGET = 1e18
REFERENCE_HIDDEN_DIM = 512
REFERENCE_ADAMH_LR = 0.007292038680986272

REFERENCE_MODEL_CONFIG = dataclasses.replace(_build_model_config(REFERENCE_HIDDEN_DIM), gated_norm_rank=GATED_NORM_RANK)
REFERENCE_FLOPS_PER_TOKEN = _compute_flops_per_token(REFERENCE_MODEL_CONFIG)
_, REFERENCE_BATCH_SIZE, REFERENCE_TRAIN_STEPS = _compute_tokens_and_batch(REFERENCE_BUDGET, REFERENCE_FLOPS_PER_TOKEN)
REFERENCE_ADAM_LR = min(0.01, (0.33 * math.sqrt(REFERENCE_BATCH_SIZE)) / REFERENCE_HIDDEN_DIM)


def _compute_data_size(*, batch_size: int, train_steps: int) -> float:
    return float(batch_size * train_steps)


def _lr_score(*, batch_size: int, train_steps: int) -> float:
    data_size = _compute_data_size(batch_size=batch_size, train_steps=train_steps)
    return math.sqrt(batch_size) / (data_size**0.25)


REFERENCE_LR_SCORE = _lr_score(batch_size=REFERENCE_BATCH_SIZE, train_steps=REFERENCE_TRAIN_STEPS)
ADAMH_LR_SCALE = REFERENCE_ADAMH_LR / REFERENCE_LR_SCORE
ADAM_LR_SCALE = REFERENCE_ADAM_LR / REFERENCE_LR_SCORE


def _compute_adamh_lr(*, batch_size: int, train_steps: int) -> float:
    return ADAMH_LR_SCALE * _lr_score(batch_size=batch_size, train_steps=train_steps)


def _compute_adam_lr(*, batch_size: int, train_steps: int) -> float:
    return ADAM_LR_SCALE * _lr_score(batch_size=batch_size, train_steps=train_steps)


def create_retry_step() -> ExecutorStep:
    model_cfg = dataclasses.replace(_build_model_config(HIDDEN_DIM), gated_norm_rank=GATED_NORM_RANK)
    flops_per_token = _compute_flops_per_token(model_cfg)
    _tokens, batch_size, train_steps = _compute_tokens_and_batch(BUDGET, flops_per_token)
    adamh_lr = _compute_adamh_lr(batch_size=batch_size, train_steps=train_steps)
    adam_lr = _compute_adam_lr(batch_size=batch_size, train_steps=train_steps)
    beta2 = max(0.95, 0.98 ** (batch_size / 128))

    config = GrugMoeLaunchConfig(
        model=versioned(model_cfg),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RUN_ID,
        resources=versioned(ResourceConfig.with_tpu(TPU_VARIANT)),
        steps=versioned(train_steps),
        batch_size=versioned(batch_size),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=[
                "grug",
                "moe-core",
                "iteration-02",
                "v3",
                "isoflop",
                "attn-mlp-lmh-adamh",
                "gatednorm",
                f"gated_norm_rank={GATED_NORM_RANK}",
                f"budget={BUDGET:.0e}",
                f"d={HIDDEN_DIM}",
                f"hardware={TPU_VARIANT}",
                "retry",
                "eval-batch-reduced",
                f"adamh_lr={adamh_lr:g}",
                f"adam_lr={adam_lr:g}",
                f"batch_size={batch_size}",
                f"train_steps={train_steps}",
                f"per_device_parallelism={TRAIN_PER_DEVICE_PARALLELISM}",
                f"eval_batch_size={EVAL_BATCH_SIZE}",
            ],
            group=GROUP,
            name=RUN_ID,
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
        name=f"grug/{RUN_ID}",
        fn=run_grug_moe_trial,
        config=config,
    )


retry_step = create_retry_step()


if __name__ == "__main__":
    executor_main(
        steps=[retry_step],
        description="Retry gatednorm 1e19 d2048 on v5p-16 with eval_batch_size=16",
    )
