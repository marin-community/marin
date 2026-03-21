# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retry the missing iteration_02 AdamH gatedschema slice on v5p-8.

This relaunches only the missing table entries for the non-GatedNorm AdamH
slice: `1e19` `d512` and `d1536`, both on `v5p-8`, with a fresh run prefix.
"""

import math

from fray.cluster import ResourceConfig
from levanter.optim import GrugAttentionMlpLmHeadAdamHConfig
from levanter.tracker.wandb import WandbConfig
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

BUDGETS: tuple[float, ...] = (1e19,)
HIDDEN_DIMS: tuple[int, ...] = (512, 1536)

REFERENCE_BUDGET = 1e18
REFERENCE_HIDDEN_DIM = 512
REFERENCE_ADAMH_LR = 0.007292038680986272
RUN_NAME_PREFIX = "isoflop-moe-adamh-r3-gatedschema-v5p8-retry2-missing"
GROUP = "isoflop-moe-adamh-r3-gatedschema-v5p8-retry2-missing"
TPU_VARIANT = "v5p-8"

REFERENCE_MODEL_CONFIG = _build_model_config(REFERENCE_HIDDEN_DIM)
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


def _run_id(*, budget: float, hidden_dim: int) -> str:
    return f"{RUN_NAME_PREFIX}-{budget:.0e}-d{hidden_dim}"


def create_retry_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []

    for budget in BUDGETS:
        for hidden_dim in HIDDEN_DIMS:
            model_cfg = _build_model_config(hidden_dim)
            flops_per_token = _compute_flops_per_token(model_cfg)
            _tokens, batch_size, train_steps = _compute_tokens_and_batch(budget, flops_per_token)
            adamh_lr = _compute_adamh_lr(batch_size=batch_size, train_steps=train_steps)
            adam_lr = _compute_adam_lr(batch_size=batch_size, train_steps=train_steps)
            beta2 = max(0.95, 0.98 ** (batch_size / 128))
            run_id = _run_id(budget=budget, hidden_dim=hidden_dim)

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
                    tags=[
                        "grug",
                        "moe-core",
                        "iteration-02",
                        "v3",
                        "isoflop",
                        "attn-mlp-lmh-adamh",
                        "retry",
                        "retry2",
                        "missing-table-rows",
                        "retry-on-v5p8",
                        "source=isoflop-moe-adamh-r3-gatedschema",
                        f"budget={budget:.0e}",
                        f"d={hidden_dim}",
                        f"hardware={TPU_VARIANT}",
                        f"adamh_lr={adamh_lr:g}",
                        f"adam_lr={adam_lr:g}",
                        f"batch_size={batch_size}",
                        f"train_steps={train_steps}",
                    ],
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
                        z_loss_weight=1e-4,
                        ema_beta=None,
                        log_every=1,
                    )
                ),
                eval=versioned(
                    GrugEvalConfig(
                        eval_batch_size=512,
                        steps_per_eval=1000,
                        max_eval_batches=8,
                        eval_current=True,
                        eval_ema=False,
                    )
                ),
            )

            steps.append(
                ExecutorStep(
                    name=f"grug/{run_id}",
                    fn=run_grug_moe_trial,
                    config=config,
                )
            )

    return steps


retry_steps = create_retry_steps()


if __name__ == "__main__":
    executor_main(
        steps=retry_steps,
        description=(
            "Retry the missing isoflop AdamH r3 gatedschema d512/d1536 slice on v5p-8 "
            "with fresh run names"
        ),
    )
