# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test d512 learning-rate transfer on the d768 norm-preserving residual run.

This launch matches ``MOE-NPR-002-d768`` except that both optimizer learning
rates come from the d512 point. The d768-specific epsilon and beta2 are kept.

Submit on us-east5-a, interactive priority, v5p-8::

    .venv/bin/iris --cluster=marin job run --no-wait --zone us-east5-a --priority interactive \
      -e WANDB_API_KEY "$WANDB_API_KEY" \
      -- python -m experiments.grug.moe_norm_preserving_residual.launch_d768_d512_lr
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep, this_output_path, versioned

from experiments.grug.moe_norm_preserving_residual.heuristic import MoeHeuristic
from experiments.grug.moe_norm_preserving_residual.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe_norm_preserving_residual.train import GrugEvalConfig, GrugTrainerConfig

_SEQ_LEN: int = 8192
_TPU: str = "v5p-8"
_GROUP_NAME: str = "MOE-NPR-gate1-issue-8860"
_RUN_ID: str = "MOE-NPR-003-d768-d512lr"

_REFERENCE_DIM: int = 512
_REFERENCE_BATCH_SIZE: int = 16
_REFERENCE_STEPS: int = 10_980

_TARGET_DIM: int = 768
_TARGET_BATCH_SIZE: int = 32
_TARGET_STEPS: int = 16_875


def _build_step() -> ExecutorStep:
    heuristic = MoeHeuristic()
    model = dataclasses.replace(
        heuristic.build_model_config(_TARGET_DIM, seq_len=_SEQ_LEN),
        disable_pko=True,
        disable_long_rope=True,
    )

    reference_tokens = float(_REFERENCE_STEPS * _REFERENCE_BATCH_SIZE * _SEQ_LEN)
    reference_optimizer = heuristic.build_optimizer_config(
        _REFERENCE_BATCH_SIZE,
        reference_tokens,
        _REFERENCE_DIM,
        seq_len=_SEQ_LEN,
    )

    target_tokens = float(_TARGET_STEPS * _TARGET_BATCH_SIZE * _SEQ_LEN)
    target_optimizer = heuristic.build_optimizer_config(
        _TARGET_BATCH_SIZE,
        target_tokens,
        _TARGET_DIM,
        seq_len=_SEQ_LEN,
    )
    optimizer = dataclasses.replace(
        target_optimizer,
        learning_rate=reference_optimizer.learning_rate,
        adam_lr=reference_optimizer.adam_lr,
    )

    return ExecutorStep(
        name=f"grug/{_RUN_ID}",
        fn=run_grug_moe_trial,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=_RUN_ID,
            resources=versioned(ResourceConfig.with_tpu(_TPU)),
            steps=versioned(_TARGET_STEPS),
            batch_size=versioned(_TARGET_BATCH_SIZE),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                entity="marin-community",
                project="dial_moe",
                tags=[
                    "moe",
                    "moe_may_compute_opt",
                    "july_baseline",
                    "norm_preserving_residual",
                    "d512_lr_transfer",
                    "issue_8860",
                    "gqa",
                    "no_pko",
                    "no_long_rope",
                    "d768",
                ],
                group=_GROUP_NAME,
                name=None,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=0.0, ema_beta=None, log_every=1)),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=256,
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


if __name__ == "__main__":
    executor_main(
        steps=[_build_step()],
        description=(
            "d768 norm-preserving residual run with the d512 learning-rate pair. " f"run_id={_RUN_ID}, TPU={_TPU}."
        ),
    )
