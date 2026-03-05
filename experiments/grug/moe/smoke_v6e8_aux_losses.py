# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Small Grug-MoE smoke run with router aux losses enabled."""

from __future__ import annotations

import dataclasses
import os

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.launch import (
    GRUG_MOE_TRIAL_MODEL,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig


def _resolve_run_id(default_run_id: str) -> str:
    return os.environ.get("GRUG_MOE_SMOKE_RUN_ID", default_run_id)


def _resolve_tpu_type(default_tpu_type: str) -> str:
    return os.environ.get("GRUG_MOE_TPU_TYPE", default_tpu_type)


RESOLVED_RUN_ID = _resolve_run_id("grug-moe-smoke-v6e8")
RESOLVED_TPU_TYPE = _resolve_tpu_type("v6e-8")


grug_moe_smoke_v6e8_aux_losses = ExecutorStep(
    name="grug/moe-smoke-v6e8-aux-losses",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(
            dataclasses.replace(
                GRUG_MOE_TRIAL_MODEL,
                load_balancing_loss_coef=0.01,
                router_z_loss_coef=0.001,
            )
        ),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        steps=versioned(20),
        batch_size=versioned(32),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            mode="disabled",
            tags=["grug", "moe", "smoke"],
            group="grug-moe-smoke-v6e8",
            name=None,
        ),
        optimizer=versioned(
            AdamConfig(
                learning_rate=3e-3,
                weight_decay=0.1,
                lr_schedule="cosine",
                decay=0.2,
                min_lr_ratio=0.1,
                warmup=10,
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
                eval_batch_size=64,
                steps_per_eval=None,
                max_eval_batches=1,
                eval_current=False,
                eval_ema=False,
            )
        ),
    ),
    resources=ResourceConfig.with_tpu(RESOLVED_TPU_TYPE),
)


if __name__ == "__main__":
    executor_main(
        steps=[grug_moe_smoke_v6e8_aux_losses],
        description="Grug MoE smoke run on v6e-8 with aux losses enabled.",
    )
