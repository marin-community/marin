# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rerun no-router-zloss d512 with frequent checkpoints for activation analysis."""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

model, optimizer, batch, num_steps = build_from_heuristic(budget=2.19e17, hidden_dim=512)
model_no_zloss = dataclasses.replace(model, router_z_loss_coef=0.0)

_CHECKPOINT_KEEP = [
    {"every": 20, "until": 300},
    {"every": 250, "until": 1000},
    {"every": 1000, "until": None},
]

# Baseline with frequent checkpoints
baseline_ckpt = ExecutorStep(
    name="grug/baseline-d512-ckpt",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id="baseline-d512-ckpt",
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        enable_cross_region_ckpt_read=True,
        steps=versioned(num_steps),
        batch_size=versioned(batch),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["baseline", "ckpt-sweep", "d=512"],
            group="baseline-ckpt",
            name="baseline-d512-ckpt",
        ),
        optimizer=versioned(optimizer),
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
        checkpoint_keep=versioned(_CHECKPOINT_KEEP),
    ),
)

# No router z-loss with frequent checkpoints
run_id = "no-router-zloss-d512-ckpt"
no_router_zloss_ckpt = ExecutorStep(
    name=f"grug/{run_id}",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(model_no_zloss),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=run_id,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        enable_cross_region_ckpt_read=True,
        steps=versioned(num_steps),
        batch_size=versioned(batch),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["no-router-zloss", "ckpt-sweep", "d=512"],
            group="no-router-zloss-ckpt",
            name=run_id,
        ),
        optimizer=versioned(optimizer),
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
        checkpoint_keep=versioned(_CHECKPOINT_KEEP),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[baseline_ckpt, no_router_zloss_ckpt],
        description="Baseline + no-router-zloss d512 with frequent checkpoints.",
    )
