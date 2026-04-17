#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses

from experiments.grug.moe.launch import (
    ExecutorStep,
    GrugEvalConfig,
    GrugMoeLaunchConfig,
    GrugTrainerConfig,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    WandbConfig,
    _baseline_batch,
    _baseline_model,
    _baseline_optimizer,
    _baseline_steps,
    _resolve_run_id,
    executor_main,
    run_grug_moe_trial,
    this_output_path,
    versioned,
)
from fray.cluster import ResourceConfig

RUN_ID = _resolve_run_id("moe_1e23_d5120_bs2048_ep8_ragged_48l_rayuvtpu_20260417_0945")
STEP_NAME = "grug/moe_1e23_d5120_bs2048_ep8_ragged_48l_fix_a2a_20260417_0945"


ragged_ep8_fix = ExecutorStep(
    name=STEP_NAME,
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(
            dataclasses.replace(
                _baseline_model,
                moe_implementation="ragged_all_to_all",
                use_array_stacked_blocks=True,
                num_layers=48,
            )
        ),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v4-2048", regions=["us-central2"])),
        steps=versioned(_baseline_steps),
        batch_size=versioned(_baseline_batch),
        expert_parallel=versioned(8),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["adamh", "qb", "sharded-qb", "gatednorm", "xsa", "zloss", "eq3e3", "ragged-fix"],
            group="moe-iter04",
            name=None,
        ),
        optimizer=versioned(_baseline_optimizer),
        priority_band="production",
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=1024,
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
        steps=[ragged_ep8_fix],
        description="Grug MoE 1e23 ragged EP8 relaunch after ragged_all_to_all offset fix.",
    )
