# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep capacity_factor for the MoE grug variant.

Runs the trial model at several capacity factors to determine whether the
default 1.25 is safe or whether it masks avoidable overflow or throughput loss.

See: https://github.com/marin-community/marin/issues/4017
"""

import dataclasses

from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.launch import (
    GRUG_MOE_TRIAL_MODEL,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe,
)
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig

CAPACITY_FACTORS = [1.0, 1.125, 1.25, 1.5, 2.0]


def _build_sweep_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for cf in CAPACITY_FACTORS:
        tag = f"cf{cf:.3f}".replace(".", "p")
        model = dataclasses.replace(GRUG_MOE_TRIAL_MODEL, capacity_factor=cf)
        run_id = f"grug-moe-sweep-cf-{tag}"
        step = ExecutorStep(
            name=f"grug/moe-sweep-cf-{tag}",
            fn=run_grug_moe,
            config=GrugMoeLaunchConfig(
                model=versioned(model),
                data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                output_path=this_output_path(),
                run_id=run_id,
                resources=versioned(ResourceConfig.with_tpu("v5p-8")),
                steps=versioned(2_000),
                batch_size=versioned(512),
                seed=versioned(0),
                mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                tracker=WandbConfig(
                    project="marin",
                    tags=["grug", "moe", "sweep", "capacity-factor"],
                    group="grug-moe-sweep-capacity-factor",
                    name=None,
                ),
                optimizer=versioned(
                    AdamConfig(
                        learning_rate=3e-3,
                        weight_decay=0.1,
                        lr_schedule="cosine",
                        decay=0.2,
                        min_lr_ratio=0.1,
                        warmup=1000,
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
            ),
        )
        steps.append(step)
    return steps


sweep_steps = _build_sweep_steps()

if __name__ == "__main__":
    executor_main(
        steps=sweep_steps,
        description="Sweep capacity_factor over {1.0, 1.125, 1.25, 1.5, 2.0} for the MoE grug trial model.",
    )
