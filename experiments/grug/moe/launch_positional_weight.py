# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Positional loss reweighting gate 1: baseline + tanh(pos/10) at d512 and d768.

Runs both a baseline (no reweighting) and the positional weight variant at
each gate 1 scale so we can compare on the same branch/code.
"""

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

GATE1_CONFIGS: list[tuple[int, float]] = [
    (512, 2.19e17),
    (768, 1.70e18),
]

POSITIONAL_TANH_DIVISOR: float = 10.0


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for dim, budget in GATE1_CONFIGS:
        model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)

        base_trainer = GrugTrainerConfig(
            z_loss_weight=1e-4,
            ema_beta=None,
            log_every=1,
        )
        base_eval = GrugEvalConfig(
            eval_batch_size=512,
            steps_per_eval=1000,
            max_eval_batches=8,
            eval_current=True,
            eval_ema=False,
            last_n_eval_tokens=500,
        )

        # Baseline (no positional reweighting)
        bl_run_id = f"pos-weight-baseline-d{dim}-{budget:.2e}"
        steps.append(
            ExecutorStep(
                name=f"grug/{bl_run_id}",
                fn=run_grug_moe_trial,
                config=GrugMoeLaunchConfig(
                    model=versioned(model),
                    data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                    output_path=this_output_path(),
                    run_id=bl_run_id,
                    resources=versioned(ResourceConfig.with_tpu("v5p-8")),
                    enable_cross_region_ckpt_read=True,
                    steps=versioned(num_steps),
                    batch_size=versioned(batch),
                    seed=versioned(0),
                    mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                    tracker=WandbConfig(
                        project="dial_moe",
                        tags=["pos-weight", "baseline", f"d={dim}", f"budget={budget:.2e}"],
                        group="pos-weight",
                        name=bl_run_id,
                    ),
                    optimizer=versioned(optimizer),
                    grug_trainer=versioned(base_trainer),
                    eval=versioned(base_eval),
                ),
            )
        )

        # Positional weight variant
        pw_run_id = f"pos-weight-tanh10-d{dim}-{budget:.2e}"
        steps.append(
            ExecutorStep(
                name=f"grug/{pw_run_id}",
                fn=run_grug_moe_trial,
                config=GrugMoeLaunchConfig(
                    model=versioned(model),
                    data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                    output_path=this_output_path(),
                    run_id=pw_run_id,
                    resources=versioned(ResourceConfig.with_tpu("v5p-8")),
                    enable_cross_region_ckpt_read=True,
                    steps=versioned(num_steps),
                    batch_size=versioned(batch),
                    seed=versioned(0),
                    mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                    tracker=WandbConfig(
                        project="dial_moe",
                        tags=["pos-weight", "tanh10", f"d={dim}", f"budget={budget:.2e}"],
                        group="pos-weight",
                        name=pw_run_id,
                    ),
                    optimizer=versioned(optimizer),
                    grug_trainer=versioned(
                        dataclasses.replace(base_trainer, positional_loss_tanh_divisor=POSITIONAL_TANH_DIVISOR)
                    ),
                    eval=versioned(base_eval),
                ),
            )
        )
    return steps


all_steps = _make_steps()

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description="Positional loss reweighting gate 1: baseline + tanh(pos/10) at d512/d768.",
    )
