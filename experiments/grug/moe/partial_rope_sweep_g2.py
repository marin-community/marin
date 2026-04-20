# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Partial RoPE gate 2: d1024 and d1280.

GitHub issue: https://github.com/marin-community/marin/issues/4946
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

GATE2_CONFIGS: list[tuple[int, float]] = [
    (1024, 9.00e18),
    (1280, 2.83e19),
]

PARTIAL_ROPE_MODES: list[str] = ["every_layer", "every_4th"]


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for mode in PARTIAL_ROPE_MODES:
        for dim, budget in GATE2_CONFIGS:
            model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
            model = dataclasses.replace(model, partial_rope_mode=mode)
            run_id = f"partial-rope-{mode}-d{dim}-{budget:.2e}"

            steps.append(
                ExecutorStep(
                    name=f"grug/{run_id}",
                    fn=run_grug_moe_trial,
                    config=GrugMoeLaunchConfig(
                        model=versioned(model),
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
                            tags=["partial-rope", f"d={dim}", f"budget={budget:.2e}", f"mode={mode}"],
                            group="partial-rope",
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
                    ),
                )
            )
    return steps


all_steps = _make_steps()

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description="Partial RoPE gate 2: every_layer and every_4th at d1024/d1280.",
    )
