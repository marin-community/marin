# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adam LR shift: scale the Adam (non-AdamH) learning rate independently.

AdamH LR stays as computed by heuristic. Adam LR is multiplied by a factor.
Tests whether the 1D params (norms, biases, router, embeddings) benefit
from a different learning rate than the heuristic prescribes.

GitHub issue: https://github.com/marin-community/marin/issues/TBD
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

ALL_CONFIGS: list[tuple[int, float]] = [
    (512, 2.19e17),
    (768, 1.70e18),
    (1024, 9.00e18),
    (1280, 2.83e19),
]

ADAM_LR_FACTORS: list[float] = [0.7, 1.3]


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for factor in ADAM_LR_FACTORS:
        for dim, budget in ALL_CONFIGS:
            model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
            optimizer = dataclasses.replace(optimizer, adam_lr=optimizer.adam_lr * factor)
            factor_label = str(factor).replace(".", "_")
            run_id = f"adam-lr-{factor_label}x-d{dim}-{budget:.2e}"

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
                            tags=["adam-lr-shift", f"d={dim}", f"budget={budget:.2e}", f"factor={factor}"],
                            group="adam-lr-shift",
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
        description="Adam LR shift: 0.7x and 1.3x at all 4 scales.",
    )
