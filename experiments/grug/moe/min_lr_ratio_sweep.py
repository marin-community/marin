# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Min LR ratio ablation: test non-zero end-of-schedule learning rates.

Baseline uses min_lr_ratio=0.0 (LR decays fully to zero).
Test 0.05, 0.1, 0.15 to see if retaining some LR at end improves loss.

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

GATE1_CONFIGS: list[tuple[int, float]] = [
    (512, 2.19e17),
    (768, 1.70e18),
]

MIN_LR_RATIOS: list[float] = [0.05, 0.1, 0.15]


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for ratio in MIN_LR_RATIOS:
        for dim, budget in GATE1_CONFIGS:
            model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
            optimizer = dataclasses.replace(optimizer, min_lr_ratio=ratio)
            ratio_label = str(ratio).replace(".", "_")
            run_id = f"min-lr-{ratio_label}-d{dim}-{budget:.2e}"

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
                            tags=["min-lr-ratio", f"d={dim}", f"budget={budget:.2e}", f"ratio={ratio}"],
                            group="min-lr-ratio",
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
        description="Min LR ratio sweep: 0.05/0.1/0.15 at gate 1 scales.",
    )
