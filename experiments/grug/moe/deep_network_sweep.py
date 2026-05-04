# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deep networks: more layers at compute-optimal budget.

Two sweep files: d512 on v5p-8, d768+d1024 on v5p-32.
Each config at LR multipliers [1.0, 0.8, 0.6]. Batch sizes doubled.

GitHub issue: https://github.com/marin-community/marin/issues/5423
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

# (dim, layers, budget, batch, steps)
# Budget and steps derived from N*(C) scaling with actual FPT from heuristic.
# Batch sizes doubled from baseline.
DEEP_CONFIGS_D512: list[tuple[int, int, float, int, int]] = [
    (512, 10, 5.69e17, 64, 4979),
    (512, 14, 1.07e18, 64, 6670),
]

DEEP_CONFIGS_D768_D1024: list[tuple[int, int, float, int, int]] = [
    (768, 12, 3.63e18, 128, 7357),
    (768, 17, 6.96e18, 128, 9958),
    (1024, 14, 1.41e19, 256, 7800),
    (1024, 18, 2.26e19, 256, 9704),
]

LR_MULTIPLIERS: list[float] = [1.0, 0.8, 0.6]


def _make_steps(configs: list[tuple[int, int, float, int, int]], tpu_type: str) -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for dim, layers, budget, batch, num_steps in configs:
        for lr_mult in LR_MULTIPLIERS:
            model, optimizer, _, _ = build_from_heuristic(budget=budget, hidden_dim=dim)
            model = dataclasses.replace(model, num_layers=layers)
            optimizer = dataclasses.replace(
                optimizer,
                learning_rate=optimizer.learning_rate * lr_mult,
                adam_lr=optimizer.adam_lr * lr_mult,
            )
            if optimizer.expert_lr is not None:
                optimizer = dataclasses.replace(optimizer, expert_lr=optimizer.expert_lr * lr_mult)

            lr_label = f"lr{lr_mult:.1f}x"
            run_id = f"deep-d{dim}-L{layers}-{lr_label}-{budget:.2e}"

            steps.append(
                ExecutorStep(
                    name=f"grug/{run_id}",
                    fn=run_grug_moe_trial,
                    config=GrugMoeLaunchConfig(
                        model=versioned(model),
                        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                        output_path=this_output_path(),
                        run_id=run_id,
                        resources=versioned(ResourceConfig.with_tpu(tpu_type)),
                        enable_cross_region_ckpt_read=True,
                        steps=versioned(num_steps),
                        batch_size=versioned(batch),
                        seed=versioned(0),
                        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                        tracker=WandbConfig(
                            project="dial_moe",
                            tags=["deep", f"d={dim}", f"L={layers}", lr_label],
                            group="deep-network",
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


d512_steps = _make_steps(DEEP_CONFIGS_D512, tpu_type="v5p-8")
d768_d1024_steps = _make_steps(DEEP_CONFIGS_D768_D1024, tpu_type="v5p-32")

if __name__ == "__main__":
    import sys

    if "--d512" in sys.argv:
        executor_main(steps=d512_steps, description="Deep networks d512 (v5p-8).")
    elif "--large" in sys.argv:
        executor_main(steps=d768_d1024_steps, description="Deep networks d768+d1024 (v5p-32).")
    else:
        executor_main(steps=d512_steps + d768_d1024_steps, description="Deep networks: all sizes.")
