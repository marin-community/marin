# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GatedNorm init scale sweep — expanded (gate 1 + gate 2).

Init multipliers 1.5, 2.0, 2.5, 3.0 at all 4 scales.
Skips init=2.0 at d512/d768 (already run in gate 1 sweep).

GitHub issue: https://github.com/marin-community/marin/issues/4904
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

MULTS = [1.5, 2.0, 2.5, 3.0]

# Already completed in gate 1 sweep
SKIP = {(2.0, 512), (2.0, 768)}


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for mult in MULTS:
        for dim, budget in ALL_CONFIGS:
            if (mult, dim) in SKIP:
                continue
            model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
            variant_model = dataclasses.replace(model, gated_norm_init_mult=mult)
            mult_label = str(mult).replace(".", "_")
            run_id = f"gn-init-{mult_label}x-d{dim}-{budget:.2e}"

            steps.append(
                ExecutorStep(
                    name=f"grug/{run_id}",
                    fn=run_grug_moe_trial,
                    config=GrugMoeLaunchConfig(
                        model=versioned(variant_model),
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
                            tags=["gn-init-scale", f"d={dim}", f"budget={budget:.2e}", f"mult={mult}"],
                            group="gn-init-scale",
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
        description="GatedNorm init scale sweep: 1.5/2/2.5/3x at all 4 scales (skipping 2x d512/d768).",
    )
