# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Partial key offset sweep: partial RoPE + key shift for induction heads.

GitHub issue: https://github.com/marin-community/marin/issues/4802
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

PKO_VARIANTS: list[tuple[str, str]] = [
    ("every_4th", "every_4th"),
    ("every_layer", "every_layer"),
]


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for dim, budget in GATE1_CONFIGS:
        model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
        for label, mode in PKO_VARIANTS:
            variant_model = dataclasses.replace(model, partial_key_offset=mode)
            run_id = f"partial-key-offset-{label}-d{dim}-{budget:.2e}"

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
                        steps=versioned(num_steps),
                        batch_size=versioned(batch),
                        seed=versioned(0),
                        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                        tracker=WandbConfig(
                            project="dial_moe",
                            tags=["partial-key-offset", f"d={dim}", f"budget={budget:.2e}", label],
                            group="partial-key-offset",
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
        description="Partial key offset sweep: every_4th and every_layer at gate 1 scales.",
    )
