# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Depth-width shift: ±1 layer from heuristic, recomputing tokens to hit same compute budget.

Tests whether the heuristic's depth formula is optimal or if slightly
deeper/shallower models perform better at the same compute.

GitHub issue: https://github.com/marin-community/marin/issues/TBD
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import (
    MoeAdamHHeuristic,
    build_from_heuristic,
    compute_flops_per_token,
    compute_tokens_and_batch,
)
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

LAYER_DELTAS: list[int] = [-1, +1]


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    heuristic = MoeAdamHHeuristic()
    for delta in LAYER_DELTAS:
        for dim, budget in GATE1_CONFIGS:
            base_model, _, _, _ = build_from_heuristic(budget=budget, hidden_dim=dim)
            model = dataclasses.replace(base_model, num_layers=base_model.num_layers + delta)
            # Recompute tokens/batch/steps for the new FLOPs-per-token.
            fpt = compute_flops_per_token(model)
            tokens, batch, num_steps = compute_tokens_and_batch(budget, fpt)
            # Recompute optimizer for the new tokens/batch.
            optimizer = heuristic.build_optimizer_config(batch, tokens, dim)
            delta_label = f"plus{delta}" if delta > 0 else f"minus{abs(delta)}"
            run_id = f"depth-shift-{delta_label}-d{dim}-{budget:.2e}"

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
                            tags=["depth-shift", f"d={dim}", f"budget={budget:.2e}", f"delta={delta}"],
                            group="depth-shift",
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
        description="Depth-width shift: ±1 layer at gate 1 scales.",
    )
