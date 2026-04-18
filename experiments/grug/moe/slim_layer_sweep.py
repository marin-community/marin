# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Slim layer sweep: replace one MoE layer at a time with a dense FFN.

For each layer index, replaces the MoE + shared expert with a single dense FFN
of matching active param count. Identifies which layers benefit least from
expert routing.

GitHub issue: https://github.com/marin-community/marin/issues/4900
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


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for dim, budget in GATE1_CONFIGS:
        model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
        num_layers = model.num_layers
        for layer_idx in range(num_layers):
            variant_model = dataclasses.replace(model, slim_layer_index=layer_idx)
            run_id = f"slim-layer-{layer_idx}-d{dim}-{budget:.2e}"

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
                            tags=["slim-layer", f"d={dim}", f"budget={budget:.2e}", f"layer={layer_idx}"],
                            group="slim-layer",
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
        description="Slim layer sweep: replace each MoE layer with dense at gate 1 scales.",
    )
