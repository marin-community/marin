# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Wide attention: 1.5x the default number of heads, head_dim stays at 128.

Increases attention capacity without changing MLP or expert dimensions.
GQA ratio preserved (4:1).

GitHub issue: https://github.com/marin-community/marin/issues/TBD
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import MoeAdamHHeuristic, build_from_heuristic
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
    h = MoeAdamHHeuristic()
    for dim, budget in GATE1_CONFIGS:
        model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
        new_heads = round(1.5 * dim / 128)
        new_kv = h._compute_kv_heads(new_heads, h.gqa_ratio)
        model = dataclasses.replace(model, num_heads=new_heads, num_kv_heads=new_kv)
        run_id = f"wide-attn-1_5x-d{dim}-{budget:.2e}"

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
                        tags=["wide-attn", f"d={dim}", f"budget={budget:.2e}"],
                        group="wide-attn",
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
        description="Wide attention (1.5x heads) sweep at gate 1 scales.",
    )
