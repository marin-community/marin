# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch isoflop sweep: 3 sizes x 3 budgets x 3 expert configs = 27 runs.

Sweep over model sizes (d512, d768, d1024), compute budgets (1e18, 3e18, 1e19),
and expert configurations (k4e256, k5e256, k6e256). All use shared_expert_dim =
hidden_dim/2, PKO+prope+last_layer_pko, AdamH embed, no long window.

Steps recalculated per config to match the target compute budget.

GitHub issue: https://github.com/marin-community/marin/issues/5371
"""

import dataclasses
import math

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import build_from_heuristic, compute_flops_per_token
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

DIMS: list[int] = [512, 768, 1024]
BUDGETS: list[float] = [1e18, 3e18, 1e19]
EXPERT_CONFIGS: list[tuple[int, int, str]] = [
    # (num_experts_per_token, num_experts, label)
    (4, 256, "k4e256"),
    (5, 256, "k5e256"),
    (6, 256, "k6e256"),
]


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for k, e, label in EXPERT_CONFIGS:
        for dim in DIMS:
            for budget in BUDGETS:
                model, optimizer, batch, _ = build_from_heuristic(budget=budget, hidden_dim=dim)
                shared_dim = math.ceil(dim / 2 / 128) * 128
                model = dataclasses.replace(
                    model,
                    partial_key_offset="every_4th",
                    use_partial_rope=True,
                    last_layer_pko=True,
                    num_experts=e,
                    num_experts_per_token=k,
                    shared_expert_intermediate_dim=shared_dim,
                )

                # Recalculate steps to match target budget with new FLOPs per token
                fpt = compute_flops_per_token(model)
                tokens = budget / (3 * fpt)
                seq_len = 4096
                num_steps = max(1, round(tokens / (batch * seq_len)))

                run_id = f"isoflop-{label}-d{dim}-{budget:.0e}"

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
                                tags=["isoflop", label, f"d={dim}", f"budget={budget:.0e}"],
                                group="may-arch-isoflop",
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
        description="may_arch isoflop: 3 sizes x 3 budgets x 3 expert configs.",
    )
