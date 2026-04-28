# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared expert sizing v2.

Variant 1: Remove shared expert, use E=96 K=6 (all 4 sizes).
Variant 2: Keep shared, set expert_dim to 0.75*hidden_dim (d512 + d1024 only).
Variant 3: Shared=expert_dim (~d/2), E=80 K=5 (all 4 sizes).

GitHub issue: https://github.com/marin-community/marin/issues/4768
"""

import dataclasses
import math

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


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []

    # Variant 4: wider experts (0.75*d), NO shared — all 4 sizes
    for dim, budget in [(512, 2.19e17), (768, 1.70e18), (1024, 9.00e18), (1280, 2.83e19)]:
        model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
        wider_expert_dim = math.ceil(0.75 * dim / 128) * 128
        model = dataclasses.replace(model, intermediate_dim=wider_expert_dim, shared_expert_intermediate_dim=0)
        run_id = f"wider-no-shared-0_75x-d{dim}-{budget:.2e}"

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
                        tags=["shared-expert-v2", "wider-no-shared", f"d={dim}", f"budget={budget:.2e}"],
                        group="shared-expert-v2",
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

    # === Already submitted variants below (commented out) ===

    # Variant 1: no shared, E=96, K=6 — all 4 sizes
    for dim, budget in [(512, 2.19e17), (768, 1.70e18), (1024, 9.00e18), (1280, 2.83e19)]:
        model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
        model = dataclasses.replace(model, num_experts=96, num_experts_per_token=6, shared_expert_intermediate_dim=0)
        run_id = f"no-shared-e96k6-d{dim}-{budget:.2e}"

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
                        tags=["shared-expert-v2", "no-shared", f"d={dim}", f"budget={budget:.2e}"],
                        group="shared-expert-v2",
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

    # Variant 2: wider experts (0.75*d), keep shared — d512 + d1024 only
    for dim, budget in [(512, 2.19e17), (1024, 9.00e18)]:
        model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
        wider_expert_dim = math.ceil(0.75 * dim / 128) * 128
        model = dataclasses.replace(model, intermediate_dim=wider_expert_dim)
        run_id = f"wider-expert-0_75x-d{dim}-{budget:.2e}"

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
                        tags=["shared-expert-v2", "wider-expert", f"d={dim}", f"budget={budget:.2e}"],
                        group="shared-expert-v2",
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

    # Variant 3: shared=expert_dim (~d/2), E=80, K=5 — all 4 sizes
    for dim, budget in [(512, 2.19e17), (768, 1.70e18), (1024, 9.00e18), (1280, 2.83e19)]:
        model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
        expert_dim = model.intermediate_dim  # already ~d/2
        model = dataclasses.replace(model, num_experts=80, num_experts_per_token=5, shared_expert_intermediate_dim=expert_dim)
        run_id = f"small-shared-e80k5-d{dim}-{budget:.2e}"

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
                        tags=["shared-expert-v2", "small-shared", f"d={dim}", f"budget={budget:.2e}"],
                        group="shared-expert-v2",
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
        description="Shared expert v2: no-shared E96K6 + wider 0.75x + small-shared E80K5.",
    )
