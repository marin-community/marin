# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch isoflop k4e256s1: 256 experts pick 4, shared_dim = hidden_dim.

Same as the other isoflop configs but shared expert is full hidden_dim
instead of half.

GitHub issue: https://github.com/marin-community/marin/issues/5371
"""

import dataclasses

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


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for dim in DIMS:
        for budget in BUDGETS:
            model, optimizer, batch, _ = build_from_heuristic(budget=budget, hidden_dim=dim)
            # shared_expert_intermediate_dim stays at hidden_dim (the heuristic default)
            model = dataclasses.replace(
                model,
                partial_key_offset="every_4th",
                use_partial_rope=True,
                last_layer_pko=True,
                num_experts=256,
                num_experts_per_token=4,
            )

            fpt = compute_flops_per_token(model)
            tokens = budget / (3 * fpt)
            num_steps = max(1, round(tokens / (batch * 4096)))

            run_id = f"isoflop-k4e256s1-d{dim}-{budget:.0e}"

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
                            tags=["isoflop", "k4e256s1", f"d={dim}", f"budget={budget:.0e}"],
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
        description="may_arch isoflop k4e256s1: full-size shared expert.",
    )
