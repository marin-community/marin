# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""k5e256s5: 256 experts pick 5, half-dim shared and routed experts.

GitHub issue: https://github.com/marin-community/marin/issues/5298
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

GATE1_CONFIGS: list[tuple[int, float]] = [
    (512, 2.19e17),
    (768, 1.70e18),
]

GATE2_CONFIGS: list[tuple[int, float]] = [
    (1024, 9.00e18),
    (1280, 2.83e19),
]


def _make_steps(configs: list[tuple[int, float]], tpu_type: str = "v5p-8") -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for dim, budget in configs:
        model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
        shared_dim = math.ceil(dim / 2 / 128) * 128
        model = dataclasses.replace(
            model,
            num_experts=256,
            num_experts_per_token=5,
            shared_expert_intermediate_dim=shared_dim,
        )
        run_id = f"k5e256s5-d{dim}-{budget:.2e}"

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
                        tags=["k5e256s5", f"d={dim}", f"budget={budget:.2e}"],
                        group="k5e256s5",
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


gate1_steps = _make_steps(GATE1_CONFIGS, tpu_type="v5p-8")
gate2_steps = _make_steps(GATE2_CONFIGS, tpu_type="v5p-16")
all_steps = gate1_steps + gate2_steps

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description="k5e256s5: 256 experts pick 5, half-dim shared and routed.",
    )
