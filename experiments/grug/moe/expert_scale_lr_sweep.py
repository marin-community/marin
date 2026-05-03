# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Routed expert output scale + expert LR sweep on may-arch k4e256.

GitHub issue: https://github.com/marin-community/marin/issues/5399
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

EXPERT_SCALES: list[float] = [1.0, 2.4]
EXPERT_LR_MULTS: list[float] = [0.6, 0.8, 1.0, 1.2, 1.4]


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for scale in EXPERT_SCALES:
        for lr_mult in EXPERT_LR_MULTS:
            for dim, budget in GATE1_CONFIGS:
                model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
                model = dataclasses.replace(
                    model,
                    partial_key_offset="every_4th",
                    use_partial_rope=True,
                    last_layer_pko=True,
                    num_experts=256,
                    num_experts_per_token=4,
                    routed_expert_scale=scale,
                )
                expert_lr = optimizer.learning_rate * lr_mult
                optimizer = dataclasses.replace(optimizer, expert_lr=expert_lr)

                scale_label = f"s{scale:.1f}" if scale != 1.0 else "s1"
                lr_label = f"elr{lr_mult:.1f}x"
                run_id = f"expert-{scale_label}-{lr_label}-d{dim}-{budget:.2e}"

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
                                tags=["expert-scale-lr", scale_label, lr_label, f"d={dim}"],
                                group="expert-scale-lr",
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
        description="Expert output scale + expert LR sweep: 2 scales x 5 LR mults x 2 sizes.",
    )
