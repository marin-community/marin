# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""may_arch: PKO+prope+last_layer_pko + AdamH embed + k6e256s5 experts.

LR sweep at d512 and d768 with multipliers 0.6, 0.8, 1.0, 1.2, 1.4.

GitHub issue: https://github.com/marin-community/marin/issues/5371
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

LR_MULTIPLIERS: list[float] = [
    # 0.6,  # finished/running
    # 0.8,  # finished/running
    # 1.0,  # finished/running
    1.2,
    # 1.4,  # finished/running
]


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for lr_mult in LR_MULTIPLIERS:
        for dim, budget in GATE1_CONFIGS:
            model, optimizer, batch, num_steps = build_from_heuristic(budget=budget, hidden_dim=dim)
            shared_dim = math.ceil(dim / 2 / 128) * 128
            model = dataclasses.replace(
                model,
                partial_key_offset="every_4th",
                use_partial_rope=True,
                last_layer_pko=True,
                num_experts=256,
                num_experts_per_token=6,
                shared_expert_intermediate_dim=shared_dim,
            )
            optimizer = dataclasses.replace(
                optimizer,
                learning_rate=optimizer.learning_rate * lr_mult,
                adam_lr=optimizer.adam_lr * lr_mult,
            )
            if optimizer.expert_lr is not None:
                optimizer = dataclasses.replace(optimizer, expert_lr=optimizer.expert_lr * lr_mult)

            lr_label = f"{lr_mult:.1f}x"
            run_id = f"may-arch-lr{lr_label}-d{dim}-{budget:.2e}"

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
                            tags=["may-arch", lr_label, f"d={dim}", f"budget={budget:.2e}"],
                            group="may-arch",
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
        description="may_arch LR sweep: 0.6x-1.4x at d512/d768.",
    )
