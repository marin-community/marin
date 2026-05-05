# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""No router z-loss with router-specific LR sweep. d512 with checkpoints."""

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

_CHECKPOINT_KEEP = [
    {"every": 100, "until": 500},
    {"every": 1000, "until": None},
]

model, optimizer, batch, num_steps = build_from_heuristic(budget=2.19e17, hidden_dim=512)
model = dataclasses.replace(model, router_z_loss_coef=0.0)

ROUTER_LR_MULTS = [
    # 0.6, 0.8, 1.2, 1.4,  # already done
    1.7,
    2.0,
    3.0,
]


def _make_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for lr_mult in ROUTER_LR_MULTS:
        router_lr = optimizer.adam_lr * lr_mult
        opt = dataclasses.replace(optimizer, router_lr=router_lr)
        run_id = f"no-rzloss-routerlr{lr_mult:.1f}x-d512-ckpt"

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
                        tags=["no-rzloss", "router-lr-only", f"rlr={lr_mult}", "d=512"],
                        group="no-rzloss-router-lr-only",
                        name=run_id,
                    ),
                    optimizer=versioned(opt),
                    grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
                    eval=versioned(
                        GrugEvalConfig(
                            eval_batch_size=512,
                            steps_per_eval=1000,
                            max_eval_batches=8,
                            eval_current=True,
                            eval_ema=False,
                        )
                    ),
                    checkpoint_keep=versioned(_CHECKPOINT_KEEP),
                ),
            )
        )
    return steps


all_steps = _make_steps()

if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description="No router z-loss + router-only LR sweep d512.",
    )
