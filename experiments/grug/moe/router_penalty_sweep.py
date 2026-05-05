# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Router penalty variants: L2 logit penalty and z-loss warmdown.

Both with frequent checkpoints for activation analysis.
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

_CHECKPOINT_KEEP = [
    {"every": 20, "until": 300},
    {"every": 250, "until": 1000},
    {"every": 1000, "until": None},
]

model, optimizer, batch, num_steps = build_from_heuristic(budget=2.19e17, hidden_dim=512)

# 1. L2 logit penalty (no z-loss)
model_l2 = dataclasses.replace(model, router_z_loss_coef=0.0, router_l2_loss_coef=0.001)
l2_step = ExecutorStep(
    name="grug/router-l2-v2-d512-ckpt",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(model_l2),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id="router-l2-v2-d512-ckpt",
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        enable_cross_region_ckpt_read=True,
        steps=versioned(num_steps),
        batch_size=versioned(batch),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["router-l2", "ckpt-sweep", "d=512"],
            group="router-penalty",
            name="router-l2-v2-d512-ckpt",
        ),
        optimizer=versioned(optimizer),
        grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=512, steps_per_eval=1000, max_eval_batches=8, eval_current=True, eval_ema=False
            )
        ),
        checkpoint_keep=versioned(_CHECKPOINT_KEEP),
    ),
)

# 2. Z-loss warmdown (decay to 0 over first 10% of training)
warmdown_step = ExecutorStep(
    name="grug/zloss-warmdown-d512-ckpt",
    fn=run_grug_moe_trial,
    config=GrugMoeLaunchConfig(
        model=versioned(model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id="zloss-warmdown-d512-ckpt",
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        enable_cross_region_ckpt_read=True,
        steps=versioned(num_steps),
        batch_size=versioned(batch),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["zloss-warmdown", "ckpt-sweep", "d=512"],
            group="router-penalty",
            name="zloss-warmdown-d512-ckpt",
        ),
        optimizer=versioned(optimizer),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
                router_z_loss_warmdown_frac=0.1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=512, steps_per_eval=1000, max_eval_batches=8, eval_current=True, eval_ema=False
            )
        ),
        checkpoint_keep=versioned(_CHECKPOINT_KEEP),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[l2_step],
        description="Router penalty variants: L2 logit + z-loss warmdown, with frequent checkpoints.",
    )
