# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Good 10T experiment: inv-sqrt LR schedule for MoE.

Compares inverse-square-root learning-rate decay against the default cosine
schedule on the standard MoE trial configuration. Everything else (model, data,
resources, training steps) is identical to the cosine baseline in launch.py so
the comparison is apples-to-apples.

Tracking issue: https://github.com/marin-community/marin/issues/4028
"""

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.launch import (
    GRUG_MOE_TRIAL_MODEL,
    GrugMoeLaunchConfig,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    _resolve_run_id,
    run_grug_moe,
)

RESOLVED_RUN_ID = _resolve_run_id("grug-moe-inv-sqrt-lr")

# Same optimizer as the cosine baseline but with inv_sqrt schedule. The decay
# fraction is omitted because inv_sqrt decays continuously from peak rather
# than using a cosine-style stable/decay split.
INV_SQRT_OPTIMIZER = AdamConfig(
    learning_rate=3e-3,
    weight_decay=0.1,
    lr_schedule="inv_sqrt",
    min_lr_ratio=0.1,
    warmup=1000,
)

grug_moe_inv_sqrt_lr = ExecutorStep(
    name="grug/moe-inv-sqrt-lr",
    fn=run_grug_moe,
    config=GrugMoeLaunchConfig(
        model=versioned(GRUG_MOE_TRIAL_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(2_000),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "moe", "inv-sqrt-lr", "good-10t"],
            group="grug-moe-inv-sqrt-lr",
            name=None,
        ),
        optimizer=versioned(INV_SQRT_OPTIMIZER),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[grug_moe_inv_sqrt_lr],
        description="Good 10T: inv-sqrt LR schedule for MoE (issue #4028).",
    )
