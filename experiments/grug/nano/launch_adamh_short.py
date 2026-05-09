# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Short (3600-step) variant of `nano_adamh_trial`.

Same architecture, init, and AdamH+AdamW hyperparameters as
`experiments/grug/nano/launch_adamh.py:nano_adamh_trial`. Only the step count
shrinks from 4875 to 3600 so the run is wall-clock-comparable to the Muon
launch (`experiments/grug/nano/launch.py:nano_trial`).

The hardcoded ref hyperparameters do not depend on step count, so we reuse
`NanoAdamHRefConfig()` unchanged. The 250-step warmup remains 250 steps
(no longer 5.1% of the run; now 6.9%) since the ref defines warmup in absolute
steps, not as a fraction.
"""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch_adamh import (
    NANO_124M_ADAMH_MODEL,
    NanoAdamHLaunchConfig,
    NanoAdamHRefConfig,
    _fineweb_gpt2_data,
    _resolve_run_id,
    run_nano_adamh_trial,
)
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

NANO_ADAMH_SHORT_TRAIN_STEPS = 3600


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-short-rsqrt_cap")


nano_adamh_short_trial = ExecutorStep(
    name="grug/nano-adamh-short-trial",
    fn=run_nano_adamh_trial,
    config=NanoAdamHLaunchConfig(
        model=versioned(NANO_124M_ADAMH_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(NANO_ADAMH_SHORT_TRAIN_STEPS),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2", "short"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(NanoAdamHRefConfig()),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=0.0,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=512,
                steps_per_eval=125,
                max_eval_batches=20,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[nano_adamh_short_trial],
        description="Nano (modded-nanogpt) 124M, AdamH+AdamW, 3600 steps on fineweb10B-gpt2.",
    )
