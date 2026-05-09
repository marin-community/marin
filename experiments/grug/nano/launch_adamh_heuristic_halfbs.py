# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Half-batch / double-steps variant of `nano_adamh_heuristic_short_trial`.

Same architecture, init, mask, and schedule shape as
`launch_adamh_heuristic_short.py`, but trades batch size for step count at
constant total compute:

    batch_size: 512 -> 256
    num_steps:  3600 -> 7200
    tpb:        524,288 -> 262,144
    total_tokens: unchanged at 1,887,436,800

The heuristic recomputes every per-tpb / per-tokens hyperparameter:

    adamh_lr  = 0.012071  (was 0.017064 at full batch; scales as sqrt(tpb))
    adam_lr   = 0.002785  (was 0.003938)
    beta1     = 0.9062
    beta2     = 0.998001  (was 0.996006; tied to tpb)
    epsilon   = 8.21e-16  (was 5.81e-16; scales as sqrt(num_steps))
    warmup    = 0.1 -> 720 steps
"""

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch_adamh_heuristic import (
    NANO_124M_HEURISTIC_MODEL,
    NanoAdamHHeuristicLaunchConfig,
    _fineweb_gpt2_data,
    _resolve_run_id,
    build_heuristic_optimizer,
    run_nano_adamh_heuristic_trial,
)
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

NANO_ADAMH_HALFBS_TRAIN_STEPS = 7200
NANO_ADAMH_HALFBS_BATCH_SIZE = 256


# Recompute every heuristic hyperparameter at the new (batch_size, num_steps).
# Total tokens is preserved, but tpb (-> beta2, eps_scale, sqrt(tpb)) changes.
NANO_124M_HEURISTIC_HALFBS_OPTIMIZER = build_heuristic_optimizer(
    batch_size=NANO_ADAMH_HALFBS_BATCH_SIZE,
    num_train_steps=NANO_ADAMH_HALFBS_TRAIN_STEPS,
    seq_len=NANO_124M_HEURISTIC_MODEL.max_seq_len,
    hidden_dim=NANO_124M_HEURISTIC_MODEL.hidden_dim,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-heuristic-halfbs-rsqrt_cap")


nano_adamh_heuristic_halfbs_trial = ExecutorStep(
    name="grug/nano-adamh-heuristic-halfbs-trial",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(NANO_124M_HEURISTIC_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(NANO_ADAMH_HALFBS_TRAIN_STEPS),
        batch_size=versioned(NANO_ADAMH_HALFBS_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic", "halfbs"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(NANO_124M_HEURISTIC_HALFBS_OPTIMIZER),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=NANO_ADAMH_HALFBS_BATCH_SIZE,
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
        steps=[nano_adamh_heuristic_halfbs_trial],
        description="Nano (modded-nanogpt) 124M, AdamH heuristic, batch=256, 7200 steps on fineweb10B-gpt2.",
    )
