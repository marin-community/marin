# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Short (3600-step) variant of `nano_adamh_heuristic_trial`.

Same architecture, init, mask, and schedule shape as
`experiments/grug/nano/launch_adamh_heuristic.py:nano_adamh_heuristic_trial`,
but the optimizer is **rebuilt at 3600 steps** so the heuristic-derived
hyperparameters (`adam_lr`, `adamh_lr`, `epsilon`) reflect the new total token
count. β2 is unchanged because it depends only on tokens-per-batch, which
stays constant.

Heuristic outputs at this scale (batch=512, seq=1024, steps=3600, dim=768,
total_tokens=1.887e9):

    adamh_lr  = 0.017064
    adam_lr   = 0.003938
    β1        = 0.9062
    β2        = 0.996006     (same as 4875-step run; tied to tpb)
    ε         = 5.81e-16
    warmup    = 0.1 -> 360 steps
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

NANO_ADAMH_HEURISTIC_SHORT_TRAIN_STEPS = 3600


# Recompute optimizer hyperparams at the new step count. `total_tokens`
# changes from 2.556e9 -> 1.887e9, so adam_lr / adamh_lr / epsilon all shift;
# beta2 stays put because it only depends on tokens-per-batch.
NANO_124M_HEURISTIC_SHORT_OPTIMIZER = build_heuristic_optimizer(
    batch_size=512,
    num_train_steps=NANO_ADAMH_HEURISTIC_SHORT_TRAIN_STEPS,
    seq_len=NANO_124M_HEURISTIC_MODEL.max_seq_len,
    hidden_dim=NANO_124M_HEURISTIC_MODEL.hidden_dim,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-heuristic-short-rsqrt_cap")


nano_adamh_heuristic_short_trial = ExecutorStep(
    name="grug/nano-adamh-heuristic-short-trial",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(NANO_124M_HEURISTIC_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(NANO_ADAMH_HEURISTIC_SHORT_TRAIN_STEPS),
        batch_size=versioned(512),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic", "short"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(NANO_124M_HEURISTIC_SHORT_OPTIMIZER),
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
        steps=[nano_adamh_heuristic_short_trial],
        description="Nano (modded-nanogpt) 124M, AdamH heuristic, 3600 steps on fineweb10B-gpt2.",
    )
