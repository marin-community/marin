# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Heuristic-AdamH walk p1: drop logit cap.

Step 1 of the heuristic adamh walk. Identical to
`launch_adamh_heuristic.py:nano_adamh_heuristic_trial` except `logit_cap=0`,
`steps=3350`, and `initializer_std = 0.5 / sqrt(hidden_dim)` to match the
moe heuristic's init recipe. (We're already using moe's heuristic for the lr,
beta2, eps; using its init keeps everything consistent.)

For hidden_dim=768 this is `0.5/sqrt(768) ≈ 0.0180`. p2-p7 inherit this
init via dataclasses.replace.
"""

import dataclasses
import math

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

P1_TRAIN_STEPS = 3350
P1_BATCH_SIZE = 512


P1_MODEL = dataclasses.replace(
    NANO_124M_HEURISTIC_MODEL,
    logit_cap=0.0,
    # Match the moe heuristic's init: std = 0.5 / sqrt(hidden_dim).
    initializer_std=0.5 / math.sqrt(NANO_124M_HEURISTIC_MODEL.hidden_dim),
)


P1_OPTIMIZER = build_heuristic_optimizer(
    batch_size=P1_BATCH_SIZE,
    num_train_steps=P1_TRAIN_STEPS,
    seq_len=P1_MODEL.max_seq_len,
    hidden_dim=P1_MODEL.hidden_dim,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-heuristic-p1")


nano_adamh_heuristic_p1_trial = ExecutorStep(
    name="grug/nano-adamh-heuristic-p1-trial",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(P1_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P1_TRAIN_STEPS),
        batch_size=versioned(P1_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic", "p1", "nocap"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P1_OPTIMIZER),
        grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=P1_BATCH_SIZE,
                steps_per_eval=125,
                max_eval_batches=20,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(steps=[nano_adamh_heuristic_p1_trial], description="adamh heuristic p1: nocap, 3350 steps.")
