# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Heuristic-AdamH walk p6: p5 + sliding window (3-short / 1-long)."""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch_adamh_heuristic import (
    NanoAdamHHeuristicLaunchConfig,
    _fineweb_gpt2_data,
    _resolve_run_id,
    build_heuristic_optimizer,
    run_nano_adamh_heuristic_trial,
)
from experiments.grug.nano.launch_adamh_heuristic_walk_p5 import P5_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P6_TRAIN_STEPS = 3350
P6_BATCH_SIZE = 128


P6_MODEL = dataclasses.replace(P5_MODEL, sliding_window=P5_MODEL.max_seq_len)


P6_OPTIMIZER = build_heuristic_optimizer(
    batch_size=P6_BATCH_SIZE,
    num_train_steps=P6_TRAIN_STEPS,
    seq_len=P6_MODEL.max_seq_len,
    hidden_dim=P6_MODEL.hidden_dim,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-heuristic-p6")


nano_adamh_heuristic_p6_trial = ExecutorStep(
    name="grug/nano-adamh-heuristic-p6-trial",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(P6_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P6_TRAIN_STEPS),
        batch_size=versioned(P6_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic", "p6", "sw"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P6_OPTIMIZER),
        grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=P6_BATCH_SIZE,
                steps_per_eval=125,
                max_eval_batches=20,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(steps=[nano_adamh_heuristic_p6_trial], description="adamh heuristic p6: + sliding window, 3350 steps.")
