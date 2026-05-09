# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Heuristic-AdamH walk p11: p10 + drop num_layers from 12 to 8.

Matches the moe heuristic's `_compute_num_layers(768)` (rounded to 8).
After p11 the model size and per-token compute drop materially: 8 layers vs
12 layers gives ~33% less per-token block compute. Total compute scales
correspondingly — for fixed step count, this is a smaller model trained
on the same number of tokens.
"""

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
from experiments.grug.nano.launch_adamh_heuristic_walk_p10 import P10_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P11_TRAIN_STEPS = 3350
P11_BATCH_SIZE = 128

P11_MODEL = dataclasses.replace(P10_MODEL, num_layers=8)

P11_OPTIMIZER = build_heuristic_optimizer(
    batch_size=P11_BATCH_SIZE,
    num_train_steps=P11_TRAIN_STEPS,
    seq_len=P11_MODEL.max_seq_len,
    hidden_dim=P11_MODEL.hidden_dim,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-heuristic-p11")


nano_adamh_heuristic_p11_trial = ExecutorStep(
    name="grug/nano-adamh-heuristic-p11-trial",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(P11_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P11_TRAIN_STEPS),
        batch_size=versioned(P11_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic", "p11", "L8"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P11_OPTIMIZER),
        grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=P11_BATCH_SIZE,
                steps_per_eval=125,
                max_eval_batches=20,
                eval_current=True,
                eval_ema=False,
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(steps=[nano_adamh_heuristic_p11_trial], description="adamh heuristic p11: num_layers=8, 3350 steps.")
