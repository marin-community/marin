# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Heuristic-AdamH walk p7: p6 + SwiGLU MLP at intermediate_dim = 3 * hidden_dim."""

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
from experiments.grug.nano.launch_adamh_heuristic_walk_p6 import P6_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P7_TRAIN_STEPS = 3350
P7_BATCH_SIZE = 128


# SwiGLU has 3 matrices (gate, up, down) so we shrink the intermediate dim from
# 4*hidden_dim (ReLU^2 default) to 3*hidden_dim. Net per-layer FLOPs are nearly
# identical (2*4*D^2 vs 3*3*D^2 = 16D^2 vs 18D^2; SwiGLU is ~12% more).
P7_MODEL = dataclasses.replace(
    P6_MODEL,
    mlp_type="swiglu",
    intermediate_dim=3 * P6_MODEL.hidden_dim,  # 3 * 768 = 2304
)


P7_OPTIMIZER = build_heuristic_optimizer(
    batch_size=P7_BATCH_SIZE,
    num_train_steps=P7_TRAIN_STEPS,
    seq_len=P7_MODEL.max_seq_len,
    hidden_dim=P7_MODEL.hidden_dim,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-heuristic-p7")


nano_adamh_heuristic_p7_trial = ExecutorStep(
    name="grug/nano-adamh-heuristic-p7-trial",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(P7_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P7_TRAIN_STEPS),
        batch_size=versioned(P7_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic", "p7", "swiglu"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P7_OPTIMIZER),
        grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=P7_BATCH_SIZE,
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
        steps=[nano_adamh_heuristic_p7_trial],
        description="adamh heuristic p7: + SwiGLU at 3*D intermediate, 3350 steps.",
    )
