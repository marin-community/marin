# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Heuristic-AdamH walk p10: p9 + drop hard-coded attn_scale=0.12 in favor of
1/sqrt(head_dim), and add qk_mult=1.3.

For head_dim=128 the effective softmax temperature shifts from
`1.0 * 0.12 = 0.12` (p9) to `1.3 * (1/sqrt(128)) ≈ 0.115` (p10). This is the
moe convention.
"""

import dataclasses
import math

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
from experiments.grug.nano.launch_adamh_heuristic_walk_p9 import P9_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P10_TRAIN_STEPS = 3350
P10_BATCH_SIZE = 128

P10_MODEL = dataclasses.replace(
    P9_MODEL,
    attn_scale=1.0 / math.sqrt(P9_MODEL.head_dim),
    qk_mult=1.3,
)

P10_OPTIMIZER = build_heuristic_optimizer(
    batch_size=P10_BATCH_SIZE,
    num_train_steps=P10_TRAIN_STEPS,
    seq_len=P10_MODEL.max_seq_len,
    hidden_dim=P10_MODEL.hidden_dim,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-heuristic-p10")


nano_adamh_heuristic_p10_trial = ExecutorStep(
    name="grug/nano-adamh-heuristic-p10-trial",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(P10_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P10_TRAIN_STEPS),
        batch_size=versioned(P10_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic", "p10", "qk_scale"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P10_OPTIMIZER),
        grug_trainer=versioned(GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1)),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=P10_BATCH_SIZE,
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
        steps=[nano_adamh_heuristic_p10_trial],
        description="adamh heuristic p10: attn_scale=1/sqrt(h), qk_mult=1.3, 3350 steps.",
    )
