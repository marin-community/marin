# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Heuristic-AdamH walk p14: p13 (full MoE) trained out to the moe d768
compute-optimal step count.

Same MoE architecture as p13 but ``num_train_steps`` is bumped from 6700 to
10343 to exactly match `experiments/grug/moe/heuristic.py:build_from_heuristic`
at hidden_dim=768, budget=1.7e18 FLOPs:

    BS=64, seq=4096, steps=10343  =>  TPB=262144, tokens=2.711B

The heuristic adamh / adam learning rates re-derive cleanly from the new
total_tokens (LR ∝ tokens^-0.2813), so they drop ~12% vs p13:

    adam_lr  : 2.841e-3 -> 2.515e-3
    adamh_lr : 1.231e-2 -> 1.090e-2
    epsilon  : 7.92e-16 -> 9.84e-16  (eps ∝ sqrt(tokens))
    beta2    : 0.99800  unchanged    (β₂ depends on TPB only)

These match `MoeAdamHHeuristic`'s output for d768 / 1.7e18 FLOPs exactly.
"""

import dataclasses

from fray.cluster import ResourceConfig
from jax.sharding import PartitionSpec as P
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch_adamh_heuristic import (
    NanoAdamHHeuristicLaunchConfig,
    _fineweb_gpt2_data,
    _resolve_run_id,
    build_heuristic_optimizer,
    run_nano_adamh_heuristic_trial,
)
from experiments.grug.nano.launch_adamh_heuristic_walk_p13 import P13_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P14_TRAIN_STEPS = 10343
P14_BATCH_SIZE = 64

# Architecture is identical to p13 — only the step count changes.
P14_MODEL = dataclasses.replace(P13_MODEL)

P14_OPTIMIZER = build_heuristic_optimizer(
    batch_size=P14_BATCH_SIZE,
    num_train_steps=P14_TRAIN_STEPS,
    seq_len=P14_MODEL.max_seq_len,
    hidden_dim=P14_MODEL.hidden_dim,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-heuristic-p14")


nano_adamh_heuristic_p14_trial = ExecutorStep(
    name="grug/nano-adamh-heuristic-p14-trial",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(P14_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P14_TRAIN_STEPS),
        batch_size=versioned(P14_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic", "p14", "moe", "compute-optimal"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P14_OPTIMIZER),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=1e-4,
                ema_beta=None,
                log_every=1,
                # Batch is sharded across both data and expert axes so the
                # MoEMLP shard_map for QB beta sees the right batch axes.
                train_batch_pspec=P(("data", "expert")),
            )
        ),
        eval=versioned(
            GrugEvalConfig(
                eval_batch_size=P14_BATCH_SIZE,
                steps_per_eval=250,
                # 40 batches x 64 (BS) x 4096 (seq) = 10.49M tokens per eval pass.
                max_eval_batches=40,
                eval_current=True,
                eval_ema=False,
                eval_batch_pspec=P(("data", "expert")),
            )
        ),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[nano_adamh_heuristic_p14_trial],
        description="adamh-heuristic p14: full MoE at moe d768 compute-optimal (b=64, steps=10343, tokens=2.71B).",
    )
