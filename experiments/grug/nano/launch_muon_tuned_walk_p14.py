# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon walk p14: p13 (full MoE) trained out to the moe d768 compute-optimal
step count.

Same MoE architecture and schedule shape as p13 (b=64, seq=4096, no MoE-axis
config knobs change), but ``num_train_steps`` is bumped from 6700 to 10343 to
exactly match `experiments/grug/moe/heuristic.py:build_from_heuristic` at
hidden_dim=768, budget=1.7e18 FLOPs:

    BS=64, seq=4096, steps=10343  =>  TPB=262144, tokens=2.711B

For muon the LR/WD are not heuristic-derived (held at lr=0.035, wd=0.025); the
extra training is what's being ablated. The cooldown_frac=0.7 schedule still
applies and rescales over the new total step count.
"""

import dataclasses

from fray.cluster import ResourceConfig
from jax.sharding import PartitionSpec as P
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch import (
    NanoAdamWMuonConfig,
    NanoLaunchConfig,
    _fineweb_gpt2_data,
    _resolve_run_id,
    run_nano_trial,
)
from experiments.grug.nano.launch_muon_tuned_walk_p13 import P13_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P14_TRAIN_STEPS = 10343
P14_BATCH_SIZE = 64
NANO_MUON_TUNED_LR = 0.035
NANO_MUON_TUNED_WD = 0.025

# Architecture is identical to p13 — only the step count changes.
P14_MODEL = dataclasses.replace(P13_MODEL)

P14_OPTIMIZER = NanoAdamWMuonConfig(
    muon_lr=NANO_MUON_TUNED_LR,
    muon_weight_decay=NANO_MUON_TUNED_WD,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-tuned-p14")


nano_muon_tuned_p14_trial = ExecutorStep(
    name="grug/nano-muon-tuned-p14-trial",
    fn=run_nano_trial,
    config=NanoLaunchConfig(
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
            tags=["grug", "nano", "muon", "fineweb-gpt2", "tuned", "p14", "moe", "compute-optimal"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P14_OPTIMIZER),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=0.0,
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
                # Keep ~one eval per ~250 steps; with 10343 steps that's ~41 evals
                # per run, plus the step-0 and final-step evals.
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
        steps=[nano_muon_tuned_p14_trial],
        description="muon-tuned p14: full MoE at moe d768 compute-optimal (b=64, steps=10343, tokens=2.71B).",
    )
