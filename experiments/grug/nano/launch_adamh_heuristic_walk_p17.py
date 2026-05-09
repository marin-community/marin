# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Heuristic-AdamH walk p17: p16 minus the two AdamH-compensator features.

Drops ``use_attn_gate=False`` and ``use_gated_norm=False`` on top of p16.
Everything else (nemotron + llama3 + fused CE + levanter intra-doc mask +
full MoE + compute-optimal schedule) carries over.

Hypothesis: see ``launch_muon_tuned_walk_p17.py``. The interesting thing
for the AdamH walk is whether removing the magnitude knobs hurts more
than it does for muon — gated_norm helped adamh at p4 (-0.014).
"""

import dataclasses

from fray.cluster import ResourceConfig
from jax.sharding import PartitionSpec as P
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch_adamh_heuristic import (
    NanoAdamHHeuristicLaunchConfig,
    _resolve_run_id,
    build_heuristic_optimizer,
    run_nano_adamh_heuristic_trial,
)
from experiments.grug.nano.launch_adamh_heuristic_walk_p16 import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    P16_MODEL,
)
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P17_TRAIN_STEPS = 10343
P17_BATCH_SIZE = 64

P17_MODEL = dataclasses.replace(
    P16_MODEL,
    use_attn_gate=False,
    use_gated_norm=False,
)

P17_OPTIMIZER = build_heuristic_optimizer(
    batch_size=P17_BATCH_SIZE,
    num_train_steps=P17_TRAIN_STEPS,
    seq_len=P17_MODEL.max_seq_len,
    hidden_dim=P17_MODEL.hidden_dim,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-heuristic-p17")


nano_adamh_heuristic_p17_trial = ExecutorStep(
    name="grug/nano-adamh-heuristic-p17-trial",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(P17_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P17_TRAIN_STEPS),
        batch_size=versioned(P17_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "nemotron", "heuristic", "p17", "moe", "fused-ce", "no-gates"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P17_OPTIMIZER),
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
                eval_batch_size=P17_BATCH_SIZE,
                steps_per_eval=250,
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
        steps=[nano_adamh_heuristic_p17_trial],
        description="adamh-heuristic p17: p16 minus attn_gate and gated_norm.",
    )
