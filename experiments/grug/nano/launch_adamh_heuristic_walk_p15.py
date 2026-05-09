# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Heuristic-AdamH walk p15: p14 (full MoE at moe d768 compute-optimal) +
intra-doc attention masking.

Same architecture, schedule, and heuristic LRs as p14. The only change:
set ``intra_doc_bos_id=50256`` on the model, which makes
``Transformer.__call__`` derive ``segment_ids`` from the `<|endoftext|>`
markers in the input tokens and attach them to the attention mask. See the
muon p15 launch and ``NanoModelConfig.intra_doc_bos_id`` for why this
lives in the model rather than the data config.
"""

import dataclasses

from fray.cluster import ResourceConfig
from jax.sharding import PartitionSpec as P
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch import _fineweb_gpt2_data
from experiments.grug.nano.launch_adamh_heuristic import (
    NanoAdamHHeuristicLaunchConfig,
    _resolve_run_id,
    build_heuristic_optimizer,
    run_nano_adamh_heuristic_trial,
)
from experiments.grug.nano.launch_adamh_heuristic_walk_p14 import P14_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P15_TRAIN_STEPS = 10343
P15_BATCH_SIZE = 64

# Architecture is identical to p14 except for `intra_doc_bos_id` — the
# model itself derives segment ids from the gpt2 EOT marker (50256) and
# attaches them to the attention mask. See `NanoModelConfig.intra_doc_bos_id`.
P15_MODEL = dataclasses.replace(P14_MODEL, intra_doc_bos_id=50256)

P15_OPTIMIZER = build_heuristic_optimizer(
    batch_size=P15_BATCH_SIZE,
    num_train_steps=P15_TRAIN_STEPS,
    seq_len=P15_MODEL.max_seq_len,
    hidden_dim=P15_MODEL.hidden_dim,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-heuristic-p15-v2")


nano_adamh_heuristic_p15_trial = ExecutorStep(
    name="grug/nano-adamh-heuristic-p15-trial",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(P15_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P15_TRAIN_STEPS),
        batch_size=versioned(P15_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "fineweb-gpt2", "heuristic", "p15", "moe", "intradoc-mask"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P15_OPTIMIZER),
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
                eval_batch_size=P15_BATCH_SIZE,
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
        steps=[nano_adamh_heuristic_p15_trial],
        description="adamh-heuristic p15: p14 + intra-doc attention masking (CausalLmDataset path).",
    )
