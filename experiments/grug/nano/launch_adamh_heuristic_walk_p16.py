# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Heuristic-AdamH walk p16: switch to grug/moe's data + tokenizer + loss
kernel. Same compute-optimal schedule as p14 (b=64, seq=4096, steps=10343).

Three changes on top of p14:

  1. **Data: nemotron_mix** (same as ``experiments/grug/moe/launch.py``),
     tokenized with llama3 (vocab=128_256). Replaces fineweb10B-gpt2.
  2. **Vocab: 128_256** (llama3) instead of 50_304 (gpt2).
  3. **Fused CE kernel**: route the lm-head + softmax + cross-entropy
     through ``levanter.grug.loss.fused_linear_softmax_cross_entropy_loss``
     (the kernel grug/moe uses) via ``use_fused_ce=True``. Requires
     ``use_bias=False`` (no lm-head bias slot) and ``logit_cap=None``.
  4. **Intra-doc masking via levanter's data path**: the llama3 tokenizer
     properly registers ``bos_token_id=128000`` and ``eos_token_id=128001``,
     so ``LmDataConfig.block_cross_document_attention=True`` (the default
     in ``lm_mixture_data_config``) actually fires.

Heuristic LRs are recomputed for the new vocab through ``hidden_dim`` (lr
formula uses dim, not vocab) but vocab affects FLOPs/token, not LR. So
``adam_lr=2.515e-3`` and ``adamh_lr=0.01090`` carry over from p14.
"""

import dataclasses

from fray.cluster import ResourceConfig
from jax.sharding import PartitionSpec as P
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.nano.launch_adamh_heuristic import (
    NanoAdamHHeuristicLaunchConfig,
    _resolve_run_id,
    build_heuristic_optimizer,
    run_nano_adamh_heuristic_trial,
)
from experiments.grug.nano.launch_adamh_heuristic_walk_p14 import P14_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig
from experiments.pretraining_datasets import nemotron_mix

P16_TRAIN_STEPS = 10343
P16_BATCH_SIZE = 64

P16_MODEL = dataclasses.replace(
    P14_MODEL,
    vocab_size=128_256,
    logit_cap=None,
    intra_doc_bos_id=None,
    use_fused_ce=True,
)

P16_OPTIMIZER = build_heuristic_optimizer(
    batch_size=P16_BATCH_SIZE,
    num_train_steps=P16_TRAIN_STEPS,
    seq_len=P16_MODEL.max_seq_len,
    hidden_dim=P16_MODEL.hidden_dim,
)


NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix,
    default_validation_sets(tokenizer=nemotron_mix.tokenizer),
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-adamh-heuristic-p16")


nano_adamh_heuristic_p16_trial = ExecutorStep(
    name="grug/nano-adamh-heuristic-p16-trial",
    fn=run_nano_adamh_heuristic_trial,
    config=NanoAdamHHeuristicLaunchConfig(
        model=versioned(P16_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P16_TRAIN_STEPS),
        batch_size=versioned(P16_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "adamh", "nemotron", "heuristic", "p16", "moe", "fused-ce", "llama3"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P16_OPTIMIZER),
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
                eval_batch_size=P16_BATCH_SIZE,
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
        steps=[nano_adamh_heuristic_p16_trial],
        description="adamh-heuristic p16: nemotron + llama3 vocab + fused CE + levanter intra-doc mask.",
    )
