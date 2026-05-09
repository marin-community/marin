# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon walk p16: switch to grug/moe's data + tokenizer + loss kernel.

Three changes on top of p14 (full MoE at moe d768 compute-optimal):

  1. **Data: nemotron_mix** (same as ``experiments/grug/moe/launch.py``),
     tokenized with llama3 (vocab=128_256). Replaces fineweb10B-gpt2.
  2. **Vocab: 128_256** (llama3) instead of 50_304 (gpt2).
  3. **Fused CE kernel**: route the lm-head + softmax + cross-entropy
     through ``levanter.grug.loss.fused_linear_softmax_cross_entropy_loss``
     (the kernel grug/moe uses) via ``use_fused_ce=True``. This requires
     ``use_bias=False`` (no lm-head bias slot) and ``logit_cap=None``
     (the fused kernel only knows the tanh soft-cap; we want no cap to
     mirror moe). The fused kernel's `shard_map` + `pmean` also avoids
     the explicit-mesh-axes psum mismatch that bit our manual CE.
  4. **Intra-doc masking via levanter's data path**: the llama3 tokenizer
     properly registers ``bos_token_id=128000`` and ``eos_token_id=128001``,
     so ``LmDataConfig.block_cross_document_attention=True`` (the default
     in ``lm_mixture_data_config``) actually fires. The model-side
     ``intra_doc_bos_id`` workaround is unset.

Same compute-optimal schedule as p14: b=64, seq=4096, steps=10343 (2.71B
tokens). Muon LR/WD remain at 0.035 / 0.025 (not heuristic-derived).
"""

import dataclasses

from fray.cluster import ResourceConfig
from jax.sharding import PartitionSpec as P
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.nano.launch import (
    NanoAdamWMuonConfig,
    NanoLaunchConfig,
    _resolve_run_id,
    run_nano_trial,
)
from experiments.grug.nano.launch_muon_tuned_walk_p14 import P14_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig
from experiments.pretraining_datasets import nemotron_mix

P16_TRAIN_STEPS = 10343
P16_BATCH_SIZE = 64
NANO_MUON_TUNED_LR = 0.035
NANO_MUON_TUNED_WD = 0.025

P16_MODEL = dataclasses.replace(
    P14_MODEL,
    vocab_size=128_256,
    logit_cap=None,
    intra_doc_bos_id=None,
    use_fused_ce=True,
)

P16_OPTIMIZER = NanoAdamWMuonConfig(
    muon_lr=NANO_MUON_TUNED_LR,
    muon_weight_decay=NANO_MUON_TUNED_WD,
)


# Same data config as `experiments/grug/moe/launch.py`: nemotron_mix +
# default validation sets (paloma, etc.), all tokenized with llama3.
NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix,
    default_validation_sets(tokenizer=nemotron_mix.tokenizer),
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-tuned-p16")


nano_muon_tuned_p16_trial = ExecutorStep(
    name="grug/nano-muon-tuned-p16-trial",
    fn=run_nano_trial,
    config=NanoLaunchConfig(
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
            tags=["grug", "nano", "muon", "nemotron", "tuned", "p16", "moe", "fused-ce", "llama3"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P16_OPTIMIZER),
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
                eval_batch_size=P16_BATCH_SIZE,
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
        steps=[nano_muon_tuned_p16_trial],
        description="muon-tuned p16: nemotron + llama3 vocab + fused CE + levanter intra-doc mask.",
    )
