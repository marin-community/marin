# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon walk p15: p14 (full MoE at moe d768 compute-optimal) + intra-doc
attention masking.

Same architecture and schedule as p14 (b=64, seq=4096, steps=10343, full MoE).
The only change: set ``intra_doc_bos_id=50256`` on the model, which makes
``Transformer.__call__`` derive ``segment_ids`` from the `<|endoftext|>`
markers in the input tokens and attach them to the attention mask.

We do this in the model rather than via
``LmDataConfig.block_cross_document_attention=True`` because the levanter
data path was a no-op for our setup: ``MarinTokenizer("gpt2")`` exposes
neither ``bos_token_id`` nor ``eos_token_id`` (both ``None``), so
``GrugLmExample.causal`` silently skipped segment-id derivation. Plus the
levanter logic puts the marker in the *previous* segment, wrong for a
cache that prepends the marker at the start of each doc (we want a doc's
tokens to be able to attend back to their own leading marker).

Rationale: prior to p15, the ~12% of training tokens that follow a
`<|endoftext|>` were learning a "what comes after the end of an unrelated
prior document?" prediction problem. With intra-doc masking those queries
only attend back to tokens within the new document, so each new doc gets
a clean context (its leading marker plus everything after it within the
same doc).
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
from experiments.grug.nano.launch_muon_tuned_walk_p14 import P14_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P15_TRAIN_STEPS = 10343
P15_BATCH_SIZE = 64
NANO_MUON_TUNED_LR = 0.035
NANO_MUON_TUNED_WD = 0.025

# Architecture is identical to p14 except for `intra_doc_bos_id` — the
# model itself derives segment ids from the gpt2 EOT marker (50256) and
# attaches them to the attention mask. See `NanoModelConfig.intra_doc_bos_id`.
P15_MODEL = dataclasses.replace(P14_MODEL, intra_doc_bos_id=50256)

P15_OPTIMIZER = NanoAdamWMuonConfig(
    muon_lr=NANO_MUON_TUNED_LR,
    muon_weight_decay=NANO_MUON_TUNED_WD,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-tuned-p15-v2")


nano_muon_tuned_p15_trial = ExecutorStep(
    name="grug/nano-muon-tuned-p15-trial",
    fn=run_nano_trial,
    config=NanoLaunchConfig(
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
            tags=["grug", "nano", "muon", "fineweb-gpt2", "tuned", "p15", "moe", "intradoc-mask"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P15_OPTIMIZER),
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
        steps=[nano_muon_tuned_p15_trial],
        description="muon-tuned p15: p14 + intra-doc attention masking (CausalLmDataset path).",
    )
