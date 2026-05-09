# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon walk p17: p16 minus the two AdamH-compensator features.

Drops ``use_attn_gate=False`` and ``use_gated_norm=False`` on top of p16.
Everything else (nemotron + llama3 + fused CE + levanter intra-doc mask +
full MoE + compute-optimal schedule) carries over.

Hypothesis: ``attn_gate`` and ``gated_norm`` were added at p3/p4 to give
AdamH something to scale into (it preserves Frobenius norm, so the
attention output and pre-residual hidden need an extra learnable
magnitude knob). On a fused-CE / nemotron / MoE setup the gates may be
redundant, particularly for the muon walk where the optimizer already
ships per-matrix scale via Newton-Schulz. The walk results so far
support this — gated_norm hurt muon at p4 (+0.014) and only slightly
helped adamh later. Worth checking whether removing them at the new
compute-optimal scale flips the sign for both walks.
"""

import dataclasses

from fray.cluster import ResourceConfig
from jax.sharding import PartitionSpec as P
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.nano.launch import (
    NanoAdamWMuonConfig,
    NanoLaunchConfig,
    _resolve_run_id,
    run_nano_trial,
)
from experiments.grug.nano.launch_muon_tuned_walk_p16 import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION, P16_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P17_TRAIN_STEPS = 10343
P17_BATCH_SIZE = 64
NANO_MUON_TUNED_LR = 0.035
NANO_MUON_TUNED_WD = 0.025

P17_MODEL = dataclasses.replace(
    P16_MODEL,
    use_attn_gate=False,
    use_gated_norm=False,
)

P17_OPTIMIZER = NanoAdamWMuonConfig(
    muon_lr=NANO_MUON_TUNED_LR,
    muon_weight_decay=NANO_MUON_TUNED_WD,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-tuned-p17")


nano_muon_tuned_p17_trial = ExecutorStep(
    name="grug/nano-muon-tuned-p17-trial",
    fn=run_nano_trial,
    config=NanoLaunchConfig(
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
            tags=["grug", "nano", "muon", "nemotron", "tuned", "p17", "moe", "fused-ce", "no-gates"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P17_OPTIMIZER),
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
                eval_batch_size=P17_BATCH_SIZE,
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
        steps=[nano_muon_tuned_p17_trial],
        description="muon-tuned p17: p16 minus attn_gate and gated_norm.",
    )
