# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon walk p18: shared expert as ReLU² @ 1.5·D (instead of SwiGLU @ 1·D).

On top of p17 (p16 minus attn_gate / gated_norm), swap the shared expert's
shape:

    p17 shared:  SwiGLU @ 1·D   = 768·{gate,up,down}  → 3 mats x 768x768
    p18 shared:  ReLU²  @ 1.5·D = 1152·{fc,proj}      → 2 mats x 768x1152

These are isoflop and have the same parameter count (1.77 M / layer):

    SwiGLU  fwd FLOPs/tok = 6·D·F = 6 · 768 · 768  ≈ 3.54 M
    ReLU²   fwd FLOPs/tok = 4·D·F = 4 · 768 · 1152 ≈ 3.54 M

Routed experts stay SwiGLU @ D/2 (= 384) — that path is hardcoded in
`levanter.grug.grug_moe.moe_mlp`. Only the dense shared expert changes.
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
from experiments.grug.nano.launch_muon_tuned_walk_p16 import NEMOTRON_MIX_WITH_DEFAULT_VALIDATION
from experiments.grug.nano.launch_muon_tuned_walk_p17 import P17_MODEL
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P18_TRAIN_STEPS = 10343
P18_BATCH_SIZE = 64
NANO_MUON_TUNED_LR = 0.035
NANO_MUON_TUNED_WD = 0.025

P18_MODEL = dataclasses.replace(
    P17_MODEL,
    # Shared expert: ReLU² activation at 1.5x hidden_dim intermediate.
    shared_expert_intermediate_dim=int(1.5 * P17_MODEL.hidden_dim),
    shared_expert_mlp_type="relu_squared",
)

P18_OPTIMIZER = NanoAdamWMuonConfig(
    muon_lr=NANO_MUON_TUNED_LR,
    muon_weight_decay=NANO_MUON_TUNED_WD,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-tuned-p18")


nano_muon_tuned_p18_trial = ExecutorStep(
    name="grug/nano-muon-tuned-p18-trial",
    fn=run_nano_trial,
    config=NanoLaunchConfig(
        model=versioned(P18_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P18_TRAIN_STEPS),
        batch_size=versioned(P18_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "muon", "nemotron", "tuned", "p18", "moe", "shared-relu2"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P18_OPTIMIZER),
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
                eval_batch_size=P18_BATCH_SIZE,
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
        steps=[nano_muon_tuned_p18_trial],
        description="muon-tuned p18: shared expert -> ReLU² @ 1.5·D (was SwiGLU @ 1·D).",
    )
