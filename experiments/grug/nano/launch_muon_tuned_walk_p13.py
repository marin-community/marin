# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Muon walk p13: p12 (halved batch / doubled steps) + full MoE.

Final feature added on top of p12's schedule change: replace the dense MLP
with a routed MoE block plus a parallel dense shared expert.

  - 4-of-64 routed experts, each at ``hidden_dim // 2`` (= 384 here).
    Stored as separate ``w_gate`` / ``w_up`` / ``w_down`` (E, D, I) tensors;
    gate and up are concatenated on the forward pass before
    ``levanter.grug.grug_moe.moe_mlp``.
  - One dense shared expert at ``hidden_dim`` (= 768 here), running in
    parallel with the routed experts.
  - QB load balancing: the trainer rewrites router biases each step from the
    previous step's per-layer beta (see ``train.py:_apply_qb_betas``).
  - Optimizer routing: the router weight goes to AdamW; the 3-D expert
    tensors go to Muon (vmaps over leading expert axis,
    Newton-Schulz on the trailing two dims).
  - Mesh: adds an ``expert`` axis (size 1) so ``MoEMLP``'s expert-sharded
    init / shard_map / pmean see a real axis.
  - Schedule (inherited from p12): batch_size=64, num_train_steps=6700;
    total tokens = 1.756B (same as p11).
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
from experiments.grug.nano.launch_muon_tuned_walk_p12 import P12_BATCH_SIZE, P12_MODEL, P12_TRAIN_STEPS
from experiments.grug.nano.train import GrugEvalConfig, GrugTrainerConfig

P13_TRAIN_STEPS = P12_TRAIN_STEPS
P13_BATCH_SIZE = P12_BATCH_SIZE
NANO_MUON_TUNED_LR = 0.035
NANO_MUON_TUNED_WD = 0.025

P13_MODEL = dataclasses.replace(
    P12_MODEL,
    use_moe=True,
    num_experts=64,
    num_experts_per_token=4,
    expert_intermediate_dim=P12_MODEL.hidden_dim // 2,
    shared_expert_intermediate_dim=P12_MODEL.hidden_dim,
    separate_gate_up=True,
    router_z_loss_coef=0.001,
)

P13_OPTIMIZER = NanoAdamWMuonConfig(
    muon_lr=NANO_MUON_TUNED_LR,
    muon_weight_decay=NANO_MUON_TUNED_WD,
)


RESOLVED_RUN_ID = _resolve_run_id("may7-nano-muon-tuned-p13")


nano_muon_tuned_p13_trial = ExecutorStep(
    name="grug/nano-muon-tuned-p13-trial",
    fn=run_nano_trial,
    config=NanoLaunchConfig(
        model=versioned(P13_MODEL),
        data=_fineweb_gpt2_data(),
        output_path=this_output_path(),
        run_id=RESOLVED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(P13_TRAIN_STEPS),
        batch_size=versioned(P13_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "nano", "muon", "fineweb-gpt2", "tuned", "p13", "moe"],
            group="nano-trial",
            name=None,
            replicate_path=this_output_path(),
        ),
        optimizer=versioned(P13_OPTIMIZER),
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
                eval_batch_size=P13_BATCH_SIZE,
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
        steps=[nano_muon_tuned_p13_trial],
        description="muon-tuned p13: full MoE (64x384 routed + shared 768), QB routing, b=64 steps=6700.",
    )
