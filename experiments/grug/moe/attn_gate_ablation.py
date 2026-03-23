# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ablation: attention gate on vs off for grug MoE.

Runs two matched configurations differing only in `use_attention_gate` to
measure the effect of per-head attention gating on the MoE architecture.

Related issue: https://github.com/marin-community/marin/issues/4038
"""

# nodryrun

import dataclasses
import os

from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.launch import (
    GrugMoeLaunchConfig,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    run_grug_moe,
)
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

# Base model config matching the trial recipe but with attention gate toggled.
_BASE_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=512,
    intermediate_dim=1792,
    shared_expert_intermediate_dim=1792,
    num_experts=8,
    num_experts_per_token=2,
    num_layers=6,
    num_heads=8,
    num_kv_heads=8,
    max_seq_len=4096,
    head_dim=None,
)

_OPTIMIZER = AdamConfig(
    learning_rate=3e-3,
    weight_decay=0.1,
    lr_schedule="cosine",
    decay=0.2,
    min_lr_ratio=0.1,
    warmup=1000,
)

_TRAINER = GrugTrainerConfig(
    z_loss_weight=1e-4,
    ema_beta=None,
    log_every=1,
)

_EVAL = GrugEvalConfig(
    eval_batch_size=512,
    steps_per_eval=1000,
    max_eval_batches=8,
    eval_current=True,
    eval_ema=False,
)

_STEPS = 2_000
_BATCH_SIZE = 512
_SEED = 0
_MP = "params=float32,compute=bfloat16,output=bfloat16"
_RESOURCES = ResourceConfig.with_tpu("v5p-8")


def _resolve_run_id(default_run_id: str) -> str:
    run_id = os.environ.get("GRUG_RUN_ID", default_run_id)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _make_step(*, use_attention_gate: bool) -> ExecutorStep:
    tag = "gate-on" if use_attention_gate else "gate-off"
    run_id = _resolve_run_id(f"attn-gate-ablation-{tag}")
    model = dataclasses.replace(_BASE_MODEL, use_attention_gate=use_attention_gate)
    return ExecutorStep(
        name=f"grug/attn-gate-ablation-{tag}",
        fn=run_grug_moe,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(_RESOURCES),
            steps=versioned(_STEPS),
            batch_size=versioned(_BATCH_SIZE),
            seed=versioned(_SEED),
            mp=versioned(_MP),
            tracker=WandbConfig(
                project="marin",
                tags=["grug", "moe", "attn-gate-ablation"],
                group="attn-gate-ablation",
                name=None,
            ),
            optimizer=versioned(_OPTIMIZER),
            grug_trainer=versioned(_TRAINER),
            eval=versioned(_EVAL),
        ),
    )


attn_gate_on = _make_step(use_attention_gate=True)
attn_gate_off = _make_step(use_attention_gate=False)


if __name__ == "__main__":
    executor_main(
        steps=[attn_gate_on, attn_gate_off],
        description="Ablation: attention gate on vs off for grug MoE (issue #4038).",
    )
