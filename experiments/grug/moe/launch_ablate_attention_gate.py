# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ablation: attention gate in the MoE grug model at ~1e19 FLOPs.

Runs two matched configurations:
  - baseline (no attention gate)
  - headwise attention gate

See https://github.com/marin-community/marin/issues/4020
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig

from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    GrugTrainerConfig,
    run_grug_moe,
)
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

_BASE_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=768,
    intermediate_dim=2048,
    shared_expert_intermediate_dim=2048,
    num_experts=8,
    num_experts_per_token=2,
    num_layers=12,
    num_heads=12,
    num_kv_heads=12,
    max_seq_len=4096,
)

_GATED_MODEL = dataclasses.replace(_BASE_MODEL, attention_gate="headwise")

_OPTIMIZER = AdamConfig(
    learning_rate=3e-3,
    weight_decay=0.1,
    lr_schedule="cosine",
    decay=0.2,
    min_lr_ratio=0.1,
    warmup=500,
)

_TRAINER = GrugTrainerConfig(
    z_loss_weight=1e-4,
    ema_beta=None,
    log_every=1,
)

_EVAL = GrugEvalConfig(
    eval_batch_size=512,
    steps_per_eval=500,
    max_eval_batches=8,
    eval_current=True,
    eval_ema=False,
)

_WANDB_TAGS = ["grug", "moe", "good-10t", "ablation", "attention-gate"]
_STEPS = 2_130
_BATCH_SIZE = 512
_RESOURCES = ResourceConfig.with_tpu("v5p-8")


def _make_launch_config(
    model: GrugModelConfig,
    run_id: str,
    wandb_group: str,
    extra_tags: list[str] | None = None,
) -> GrugMoeLaunchConfig:
    tags = list(_WANDB_TAGS) + (extra_tags or [])
    return GrugMoeLaunchConfig(
        model=versioned(model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=run_id,
        resources=versioned(_RESOURCES),
        steps=versioned(_STEPS),
        batch_size=versioned(_BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="marin",
            tags=tags,
            group=wandb_group,
            name=None,
        ),
        optimizer=versioned(_OPTIMIZER),
        grug_trainer=versioned(_TRAINER),
        eval=versioned(_EVAL),
    )


ablate_attn_gate_baseline = ExecutorStep(
    name="grug/ablate-attn-gate-baseline",
    fn=run_grug_moe,
    config=_make_launch_config(
        model=_BASE_MODEL,
        run_id="ablate-attn-gate-baseline",
        wandb_group="ablate-attn-gate",
        extra_tags=["baseline"],
    ),
)

ablate_attn_gate_headwise = ExecutorStep(
    name="grug/ablate-attn-gate-headwise",
    fn=run_grug_moe,
    config=_make_launch_config(
        model=_GATED_MODEL,
        run_id="ablate-attn-gate-headwise",
        wandb_group="ablate-attn-gate",
        extra_tags=["headwise"],
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[ablate_attn_gate_baseline, ablate_attn_gate_headwise],
        description="Ablation: attention gate in MoE grug at ~1e19 FLOPs (issue #4020).",
    )
