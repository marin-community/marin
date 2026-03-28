# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep expert count E in {128, 256, 512} for the 10T gate MoE recipe.

This experiment varies only the number of routed experts while holding the
per-expert intermediate dimension, shared expert, K (experts per token), and
all other hyperparameters fixed.  The goal is to determine whether expert
count is a significant remaining lever for the baseline recipe.

Ref: https://github.com/marin-community/marin/issues/4030
Parent sweep: https://github.com/marin-community/marin/issues/3469
Gate: https://github.com/marin-community/marin/issues/4013
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    _resolve_run_id,
    run_grug_moe,
)
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

# ---------------------------------------------------------------------------
# Base model config for the 10T gate recipe.
#
# This mirrors the trial model dimensions but can be replaced with the final
# gate config once #4013 locks in the architecture.  Only `num_experts` is
# swept; everything else stays constant across arms.
# ---------------------------------------------------------------------------
BASE_MODEL = GrugModelConfig(
    vocab_size=128_256,
    hidden_dim=512,
    intermediate_dim=1792,
    shared_expert_intermediate_dim=1792,
    num_experts=128,  # overridden per arm
    num_experts_per_token=2,
    num_layers=6,
    num_heads=8,
    num_kv_heads=8,
    max_seq_len=4096,
    head_dim=None,
)

EXPERT_COUNTS = (128, 256, 512)

OPTIMIZER = AdamConfig(
    learning_rate=3e-3,
    weight_decay=0.1,
    lr_schedule="cosine",
    decay=0.2,
    min_lr_ratio=0.1,
    warmup=1000,
)

TRAINER = GrugTrainerConfig(
    z_loss_weight=1e-4,
    ema_beta=None,
    log_every=1,
)

EVAL = GrugEvalConfig(
    eval_batch_size=512,
    steps_per_eval=1000,
    max_eval_batches=8,
    eval_current=True,
    eval_ema=False,
)

STEPS = 2_000
BATCH_SIZE = 512
SEED = 0
MP_POLICY = "params=float32,compute=bfloat16,output=bfloat16"


def _build_step(num_experts: int) -> ExecutorStep:
    """Build an ExecutorStep for a single expert-count arm."""
    tag = f"e{num_experts}"
    run_id = _resolve_run_id(f"grug-moe-sweep-E-{tag}")
    model = dataclasses.replace(BASE_MODEL, num_experts=num_experts)

    config = GrugMoeLaunchConfig(
        model=versioned(model),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=run_id,
        resources=versioned(
            # Start with v5p-8 (matches the trial template).  For large E
            # the expert mesh axis or slice count may need adjustment.
            ResourceConfig.with_tpu("v5p-8"),
        ),
        steps=versioned(STEPS),
        batch_size=versioned(BATCH_SIZE),
        seed=versioned(SEED),
        mp=versioned(MP_POLICY),
        tracker=WandbConfig(
            project="marin",
            tags=["grug", "moe", "sweep-E", tag],
            group="grug-moe-sweep-E",
            name=None,
        ),
        optimizer=versioned(OPTIMIZER),
        grug_trainer=versioned(TRAINER),
        eval=versioned(EVAL),
    )

    return ExecutorStep(
        name=f"grug/moe-sweep-E-{tag}",
        fn=run_grug_moe,
        config=config,
    )


sweep_steps = [_build_step(e) for e in EXPERT_COUNTS]


if __name__ == "__main__":
    executor_main(
        steps=sweep_steps,
        description="Sweep expert count E in {128, 256, 512} for the 10T gate MoE recipe (#4030).",
    )
