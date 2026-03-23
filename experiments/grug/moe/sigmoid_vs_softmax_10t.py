# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Great gate ablation: sigmoid vs softmax gating for MoE router.

Runs two matched training jobs that differ only in the gating function used to
compute combine weights after top-k expert selection. Part of the great gate
sweep for the 10T TPU recipe (#4014, parent #3469).

Sigmoid gating applies per-expert sigmoid then normalizes to sum to 1.
Softmax gating applies softmax over selected experts (the default).
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
from experiments.grug.moe.model import GatingFunction, GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

# Shared model architecture for the comparison.  Matches the current template
# defaults so that results slot into the broader sweep without confounds.
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


def _make_step(gating: GatingFunction) -> ExecutorStep:
    tag = gating.value
    run_id = _resolve_run_id(f"grug-moe-gate-{tag}")
    model = dataclasses.replace(_BASE_MODEL, gating_fn=gating)
    return ExecutorStep(
        name=f"grug/moe-gate-{tag}",
        fn=run_grug_moe,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(2_000),
            batch_size=versioned(512),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin",
                tags=["grug", "moe", "great-gate", f"gate-{tag}"],
                group="grug-moe-sigmoid-vs-softmax",
                name=None,
            ),
            optimizer=versioned(_OPTIMIZER),
            grug_trainer=versioned(_TRAINER),
            eval=versioned(_EVAL),
        ),
    )


softmax_step = _make_step(GatingFunction.SOFTMAX)
sigmoid_step = _make_step(GatingFunction.SIGMOID)


if __name__ == "__main__":
    executor_main(
        steps=[softmax_step, sigmoid_step],
        description="Great gate ablation: sigmoid vs softmax MoE gating (#4037).",
    )
