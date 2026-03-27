# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment 4021: ablate shared expert in the good-enough 10T MoE gate.

Run a single ~1e19-FLOP comparison: baseline (with shared expert) vs ablation
(shared_expert_intermediate_dim=0). Both arms use the trial model architecture
and optimizer, differing only in the shared expert.  Each arm's step count is
chosen so that 3 * flops_per_token * batch_size * seq_len * steps ≈ 1e19.
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.launch import (
    GRUG_MOE_TRIAL_MODEL,
    GrugMoeLaunchConfig,
    GrugTrainerConfig,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    _resolve_run_id,
    run_grug_moe,
)
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig

FLOP_BUDGET = 1e19
BATCH_SIZE = 512
SEQ_LEN = GRUG_MOE_TRIAL_MODEL.max_seq_len


def _steps_for_budget(flops_per_token: float) -> int:
    """Compute training steps so total FLOPs ≈ FLOP_BUDGET."""
    tokens = FLOP_BUDGET / (3 * flops_per_token)
    return max(1, round(tokens / (BATCH_SIZE * SEQ_LEN)))


def _flops_per_token(model: GrugModelConfig) -> float:
    from levanter.utils.flop_utils import lm_flops_per_token

    return lm_flops_per_token(
        hidden_dim=model.hidden_dim,
        intermediate_dim=model.intermediate_dim,
        shared_intermediate_dim=model.shared_expert_intermediate_dim,
        num_layers=model.num_layers,
        num_kv_heads=model.num_kv_heads,
        num_heads=model.num_heads,
        seq_len=model.max_seq_len,
        vocab_size=model.vocab_size,
        glu=True,
        num_experts=model.num_experts,
        num_shared_experts=1 if model.shared_expert_intermediate_dim > 0 else 0,
        num_experts_per_tok=model.num_experts_per_token,
    )


# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

SHARED_MODEL = GRUG_MOE_TRIAL_MODEL  # shared_expert_intermediate_dim=1792

NO_SHARED_MODEL = dataclasses.replace(GRUG_MOE_TRIAL_MODEL, shared_expert_intermediate_dim=0)

SHARED_STEPS = _steps_for_budget(_flops_per_token(SHARED_MODEL))
NO_SHARED_STEPS = _steps_for_budget(_flops_per_token(NO_SHARED_MODEL))


# ---------------------------------------------------------------------------
# Common training knobs (mirrors trial defaults)
# ---------------------------------------------------------------------------

_OPTIMIZER = AdamConfig(
    learning_rate=3e-3,
    weight_decay=0.1,
    lr_schedule="cosine",
    decay=0.2,
    min_lr_ratio=0.1,
    warmup=1000,
)

_GRUG_TRAINER = GrugTrainerConfig(
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


def _wandb(group: str, run_id: str) -> WandbConfig:
    return WandbConfig(
        project="marin",
        tags=["grug", "moe", "exp4021", "shared-expert-ablation"],
        group=group,
        name=None,
    )


# ---------------------------------------------------------------------------
# Executor steps
# ---------------------------------------------------------------------------

_SHARED_RUN_ID = _resolve_run_id("exp4021-shared")
_NO_SHARED_RUN_ID = _resolve_run_id("exp4021-no-shared")

shared_expert_baseline = ExecutorStep(
    name="grug/exp4021-shared",
    fn=run_grug_moe,
    config=GrugMoeLaunchConfig(
        model=versioned(SHARED_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=_SHARED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(SHARED_STEPS),
        batch_size=versioned(BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_wandb("exp4021-ablate-shared-expert", _SHARED_RUN_ID),
        optimizer=versioned(_OPTIMIZER),
        grug_trainer=versioned(_GRUG_TRAINER),
        eval=versioned(_EVAL),
    ),
)

no_shared_expert_ablation = ExecutorStep(
    name="grug/exp4021-no-shared",
    fn=run_grug_moe,
    config=GrugMoeLaunchConfig(
        model=versioned(NO_SHARED_MODEL),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=_NO_SHARED_RUN_ID,
        resources=versioned(ResourceConfig.with_tpu("v5p-8")),
        steps=versioned(NO_SHARED_STEPS),
        batch_size=versioned(BATCH_SIZE),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=_wandb("exp4021-ablate-shared-expert", _NO_SHARED_RUN_ID),
        optimizer=versioned(_OPTIMIZER),
        grug_trainer=versioned(_GRUG_TRAINER),
        eval=versioned(_EVAL),
    ),
)


if __name__ == "__main__":
    executor_main(
        steps=[shared_expert_baseline, no_shared_expert_ablation],
        description="Exp 4021: ablate shared expert at ~1e19 FLOPs. Fixes #4021.",
    )
