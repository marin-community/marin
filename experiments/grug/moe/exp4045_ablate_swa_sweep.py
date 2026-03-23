# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Experiment 4045: multi-budget sliding-window attention ablation for the great 10T gate.

Runs full-attention vs sliding-window (window=4096) at multiple FLOP budgets
to build an isoflop scaling curve. Each arm is compute-matched:
3 * flops_per_token * batch_size * seq_len * steps ~ budget.

Tracking issue: https://github.com/marin-community/marin/issues/4045
Parent gate: https://github.com/marin-community/marin/issues/4014
"""

import dataclasses

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.utils.flop_utils import lm_flops_per_token
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.launch import (
    GrugMoeLaunchConfig,
    GrugTrainerConfig,
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    _resolve_run_id,
    run_grug_moe,
)
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig

# ---------------------------------------------------------------------------
# FLOP budgets — sweep from small to moderate scale
# ---------------------------------------------------------------------------

FLOP_BUDGETS: tuple[float, ...] = (3e18, 9e18, 1.8e19, 3e19, 9e19)

BATCH_SIZE = 512
SEQ_LEN = 4096
VOCAB_SIZE = 128_256
NUM_EXPERTS = 8
NUM_EXPERTS_PER_TOKEN = 2
SLIDING_WINDOW_SIZE = 4096

# ---------------------------------------------------------------------------
# Model configs per budget — wider + deeper at higher budgets
# ---------------------------------------------------------------------------

# Each entry: (hidden_dim, intermediate_dim, shared_expert_intermediate_dim,
#              num_layers, num_heads, num_kv_heads)
# intermediate_dim ~ 3.5 * hidden_dim (SwiGLU convention).

_MODEL_SPECS: dict[float, tuple[int, int, int, int, int, int]] = {
    3e18: (384, 1344, 1344, 6, 6, 6),
    9e18: (512, 1792, 1792, 6, 8, 8),
    1.8e19: (512, 1792, 1792, 12, 8, 8),
    3e19: (768, 2688, 2688, 12, 12, 12),
    9e19: (1024, 3584, 3584, 16, 16, 16),
}


def _make_model(budget: float, *, swa: bool) -> GrugModelConfig:
    hidden, inter, shared_inter, layers, heads, kv_heads = _MODEL_SPECS[budget]
    return GrugModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=hidden,
        intermediate_dim=inter,
        shared_expert_intermediate_dim=shared_inter,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        num_layers=layers,
        num_heads=heads,
        num_kv_heads=kv_heads,
        max_seq_len=SEQ_LEN,
        head_dim=None,
        sliding_window=SLIDING_WINDOW_SIZE if swa else None,
    )


def model_flops_per_token(model: GrugModelConfig) -> float:
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


def steps_for_budget(fpt: float, budget: float) -> int:
    """Compute training steps so total FLOPs ~ budget."""
    tokens = budget / (3 * fpt)
    return max(1, round(tokens / (BATCH_SIZE * SEQ_LEN)))


# ---------------------------------------------------------------------------
# Common training knobs
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


def _wandb(group: str) -> WandbConfig:
    return WandbConfig(
        project="marin",
        tags=["grug", "moe", "exp4045", "swa-ablation", "great-gate"],
        group=group,
        name=None,
    )


# ---------------------------------------------------------------------------
# Build executor steps for every (budget, swa/full) pair
# ---------------------------------------------------------------------------


def _build_steps() -> list[ExecutorStep]:
    steps: list[ExecutorStep] = []
    for budget in FLOP_BUDGETS:
        budget_tag = f"{budget:.0e}"
        for swa in (False, True):
            arm = "swa-4096" if swa else "full-attn"
            model = _make_model(budget, swa=swa)
            fpt = model_flops_per_token(model)
            num_steps = steps_for_budget(fpt, budget)
            run_id = _resolve_run_id(f"exp4045-{arm}-{budget_tag}")
            step = ExecutorStep(
                name=f"grug/exp4045-{arm}-{budget_tag}",
                fn=run_grug_moe,
                config=GrugMoeLaunchConfig(
                    model=versioned(model),
                    data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                    output_path=this_output_path(),
                    run_id=run_id,
                    resources=versioned(ResourceConfig.with_tpu("v5p-8")),
                    steps=versioned(num_steps),
                    batch_size=versioned(BATCH_SIZE),
                    seed=versioned(0),
                    mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                    tracker=_wandb(f"exp4045-swa-ablation-{budget_tag}"),
                    optimizer=versioned(_OPTIMIZER),
                    grug_trainer=versioned(_GRUG_TRAINER),
                    eval=versioned(_EVAL),
                ),
            )
            steps.append(step)
    return steps


ALL_STEPS = _build_steps()


if __name__ == "__main__":
    executor_main(
        steps=ALL_STEPS,
        description="Exp 4045: multi-budget SWA ablation for the great 10T gate. Fixes #4045.",
    )
