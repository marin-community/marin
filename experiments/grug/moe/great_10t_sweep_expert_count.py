# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Great 10T ablation: sweep num_experts across isoflop budgets.

Generates an isoflop grid varying num_experts (128, 256, 512) at three FLOP
budgets (1e18, 3e18, 1e19) with five hidden dims per budget.  Architecture
follows the E=128, K=4 MoE recipe from iteration-02 scaling work; the sweep
varies only E while holding per-expert intermediate dim, shared expert, and K
constant.

See: https://github.com/marin-community/marin/issues/4048
Parent sweep: https://github.com/marin-community/marin/issues/3469
Gate: https://github.com/marin-community/marin/issues/4014
"""

import math

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.utils.flop_utils import lm_flops_per_token
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe,
)
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

# ---------------------------------------------------------------------------
# Sweep axes
# ---------------------------------------------------------------------------
FLOP_BUDGETS: tuple[float, ...] = (1e18, 3e18, 1e19)
HIDDEN_DIMS: tuple[int, ...] = (512, 768, 1024, 1536, 2048)
EXPERT_COUNTS: tuple[int, ...] = (128, 256, 512)

# ---------------------------------------------------------------------------
# Architecture constants (K=4 recipe, E varies)
# ---------------------------------------------------------------------------
VOCAB_SIZE = 128_256
NUM_EXPERTS_PER_TOKEN = 4
SEQ_LEN = 4096
TARGET_STEPS_LOG2 = 14  # ~16384 steps per run

# Scaling heuristics (from iteration-02 MoE recipe).
BASE_HIDDEN_LAYER_RATIO = 64
LAYER_SCALING_FACTOR = 4.0
LAYER_FORMULA_OFFSET = 9
LR_CONSTANT = 0.33
MAX_LR = 0.01
BETA2_BASE = 0.98
BETA2_BATCH_DIVISOR = 128


def _compute_num_layers(hidden_dim: int) -> int:
    hs_pow = math.log2(hidden_dim)
    return round(hidden_dim / (BASE_HIDDEN_LAYER_RATIO + LAYER_SCALING_FACTOR * hs_pow - LAYER_FORMULA_OFFSET))


def _round_up_pow2(x: float) -> int:
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def _flops_per_token_for_config(cfg: GrugModelConfig) -> float:
    """Compute forward-pass FLOPs per token for a MoE model."""
    return lm_flops_per_token(
        hidden_dim=cfg.hidden_dim,
        intermediate_dim=cfg.intermediate_dim,
        shared_intermediate_dim=cfg.shared_expert_intermediate_dim,
        num_layers=cfg.num_layers,
        num_kv_heads=cfg.num_kv_heads,
        num_heads=cfg.num_heads,
        seq_len=cfg.max_seq_len,
        vocab_size=cfg.vocab_size,
        glu=True,
        num_experts=cfg.num_experts,
        num_shared_experts=1 if cfg.shared_expert_intermediate_dim > 0 else 0,
        num_experts_per_tok=cfg.num_experts_per_token,
    )


def _build_model_config(hidden_dim: int, num_experts: int) -> GrugModelConfig:
    num_layers = _compute_num_layers(hidden_dim)
    intermediate_dim = hidden_dim // 2
    shared_expert_dim = hidden_dim
    num_heads = max(1, hidden_dim // 128)
    return GrugModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        shared_expert_intermediate_dim=shared_expert_dim,
        num_experts=num_experts,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        max_seq_len=SEQ_LEN,
    )


def _build_sweep_step(
    budget: float,
    hidden_dim: int,
    num_experts: int,
) -> ExecutorStep:
    model = _build_model_config(hidden_dim, num_experts)
    fpt = _flops_per_token_for_config(model)
    tokens = budget / (3 * fpt)
    target_steps = 2**TARGET_STEPS_LOG2

    batch_exact = tokens / (target_steps * SEQ_LEN)
    effective_bs = _round_up_pow2(batch_exact)
    effective_bs = max(8, effective_bs)

    lr = min(MAX_LR, (LR_CONSTANT * math.sqrt(effective_bs)) / hidden_dim)
    beta2 = max(0.95, BETA2_BASE ** (effective_bs / BETA2_BATCH_DIVISOR))
    steps = max(1, round(tokens / (effective_bs * SEQ_LEN)))

    budget_tag = f"{budget:.0e}"
    expert_tag = f"e{num_experts}"
    run_id = f"great-10t-sweep-E-{budget_tag}-d{hidden_dim}-{expert_tag}"

    return ExecutorStep(
        name=f"grug/great-10t-sweep-E/{budget_tag}/d{hidden_dim}/{expert_tag}",
        fn=run_grug_moe,
        config=GrugMoeLaunchConfig(
            model=versioned(model),
            data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(steps),
            batch_size=versioned(effective_bs),
            seed=versioned(0),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin",
                tags=["grug", "moe", "great-10t", "sweep-E", budget_tag, expert_tag],
                group="great-10t-sweep-E",
                name=None,
            ),
            optimizer=versioned(
                AdamConfig(
                    learning_rate=lr,
                    weight_decay=0.1,
                    lr_schedule="cosine",
                    decay=0.2,
                    min_lr_ratio=0.1,
                    warmup=0.1,
                    beta2=beta2,
                )
            ),
            grug_trainer=versioned(
                GrugTrainerConfig(
                    z_loss_weight=1e-4,
                    ema_beta=None,
                    log_every=1,
                )
            ),
            eval=versioned(
                GrugEvalConfig(
                    eval_batch_size=min(512, effective_bs),
                    steps_per_eval=1000,
                    max_eval_batches=8,
                    eval_current=True,
                    eval_ema=False,
                )
            ),
        ),
    )


def build_sweep_steps() -> list[ExecutorStep]:
    """Generate all ExecutorSteps for the great 10T expert count sweep."""
    steps: list[ExecutorStep] = []
    for budget in FLOP_BUDGETS:
        for hidden_dim in HIDDEN_DIMS:
            for num_experts in EXPERT_COUNTS:
                steps.append(_build_sweep_step(budget, hidden_dim, num_experts))
    return steps


ALL_STEPS = build_sweep_steps()


if __name__ == "__main__":
    executor_main(
        steps=ALL_STEPS,
        description="Great 10T ablation: sweep num_experts across isoflop budgets (issue #4048).",
    )
