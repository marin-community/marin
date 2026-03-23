# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-scale isoflop comparison of AdamH vs Adam on the grug MoE architecture.

Launches paired Adam / AdamH runs at four FLOP budgets (3e18, 1e19, 3e19, 1e20)
on appropriately-sized grug MoE models (E=8, K=2, shared expert). Each budget
gets a model sized so that token count stays within ~20-40x parameters (roughly
Chinchilla-optimal for MoE). The only variable within each pair is the optimizer.

This produces 8 runs total, enough to check whether AdamH vs Adam trends hold
across scales before locking in the optimizer for the 10T TPU path.

Part of #4042 / #4014.
"""

import math
import os
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig, GrugAdamHConfig, OptimizerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.utils.flop_utils import lm_flops_per_token

from experiments.defaults import default_validation_sets
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.pretraining_datasets import nemotron_mix_block_shuffle
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

# ---------- constants ----------
SEQ_LEN = 4096
VOCAB_SIZE = 128_256
NUM_EXPERTS = 8
NUM_EXPERTS_PER_TOKEN = 2
HEAD_DIM = 128

NEMOTRON_MIX = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)

GRUG_TRAINER = GrugTrainerConfig(
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


@dataclass(frozen=True)
class ScalePoint:
    """One point in the isoflop suite."""

    budget: float
    hidden_dim: int
    num_layers: int
    batch_size: int


def _num_layers_for_hidden(hidden_dim: int) -> int:
    """Depth-to-width heuristic: ~hidden/64, clamped to reasonable range."""
    raw = hidden_dim / 64
    return max(6, min(48, round(raw)))


def _flops_per_token(hidden_dim: int, intermediate_dim: int, num_layers: int, num_heads: int) -> float:
    return lm_flops_per_token(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        num_layers=num_layers,
        num_kv_heads=num_heads,
        num_heads=num_heads,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        glu=True,
        num_experts=NUM_EXPERTS,
        num_shared_experts=1,
        num_experts_per_tok=NUM_EXPERTS_PER_TOKEN,
        shared_intermediate_dim=intermediate_dim,
    )


def _pick_batch_size(tokens: float, target_steps: int = 2**16) -> int:
    """Pick batch size (power of 2) to hit ~target_steps of training."""
    raw = tokens / (target_steps * SEQ_LEN)
    bs = 2 ** round(math.log2(max(8, raw)))
    return max(8, min(512, bs))


def _find_hidden_dim_for_budget(budget: float, step_size: int = 128) -> int:
    """Binary-ish search for the hidden_dim that makes the model ~Chinchilla-optimal.

    We want tokens/params ~ 20-40x. Search over hidden_dim in steps of step_size.
    """
    best_dim = 512
    best_score = float("inf")
    for dim in range(384, 4096 + 1, step_size):
        n_layers = _num_layers_for_hidden(dim)
        n_heads = dim // HEAD_DIM
        if n_heads < 1 or dim % HEAD_DIM != 0:
            continue
        intermediate = dim * 3
        fpt = _flops_per_token(dim, intermediate, n_layers, n_heads)
        tokens = budget / (3 * fpt)
        # Rough param count for MoE: embedding + layers*(attn + K*expert_mlp + shared_mlp)
        # Simplified: use a proxy based on dense equivalent
        attn_params = 4 * dim * dim * n_layers
        expert_params = 3 * dim * intermediate * NUM_EXPERTS * n_layers  # GLU: 3 matrices
        shared_params = 3 * dim * intermediate * n_layers
        embed_params = 2 * VOCAB_SIZE * dim
        total_params = attn_params + expert_params + shared_params + embed_params
        # Active params per token (what matters for Chinchilla)
        active_params = attn_params + 3 * dim * intermediate * NUM_EXPERTS_PER_TOKEN * n_layers + shared_params + embed_params
        ratio = tokens / active_params
        # Target ratio ~20-40x, aim for ~25
        score = abs(math.log(ratio / 25))
        if score < best_score:
            best_score = score
            best_dim = dim
    return best_dim


def make_scale_point(budget: float) -> ScalePoint:
    hidden_dim = _find_hidden_dim_for_budget(budget)
    num_layers = _num_layers_for_hidden(hidden_dim)
    num_heads = hidden_dim // HEAD_DIM
    intermediate_dim = hidden_dim * 3
    fpt = _flops_per_token(hidden_dim, intermediate_dim, num_layers, num_heads)
    tokens = budget / (3 * fpt)
    batch_size = _pick_batch_size(tokens)
    return ScalePoint(budget=budget, hidden_dim=hidden_dim, num_layers=num_layers, batch_size=batch_size)


def make_model(sp: ScalePoint) -> GrugModelConfig:
    num_heads = sp.hidden_dim // HEAD_DIM
    return GrugModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=sp.hidden_dim,
        intermediate_dim=sp.hidden_dim * 3,
        shared_expert_intermediate_dim=sp.hidden_dim * 3,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        num_layers=sp.num_layers,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        max_seq_len=SEQ_LEN,
    )


def compute_train_steps(sp: ScalePoint) -> int:
    num_heads = sp.hidden_dim // HEAD_DIM
    fpt = _flops_per_token(sp.hidden_dim, sp.hidden_dim * 3, sp.num_layers, num_heads)
    tokens = sp.budget / (3 * fpt)
    return round(tokens / (sp.batch_size * SEQ_LEN))


def make_adam_optimizer(sp: ScalePoint) -> AdamConfig:
    """Adam optimizer with LR/beta2 scaled by batch size and model width."""
    effective_bs = sp.batch_size * SEQ_LEN / 4096
    lr = min(0.01, (0.33 * math.sqrt(effective_bs)) / sp.hidden_dim)
    beta2 = max(0.95, 0.98 ** (effective_bs / 128))
    return AdamConfig(
        learning_rate=lr,
        weight_decay=0.1,
        beta1=0.9,
        beta2=beta2,
        epsilon=1e-8,
        lr_schedule="linear",
        decay=0.2,
        min_lr_ratio=0.0,
        warmup=0.1,
        max_grad_norm=1.0,
    )


def make_adamh_optimizer(sp: ScalePoint) -> GrugAdamHConfig:
    """AdamH optimizer: sqrt(lr * wd) for scale-invariant weights, standard lr for Adam params."""
    effective_bs = sp.batch_size * SEQ_LEN / 4096
    adam_lr = min(0.01, (0.33 * math.sqrt(effective_bs)) / sp.hidden_dim)
    beta2 = max(0.95, 0.98 ** (effective_bs / 128))
    adamh_lr = math.sqrt(adam_lr * 0.1)
    return GrugAdamHConfig(
        learning_rate=adamh_lr,
        adam_lr=adam_lr,
        beta1=0.9,
        beta2=beta2,
        epsilon=1e-8,
        lr_schedule="linear",
        decay=0.2,
        min_lr_ratio=0.0,
        warmup=0.1,
        max_grad_norm=0.1,
        weight_decay=0.0,
    )


def _resolve_run_id(base: str) -> str:
    run_id = os.environ.get("GRUG_RUN_ID", base)
    ferry_date = os.environ.get("FERRY_DATE")
    if ferry_date:
        run_id = f"{run_id}-{ferry_date}"
    return run_id


def _make_step(
    sp: ScalePoint, optimizer: OptimizerConfig, opt_name: str, tags: list[str]
) -> ExecutorStep:
    budget_str = f"{sp.budget:.0e}"
    name = f"moe-{opt_name}-{budget_str}-d{sp.hidden_dim}"
    run_id = _resolve_run_id(name)
    train_steps = compute_train_steps(sp)
    return ExecutorStep(
        name=f"grug/{name}",
        fn=run_grug_moe,
        config=GrugMoeLaunchConfig(
            model=versioned(make_model(sp)),
            data=NEMOTRON_MIX,
            output_path=this_output_path(),
            run_id=run_id,
            resources=versioned(ResourceConfig.with_tpu("v5p-8")),
            steps=versioned(train_steps),
            batch_size=versioned(sp.batch_size),
            seed=versioned(42),
            mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
            tracker=WandbConfig(
                project="marin",
                tags=["grug", "moe", "adamh-vs-adam", "isoflop", budget_str, *tags],
                group="moe-adamh-vs-adam-isoflop",
                name=None,
            ),
            optimizer=versioned(optimizer),
            grug_trainer=versioned(GRUG_TRAINER),
            eval=versioned(EVAL),
        ),
    )


# ---------- isoflop suite ----------
BUDGETS = (3e18, 1e19, 3e19, 1e20)
SCALE_POINTS = [make_scale_point(b) for b in BUDGETS]

all_steps: list[ExecutorStep] = []
for sp in SCALE_POINTS:
    adam_opt = make_adam_optimizer(sp)
    adamh_opt = make_adamh_optimizer(sp)
    all_steps.append(_make_step(sp, adam_opt, "adam", ["adam", "baseline"]))
    all_steps.append(_make_step(sp, adamh_opt, "adamh", ["adamh"]))


if __name__ == "__main__":
    executor_main(
        steps=all_steps,
        description=(
            "AdamH vs Adam isoflop suite on grug MoE (E=8, K=2) at 3e18/1e19/3e19/1e20 FLOPs. "
            "Part of #4042."
        ),
    )
