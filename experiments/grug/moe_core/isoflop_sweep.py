# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ISOFlop sweep for grug MoE core architecture.

Sweeps over a grid of FLOP budgets and hidden dims, scaling all other
architecture dimensions from hidden_dim per the moe_core defaults:
- num_heads = hidden_dim // 128
- expert intermediate_dim = hidden_dim // 2
- shared expert intermediate_dim = hidden_dim
- dense intermediate_dim = 3 * hidden_dim
- num_layers = round(hidden_dim / (64 + 4*log2(hidden_dim) - 9))
- 2 leading dense layers, MoE (E=64, K=4) in all subsequent layers
"""

import math

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.utils.flop_utils import lm_flops_per_token
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe_core.model import GrugModelConfig
from experiments.grug.moe_core.train import GrugEvalConfig, GrugTrainerConfig
from experiments.pretraining_datasets import nemotron_mix_block_shuffle

# Re-use the launch config from launch.py for wiring.
from experiments.grug.moe_core.launch import GrugMoeLaunchConfig, run_grug_moe_trial

BUDGETS: tuple[float, ...] = (1e18, 3e18, 1e19)
HIDDEN_DIMS: tuple[int, ...] = (512, 768, 1024, 1536, 2048)
SEQ_LEN: int = 4096
VOCAB_SIZE: int = 128_256

# --- LR scaling from Marin2025Recipe ---
LR_CONSTANT: float = 0.33
MAX_LEARNING_RATE: float = 0.01
MIN_BATCH_SIZE: int = 8
TARGET_STEPS: int = 2**16

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)


def _round_to_power_of_two(x: float) -> int:
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def _compute_num_layers(hidden_dim: int) -> int:
    """Compute number of layers from hidden dim using the depth-width formula from Marin2025Recipe."""
    hs_pow = math.log2(hidden_dim)
    return round(hidden_dim / (64 + (hs_pow * 4.0) - 9))


def _build_model_config(hidden_dim: int) -> GrugModelConfig:
    """Build a GrugModelConfig scaling all dims from hidden_dim."""
    num_heads = hidden_dim // 128
    num_layers = _compute_num_layers(hidden_dim)
    return GrugModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=hidden_dim,
        intermediate_dim=hidden_dim // 2,
        shared_expert_intermediate_dim=hidden_dim,
        num_experts=64,
        num_experts_per_token=4,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        max_seq_len=SEQ_LEN,
        num_dense_layers=2,
        dense_intermediate_dim=3 * hidden_dim,
        bias_update_rate=0.0,
        load_balancing_loss_coef=0.01,
    )


def _compute_flops_per_token(cfg: GrugModelConfig) -> float:
    return lm_flops_per_token(
        hidden_dim=cfg.hidden_dim,
        intermediate_dim=cfg.intermediate_dim,
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


def _compute_tokens_and_batch(budget: float, flops_per_token: float) -> tuple[float, int, int]:
    """Compute tokens, batch_size, and train_steps for a FLOP budget.

    Uses 3x multiplier (forward + backward) and targets ~2^16 steps.
    """
    tokens = budget / (3 * flops_per_token)
    target_steps = 2**14
    batch_exact = tokens / (target_steps * SEQ_LEN)
    batch_size = max(8, _round_to_power_of_two(batch_exact))
    train_steps = max(1, round(tokens / (batch_size * SEQ_LEN)))
    return tokens, batch_size, train_steps


def create_moe_isoflop_steps() -> list[ExecutorStep]:
    """Create ExecutorSteps for the MoE isoflop grid."""
    steps: list[ExecutorStep] = []

    for budget in BUDGETS:
        for hidden_dim in HIDDEN_DIMS:
            model_cfg = _build_model_config(hidden_dim)
            fpt = _compute_flops_per_token(model_cfg)
            _tokens, batch_size, train_steps = _compute_tokens_and_batch(budget, fpt)
            lr = min(0.01, (0.33 * math.sqrt(batch_size)) / hidden_dim)
            beta2 = max(0.95, 0.98 ** (batch_size / 128))

            run_id = f"moe-isoflop-adamw-lbl-{budget:.0e}-d{hidden_dim}"

            config = GrugMoeLaunchConfig(
                model=versioned(model_cfg),
                data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                output_path=this_output_path(),
                run_id=run_id,
                resources=versioned(ResourceConfig.with_tpu("v5p-8")),
                steps=versioned(train_steps),
                batch_size=versioned(batch_size),
                seed=versioned(0),
                mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                tracker=WandbConfig(
                    project="dial_moe",
                    tags=["grug", "moe-core", "isoflop", f"budget={budget:.0e}", f"d={hidden_dim}"],
                    group="moe-core-isoflop",
                    name=run_id,
                ),
                optimizer=versioned(
                    AdamConfig(
                        learning_rate=lr,
                        beta1=0.95,
                        beta2=beta2,
                        epsilon=1e-15,
                        weight_decay=0.1,
                        lr_schedule="linear",
                        decay=0.2,
                        min_lr_ratio=0.0,
                        warmup=0.1,
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
                        eval_batch_size=512,
                        steps_per_eval=1000,
                        max_eval_batches=8,
                        eval_current=True,
                        eval_ema=False,
                    )
                ),
            )

            step = ExecutorStep(
                name=f"grug/moe-isoflop-adamw-lbl-{budget:.0e}-d{hidden_dim}",
                fn=run_grug_moe_trial,
                config=config,
            )
            steps.append(step)

    return steps


moe_isoflop_steps = create_moe_isoflop_steps()

if __name__ == "__main__":
    executor_main(
        steps=moe_isoflop_steps,
        description="MoE core isoflop sweep: 3 budgets x 5 hidden dims on Nemotron mix.",
    )
