# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Good-enough 10T gate: E=128 candidate baseline.

Runs the grug MoE template with 128 experts (K=4 active per token) at
isoflop budgets from 1e18 to 1e19, following the scaling recipe from
the parent sweep (#3469) and gate (#4013).

Model dimensions scale from hidden_dim following the iteration-02 recipe:
    intermediate_dim      = hidden_dim // 2   (per-expert FFN)
    shared_expert_dim     = hidden_dim         (shared expert)
    num_heads             = hidden_dim // 128
    num_layers            = round(hidden_dim / (64 + 4*log2(hidden_dim) - 9))

Ref: https://github.com/marin-community/marin/issues/4015
"""

import math

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from levanter.tracker.wandb import WandbConfig
from levanter.utils.flop_utils import lm_flops_per_token
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe.launch import GrugMoeLaunchConfig, run_grug_moe
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig
from experiments.pretraining_datasets import nemotron_mix_block_shuffle

SEQ_LEN = 4096
VOCAB_SIZE = 128_256
NUM_EXPERTS = 128
NUM_EXPERTS_PER_TOKEN = 4
MIN_BATCH_SIZE = 32

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)


def _round_to_power_of_two(x: float) -> int:
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def _compute_num_layers(hidden_dim: int) -> int:
    """Depth-width formula from the Marin 2025 recipe."""
    hs_pow = math.log2(hidden_dim)
    return round(hidden_dim / (64 + (hs_pow * 4.0) - 9))


def build_e128_model_config(hidden_dim: int) -> GrugModelConfig:
    """Build a GrugModelConfig with E=128, scaling all dims from hidden_dim."""
    num_heads = hidden_dim // 128
    num_layers = _compute_num_layers(hidden_dim)
    return GrugModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=hidden_dim,
        intermediate_dim=hidden_dim // 2,
        shared_expert_intermediate_dim=hidden_dim,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=NUM_EXPERTS_PER_TOKEN,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        max_seq_len=SEQ_LEN,
        load_balancing_loss_coef=0.001,
        router_z_loss_coef=0.001,
        initializer_std=0.5 / math.sqrt(hidden_dim),
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


# ---------------------------------------------------------------------------
# Isoflop sweep: same budgets and hidden dims as iteration-02, with E=128
# ---------------------------------------------------------------------------

BUDGETS: tuple[float, ...] = (1e18, 3e18, 1e19)
HIDDEN_DIMS: tuple[int, ...] = (512, 768, 1024, 1536, 2048)


def _compute_tokens_and_batch(budget: float, flops_per_token: float) -> tuple[float, int, int]:
    """Compute tokens, batch_size, and steps for a FLOP budget.

    Uses 3x multiplier (forward + backward), targets ~2^14 steps.
    """
    tokens = budget / (3 * flops_per_token)
    target_steps = 2**14
    batch_exact = tokens / (target_steps * SEQ_LEN)
    batch_size = max(MIN_BATCH_SIZE, _round_to_power_of_two(batch_exact))
    train_steps = max(1, round(tokens / (batch_size * SEQ_LEN)))
    return tokens, batch_size, train_steps


def create_isoflop_steps() -> list[ExecutorStep]:
    """Create ExecutorSteps for the E=128 isoflop grid."""
    steps: list[ExecutorStep] = []

    for budget in BUDGETS:
        budget_tag = f"{budget:.0e}"
        for hidden_dim in HIDDEN_DIMS:
            model_cfg = build_e128_model_config(hidden_dim)
            fpt = _compute_flops_per_token(model_cfg)
            _tokens, batch_size, train_steps = _compute_tokens_and_batch(budget, fpt)

            effective_bs = batch_size * SEQ_LEN / 4096
            lr = min(0.01, (0.33 * math.sqrt(effective_bs)) / hidden_dim)
            beta2 = max(0.95, 0.98 ** (effective_bs / 128))

            run_id = f"good10t-e128-{budget_tag}-d{hidden_dim}"
            step_name = f"grug/good10t-e128-{budget_tag}-d{hidden_dim}"

            config = GrugMoeLaunchConfig(
                model=versioned(model_cfg),
                data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
                output_path=this_output_path(),
                run_id=run_id,
                resources=versioned(ResourceConfig.with_tpu("v4-8")),
                steps=versioned(train_steps),
                batch_size=versioned(batch_size),
                seed=versioned(0),
                mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
                tracker=WandbConfig(
                    project="marin",
                    tags=["grug", "moe", "good-10t", "e128", budget_tag],
                    group="good10t-e128-isoflop",
                    name=run_id,
                ),
                optimizer=versioned(
                    AdamConfig(
                        learning_rate=lr,
                        beta1=0.96,
                        beta2=beta2,
                        epsilon=1e-15,
                        weight_decay=0.1,
                        lr_schedule="linear",
                        decay=0.2,
                        min_lr_ratio=0.0,
                        warmup=0.1,
                        max_grad_norm=1,
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
                        eval_batch_size=min(batch_size, 512),
                        steps_per_eval=1000,
                        max_eval_batches=8,
                        eval_current=True,
                        eval_ema=False,
                    )
                ),
            )

            steps.append(
                ExecutorStep(
                    name=step_name,
                    fn=run_grug_moe,
                    config=config,
                )
            )

    return steps


good_10t_e128_steps = create_isoflop_steps()


if __name__ == "__main__":
    executor_main(
        steps=good_10t_e128_steps,
        description="Good-enough 10T gate: E=128 isoflop sweep (3 budgets x 5 hidden dims). Fixes #4015.",
    )
