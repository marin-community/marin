# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep K (num_experts_per_token) in {1, 2, 4, 8} for issue #4029.

Runs an isoflop grid at three FLOP budgets and three model widths, varying
only K.  Everything else (E=64, shared expert, optimizer, data) stays fixed
so the comparison isolates the effect of top-K routing breadth.

The FLOP accounting in ``lm_flops_per_token`` includes K, so higher K
means more active FLOPs per token and therefore fewer training steps at
the same budget.  This is the standard isoflop comparison.
"""

import dataclasses
import math
import os
from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import AdamConfig, OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.pretraining_datasets import nemotron_mix_block_shuffle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEQ_LEN: int = 4096
VOCAB_SIZE: int = 128_256
NUM_EXPERTS: int = 64
MIN_BATCH_SIZE: int = 32

BUDGETS: tuple[float, ...] = (1e18, 3e18, 1e19)
HIDDEN_DIMS: tuple[int, ...] = (512, 768, 1024)
K_VALUES: tuple[int, ...] = (1, 2, 4, 8)

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)


# ---------------------------------------------------------------------------
# Launch config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepKLaunchConfig:
    """Per-trial launch config for the K sweep."""

    model: GrugModelConfig
    data: LmDataConfig
    output_path: str
    run_id: str
    resources: ResourceConfig
    steps: int
    batch_size: int
    seed: int
    mp: str
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    grug_trainer: GrugTrainerConfig = field(default_factory=GrugTrainerConfig)
    eval: GrugEvalConfig | None = field(default_factory=GrugEvalConfig)


def _resolve_tracker(tracker: TrackerConfig, run_id: str, output_path: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id, replicate_path=output_path)
    return tracker


def run_sweep_k_trial(config: SweepKLaunchConfig) -> None:
    """Map launch config to GrugRunConfig and dispatch training."""
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id, config.output_path),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 1}),
        require_accelerator=True,
        allow_nondivisible_batch_size=False,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(config.output_path, "checkpoints"),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 1000}],
        ),
    )

    grug_trainer = dataclasses.replace(config.grug_trainer, trainer=trainer)
    run_config = GrugRunConfig(
        model=config.model,
        data=config.data,
        resources=config.resources,
        optimizer=config.optimizer,
        trainer=grug_trainer,
        eval=config.eval,
    )
    run_grug(run_config)


# ---------------------------------------------------------------------------
# Model / schedule helpers
# ---------------------------------------------------------------------------


def _round_to_power_of_two(x: float) -> int:
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def _compute_num_layers(hidden_dim: int) -> int:
    """Depth-width scaling formula from Marin2025Recipe."""
    hs_pow = math.log2(hidden_dim)
    return round(hidden_dim / (64 + (hs_pow * 4.0) - 9))


def _build_model_config(hidden_dim: int, k: int) -> GrugModelConfig:
    num_heads = hidden_dim // 128
    num_layers = _compute_num_layers(hidden_dim)
    return GrugModelConfig(
        vocab_size=VOCAB_SIZE,
        hidden_dim=hidden_dim,
        intermediate_dim=hidden_dim // 2,
        shared_expert_intermediate_dim=hidden_dim,
        num_experts=NUM_EXPERTS,
        num_experts_per_token=k,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        max_seq_len=SEQ_LEN,
        load_balancing_loss_coef=0.001,
        router_z_loss_coef=0.001,
    )


def _flops_per_token(cfg: GrugModelConfig) -> float:
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


def _tokens_and_schedule(budget: float, flops_per_token: float) -> tuple[int, int]:
    """Return (batch_size, train_steps) for a FLOP budget.

    Targets ~2^14 steps.  Uses 3x multiplier (forward + backward).
    """
    tokens = budget / (3 * flops_per_token)
    target_steps = 2**14
    batch_exact = tokens / (target_steps * SEQ_LEN)
    batch_size = max(MIN_BATCH_SIZE, _round_to_power_of_two(batch_exact))
    train_steps = max(1, round(tokens / (batch_size * SEQ_LEN)))
    return batch_size, train_steps


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------


def create_sweep_k_steps() -> list[ExecutorStep]:
    """Create one ExecutorStep per (budget, hidden_dim, K) triple."""
    steps: list[ExecutorStep] = []

    for budget in BUDGETS:
        budget_tag = f"{budget:.0e}"
        for hidden_dim in HIDDEN_DIMS:
            for k in K_VALUES:
                model_cfg = _build_model_config(hidden_dim, k)
                fpt = _flops_per_token(model_cfg)
                batch_size, train_steps = _tokens_and_schedule(budget, fpt)

                effective_bs = batch_size * SEQ_LEN / 4096
                lr = min(0.01, (0.33 * math.sqrt(effective_bs)) / hidden_dim)
                beta2 = max(0.95, 0.98 ** (effective_bs / 128))

                run_id = f"sweep-k-d{hidden_dim}-k{k}-{budget_tag}"
                step_name = f"grug/sweep-k-4029-d{hidden_dim}-k{k}-{budget_tag}"

                config = SweepKLaunchConfig(
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
                        tags=["grug", "sweep-k", "4029", f"d{hidden_dim}", f"k{k}", budget_tag],
                        group="sweep-k-4029",
                        name=run_id,
                    ),
                    optimizer=versioned(
                        AdamConfig(
                            learning_rate=lr,
                            weight_decay=0.1,
                            lr_schedule="cosine",
                            decay=0.2,
                            min_lr_ratio=0.1,
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

                steps.append(
                    ExecutorStep(
                        name=step_name,
                        fn=run_sweep_k_trial,
                        config=config,
                    )
                )

    return steps


sweep_k_steps = create_sweep_k_steps()

if __name__ == "__main__":
    executor_main(
        steps=sweep_k_steps,
        description="Sweep K (num_experts_per_token) in {1,2,4,8} across isoflop budgets. Issue #4029.",
    )
