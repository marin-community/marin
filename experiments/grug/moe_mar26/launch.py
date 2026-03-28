# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug MoE experiment launcher (mar26).

Pulls optimizer hyperparameters from CompletedAdamHHeuristic, runs through
the grug copy-first training pipeline with QB routing, GatedNorm, XSA.
All layers are MoE (E=64, K=4), no dense layers.
"""

import dataclasses
import math
import os

from dataclasses import dataclass, field
from datetime import timedelta

import jmp
from fray.cluster import ResourceConfig
from levanter.callbacks.profiler import ProfilerConfig
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import LmDataConfig
from levanter.optim import OptimizerConfig
from levanter.tracker import TrackerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.processing.tokenize import add_validation_sets_to_mixture

from experiments.defaults import default_validation_sets
from experiments.grug.moe_mar26.heuristic import CompletedAdamHHeuristic
from experiments.grug.moe_mar26.model import GrugModelConfig
from experiments.grug.moe_mar26.train import GrugEvalConfig, GrugRunConfig, GrugTrainerConfig, run_grug
from experiments.pretraining_datasets import nemotron_mix_block_shuffle


@dataclass(frozen=True)
class GrugMoeLaunchConfig:
    """Last-mile run config for the MoE grug template."""

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


def _resolve_tracker(tracker: TrackerConfig, run_id: str) -> TrackerConfig:
    if isinstance(tracker, WandbConfig):
        return dataclasses.replace(tracker, name=run_id)
    return tracker


def run_grug_moe_trial(config: GrugMoeLaunchConfig) -> None:
    trainer = TrainerConfig(
        id=config.run_id,
        seed=config.seed,
        train_batch_size=config.batch_size,
        num_train_steps=config.steps,
        profiler=ProfilerConfig(enabled=False, start_step=5, num_steps=100, perfetto_link=False),
        mp=jmp.get_policy(config.mp),
        tracker=_resolve_tracker(config.tracker, config.run_id),
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"expert": 4}),
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


# ============================================================
# Config
# ============================================================

SEQ_LEN: int = 4096
MIN_BATCH_SIZE: int = 32

HEURISTIC = CompletedAdamHHeuristic()

NEMOTRON_MIX_WITH_DEFAULT_VALIDATION = add_validation_sets_to_mixture(
    nemotron_mix_block_shuffle,
    default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer),
)


def _round_to_power_of_two(x: float) -> int:
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def _compute_flops_per_token(cfg: GrugModelConfig) -> float:
    """Non-embedding FLOPs per token (excludes lm_head), with correct shared expert dim."""
    fpt_with_lm_head = lm_flops_per_token(
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
    lm_head_flops = 2 * cfg.hidden_dim * cfg.vocab_size
    num_shared = 1 if cfg.shared_expert_intermediate_dim > 0 else 0
    shared_correction = 2 * 3 * cfg.hidden_dim * (cfg.shared_expert_intermediate_dim - cfg.intermediate_dim) * num_shared
    shared_correction *= cfg.num_layers
    return fpt_with_lm_head - lm_head_flops + shared_correction


def _compute_tokens_and_batch(
    budget: float, flops_per_token: float, target_steps: int = 2**14
) -> tuple[float, int, int]:
    tokens = budget / (3 * flops_per_token)
    batch_exact = tokens / (target_steps * SEQ_LEN)
    batch_size = max(MIN_BATCH_SIZE, _round_to_power_of_two(batch_exact))
    train_steps = max(1, round(tokens / (batch_size * SEQ_LEN)))
    return tokens, batch_size, train_steps


# ============================================================
# ISOFlop sweep grid
# ============================================================

BUDGETS: tuple[float, ...] = (1e18, 3e18, 1e19)
HIDDEN_DIMS: tuple[int, ...] = (512, 768, 1024, 1536, 2048)


def create_moe_isoflop_steps() -> list[ExecutorStep]:
    """Create ExecutorSteps for the MoE isoflop grid."""
    steps: list[ExecutorStep] = []

    for budget in BUDGETS:
        for hidden_dim in HIDDEN_DIMS:
            model_cfg = HEURISTIC.build_model_config(hidden_dim)
            fpt = _compute_flops_per_token(model_cfg)
            tokens, batch_size, train_steps = _compute_tokens_and_batch(budget, fpt)

            optimizer = HEURISTIC.build_optimizer_config(batch_size, tokens)

            run_id = f"isoflop-moe-v7-{budget:.0e}-d{hidden_dim}"

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
                    tags=["grug", "moe-core", "isoflop", "v7-tune", f"budget={budget:.0e}", f"d={hidden_dim}", "lr=3x"],
                    group="isoflop-moe-v7-tune",
                    name=run_id,
                ),
                optimizer=versioned(optimizer),
                grug_trainer=versioned(
                    GrugTrainerConfig(
                        z_loss_weight=HEURISTIC.z_loss_weight,
                        ema_beta=None,
                        log_every=1,
                    )
                ),
                eval=versioned(
                    GrugEvalConfig(
                        eval_batch_size=64 if hidden_dim >= 2048 else 512,
                        steps_per_eval=1000,
                        max_eval_batches=64 if hidden_dim >= 2048 else 8,
                        eval_current=True,
                        eval_ema=False,
                    )
                ),
            )

            step = ExecutorStep(
                name=f"grug/{run_id}",
                fn=run_grug_moe_trial,
                config=config,
            )
            steps.append(step)

    return steps


moe_isoflop_steps = create_moe_isoflop_steps()


# ============================================================
# 1e22 run: d=3200, 326B tokens
# ============================================================


def create_1e22_run() -> list[ExecutorStep]:
    hidden_dim = 3200
    tokens_target = 326e9
    model_cfg = HEURISTIC.build_model_config(hidden_dim)

    batch_size = 1024
    train_steps = max(1, round(tokens_target / (batch_size * SEQ_LEN)))
    actual_tokens = train_steps * batch_size * SEQ_LEN
    optimizer = HEURISTIC.build_optimizer_config(batch_size, actual_tokens)

    run_id = "moe-v7-1e22-d3200"

    config = GrugMoeLaunchConfig(
        model=versioned(model_cfg),
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=this_output_path(),
        run_id=run_id,
        resources=versioned(ResourceConfig.with_tpu("v4-512")),
        steps=versioned(train_steps),
        batch_size=versioned(batch_size),
        seed=versioned(0),
        mp=versioned("params=float32,compute=bfloat16,output=bfloat16"),
        tracker=WandbConfig(
            project="dial_moe",
            tags=["grug", "moe-core", "v7", "budget=1e+22", "d=3200"],
            group="moe-1e22",
            name=run_id,
        ),
        optimizer=versioned(optimizer),
        grug_trainer=versioned(
            GrugTrainerConfig(
                z_loss_weight=HEURISTIC.z_loss_weight,
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

    return [
        ExecutorStep(
            name=f"grug/{run_id}",
            fn=run_grug_moe_trial,
            config=config,
        )
    ]


e22_run_steps = create_1e22_run()


if __name__ == "__main__":
    executor_main(
        steps=e22_run_steps,
        description="MoE v7: 1e22 d=3200, 326B tokens",
    )
