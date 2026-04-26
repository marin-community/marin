# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Depth MuP LR sweep for the current Grug MoE recipe."""

import dataclasses
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

from experiments.grug.moe.heuristic import MoeAdamHHeuristic, build_from_heuristic
from experiments.grug.moe.launch import (
    NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
    GrugMoeLaunchConfig,
    run_grug_moe_trial,
)
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig
from experiments.grug.moe.train import GrugEvalConfig, GrugTrainerConfig

DEPTH_MUP_TARGET_STEPS: int = 2**14
DEPTH_MUP_WANDB_GROUP: str = "moe-depth-mup-lr-sweep"


@dataclass(frozen=True)
class DepthMupSweepScale:
    label: str
    budget: float
    hidden_dim: int


DEPTH_MUP_SWEEP_SCALES: tuple[DepthMupSweepScale, ...] = (
    DepthMupSweepScale(label="d512", budget=2.19e17, hidden_dim=512),
    DepthMupSweepScale(label="d768", budget=1.70e18, hidden_dim=768),
    DepthMupSweepScale(label="d1024", budget=9.00e18, hidden_dim=1024),
    DepthMupSweepScale(label="d1280", budget=2.83e19, hidden_dim=1280),
)

DEPTH_MUP_LR_MULTIPLIERS: tuple[float, ...] = (
    0.25,
    0.3535533905932738,
    0.5,
    0.7071067811865476,
    1.0,
    1.4142135623730951,
    2.0,
    2.8284271247461903,
    4.0,
)


def _format_lr_multiplier(multiplier: float) -> str:
    if multiplier <= 0:
        raise ValueError(f"lr_multiplier must be positive, got {multiplier}")
    if multiplier.is_integer():
        return f"{int(multiplier)}x"
    return f"{multiplier:.3g}".replace(".", "p") + "x"


def _scale_optimizer_lrs(optimizer: GrugMoeAdamHConfig, lr_multiplier: float) -> GrugMoeAdamHConfig:
    if lr_multiplier <= 0:
        raise ValueError(f"lr_multiplier must be positive, got {lr_multiplier}")
    expert_lr = optimizer.expert_lr * lr_multiplier if optimizer.expert_lr is not None else None
    return dataclasses.replace(
        optimizer,
        learning_rate=optimizer.learning_rate * lr_multiplier,
        adam_lr=optimizer.adam_lr * lr_multiplier,
        expert_lr=expert_lr,
    )


def build_depth_mup_lr_sweep_config(
    scale: DepthMupSweepScale,
    lr_multiplier: float,
    *,
    output_path: str,
    seed: int = 0,
) -> GrugMoeLaunchConfig:
    heuristic = MoeAdamHHeuristic(depth_mup_residual_scaling=True)
    model, optimizer, batch_size, steps = build_from_heuristic(
        budget=scale.budget,
        hidden_dim=scale.hidden_dim,
        heuristic=heuristic,
        target_steps=DEPTH_MUP_TARGET_STEPS,
    )
    optimizer = _scale_optimizer_lrs(optimizer, lr_multiplier)
    run_id = f"moe-depth-mup-lr-{scale.label}-lr{_format_lr_multiplier(lr_multiplier)}"

    return GrugMoeLaunchConfig(
        model=model,
        data=NEMOTRON_MIX_WITH_DEFAULT_VALIDATION,
        output_path=output_path,
        run_id=run_id,
        resources=ResourceConfig.with_tpu("v5p-8"),
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project="marin_moe",
            tags=["moe", "depth-mup", "lr-sweep"],
            group=DEPTH_MUP_WANDB_GROUP,
            name=None,
        ),
        optimizer=optimizer,
        grug_trainer=GrugTrainerConfig(
            z_loss_weight=1e-4,
            ema_beta=None,
            log_every=1,
        ),
        eval=GrugEvalConfig(
            eval_batch_size=512,
            steps_per_eval=1000,
            max_eval_batches=8,
            eval_current=True,
            eval_ema=False,
        ),
    )


def _versioned_launch_config(config: GrugMoeLaunchConfig) -> GrugMoeLaunchConfig:
    return dataclasses.replace(
        config,
        model=versioned(config.model),
        resources=versioned(config.resources),
        steps=versioned(config.steps),
        batch_size=versioned(config.batch_size),
        seed=versioned(config.seed),
        mp=versioned(config.mp),
        optimizer=versioned(config.optimizer),
        grug_trainer=versioned(config.grug_trainer),
        eval=versioned(config.eval) if config.eval is not None else None,
    )


def build_depth_mup_lr_sweep_step(scale: DepthMupSweepScale, lr_multiplier: float) -> ExecutorStep:
    config = build_depth_mup_lr_sweep_config(scale, lr_multiplier, output_path=this_output_path())
    return ExecutorStep(
        name=f"grug/moe_depth_mup_lr/{config.run_id}",
        fn=run_grug_moe_trial,
        config=_versioned_launch_config(config),
    )


depth_mup_lr_sweep_steps: tuple[ExecutorStep, ...] = tuple(
    build_depth_mup_lr_sweep_step(scale, lr_multiplier)
    for scale in DEPTH_MUP_SWEEP_SCALES
    for lr_multiplier in DEPTH_MUP_LR_MULTIPLIERS
)


if __name__ == "__main__":
    executor_main(
        steps=list(depth_mup_lr_sweep_steps),
        description="Depth MuP residual scaling LR sweep for Grug MoE.",
    )
