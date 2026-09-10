# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the one-layer d512 LR-scaling sweep with raw-gradient SGD-H."""

import dataclasses

import click
from marin.execution.lazy import ArtifactStep
from marin.execution.step_runner import StepRunner
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe_hero_fsdp_constant_lr_tpu.launch import (
    D512_STEPS,
    D512_TOKEN_MULTIPLES,
    MAX_CONCURRENT_RUNS,
    D512ConstantLrPoint,
    build_d512_constant_lr_run_with_model_and_optimizer,
    constant_lr_optimizer,
    d512_model_config,
)
from experiments.grug.moe_hero_fsdp_constant_lr_tpu.optimizer import GrugMoeSGDHConfig

SGDH_EXPERIMENT_PREFIX = "AUG-LRC-1L-SGDH"
SGDH_EXPERIMENT_VERSION = "2026.09.10"
SGDH_WANDB_GROUP = "issue-7856-d512-constant-lr-one-layer-sgdh-tpu"
SGDH_LR_MULTIPLIERS = (0.1, 0.2, 0.32, 0.45, 0.7)
SGDH_LR_SCALING_POINTS = tuple(
    D512ConstantLrPoint(
        experiment_id=f"{SGDH_EXPERIMENT_PREFIX}-{index:03d}",
        token_multiple=token_multiple,
        lr_multiplier=lr_multiplier,
        num_train_steps=D512_STEPS[token_multiple],
    )
    for index, (token_multiple, lr_multiplier) in enumerate(
        (
            (token_multiple, lr_multiplier)
            for token_multiple in D512_TOKEN_MULTIPLES
            for lr_multiplier in SGDH_LR_MULTIPLIERS
        ),
        start=1,
    )
)


def sgdh_optimizer(point: D512ConstantLrPoint) -> GrugMoeSGDHConfig:
    """Match the MuonH cell's schedules and fallback groups, replacing its matrix update."""
    muonh = constant_lr_optimizer(point)
    return GrugMoeSGDHConfig(
        learning_rate=muonh.learning_rate,
        adam_lr=muonh.adam_lr,
        weight_decay=muonh.weight_decay,
        min_lr_ratio=muonh.min_lr_ratio,
        warmup=muonh.warmup,
        decay=muonh.decay,
        rewarmup=muonh.rewarmup,
        cooldown=muonh.cooldown,
        cycle_length=muonh.cycle_length,
        cycles=muonh.cycles,
        lr_schedule=muonh.lr_schedule,
        haps=muonh.haps,
        weight_decay_modules=muonh.weight_decay_modules,
        default_weight_decay_mask=muonh.default_weight_decay_mask,
        beta1=muonh.beta1,
        beta2=muonh.beta2,
        epsilon=muonh.epsilon,
        max_grad_norm=muonh.max_grad_norm,
    )


def build_one_layer_sgdh_lr_scaling_run(
    point: D512ConstantLrPoint,
    *,
    version: str = SGDH_EXPERIMENT_VERSION,
) -> ArtifactStep[LevanterCheckpoint]:
    """Build one cell of the one-layer raw-gradient SGD-H sweep."""
    model = dataclasses.replace(d512_model_config(), num_layers=1)
    return build_d512_constant_lr_run_with_model_and_optimizer(
        point,
        model=model,
        optimizer=sgdh_optimizer(point),
        version=version,
        wandb_group=SGDH_WANDB_GROUP,
        wandb_sweep_tag=SGDH_EXPERIMENT_PREFIX,
    )


@click.command()
@click.option(
    "--version",
    default=SGDH_EXPERIMENT_VERSION,
    show_default=True,
    help="Artifact version shared by this sweep and exact retries.",
)
@click.option(
    "--max-concurrent",
    type=click.IntRange(min=1),
    default=MAX_CONCURRENT_RUNS,
    show_default=True,
    help="Maximum TPU cells materialized concurrently by this parent.",
)
def main(version: str, max_concurrent: int) -> None:
    """Materialize the 25-cell one-layer SGD-H LR-scaling sweep."""
    StepRunner().run(
        [build_one_layer_sgdh_lr_scaling_run(point, version=version).lower() for point in SGDH_LR_SCALING_POINTS],
        max_concurrent=min(max_concurrent, len(SGDH_LR_SCALING_POINTS)),
    )


if __name__ == "__main__":
    main()
