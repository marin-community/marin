# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the bracketed 600x constant-LR sweep with a one-layer d512 model."""

import dataclasses

import click
from marin.execution.lazy import ArtifactStep
from marin.execution.step_runner import StepRunner
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe_hero_fsdp_constant_lr_tpu.launch import (
    D512_STEPS,
    D512ConstantLrPoint,
    build_d512_constant_lr_run_with_model,
    d512_model_config,
)

ONE_LAYER_EXPERIMENT_PREFIX = "AUG-LRC-1L"
ONE_LAYER_EXPERIMENT_VERSION = "2026.09.10"
ONE_LAYER_WANDB_GROUP = "issue-7856-d512-constant-lr-one-layer-tpu"
ONE_LAYER_LR_MULTIPLIERS = (0.1, 0.2, 0.32, 0.45, 0.7)
ONE_LAYER_600X_POINTS = tuple(
    D512ConstantLrPoint(
        experiment_id=f"{ONE_LAYER_EXPERIMENT_PREFIX}-{index:03d}",
        token_multiple=600,
        lr_multiplier=lr_multiplier,
        num_train_steps=D512_STEPS[600],
    )
    for index, lr_multiplier in enumerate(ONE_LAYER_LR_MULTIPLIERS, start=1)
)


def build_one_layer_600x_run(
    point: D512ConstantLrPoint,
    *,
    version: str = ONE_LAYER_EXPERIMENT_VERSION,
) -> ArtifactStep[LevanterCheckpoint]:
    """Build one one-layer comparison cell at the original 600x horizon."""
    model = dataclasses.replace(d512_model_config(), num_layers=1)
    return build_d512_constant_lr_run_with_model(
        point,
        model=model,
        version=version,
        wandb_group=ONE_LAYER_WANDB_GROUP,
        wandb_sweep_tag=ONE_LAYER_EXPERIMENT_PREFIX,
    )


@click.command()
@click.option(
    "--version",
    default=ONE_LAYER_EXPERIMENT_VERSION,
    show_default=True,
    help="Artifact version shared by this run and exact retries.",
)
@click.option(
    "--max-concurrent",
    type=click.IntRange(min=1),
    default=len(ONE_LAYER_600X_POINTS),
    show_default=True,
    help="Maximum TPU cells materialized concurrently by this parent.",
)
def main(version: str, max_concurrent: int) -> None:
    """Materialize the five-point one-layer 600x comparison sweep."""
    StepRunner().run(
        [build_one_layer_600x_run(point, version=version).lower() for point in ONE_LAYER_600X_POINTS],
        max_concurrent=min(max_concurrent, len(ONE_LAYER_600X_POINTS)),
    )


if __name__ == "__main__":
    main()
