# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the 600x / 0.70x constant-LR cell with a one-layer d512 model."""

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
ONE_LAYER_600X_POINT = D512ConstantLrPoint(
    experiment_id=f"{ONE_LAYER_EXPERIMENT_PREFIX}-001",
    token_multiple=600,
    lr_multiplier=0.7,
    num_train_steps=D512_STEPS[600],
)


def build_one_layer_600x_run(
    *, version: str = ONE_LAYER_EXPERIMENT_VERSION
) -> ArtifactStep[LevanterCheckpoint]:
    """Build the isolated one-layer comparison at the original 600x horizon."""
    model = dataclasses.replace(d512_model_config(), num_layers=1)
    return build_d512_constant_lr_run_with_model(
        ONE_LAYER_600X_POINT,
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
def main(version: str) -> None:
    """Materialize the one-layer 600x / 0.70x comparison cell."""
    StepRunner().run([build_one_layer_600x_run(version=version).lower()], max_concurrent=1)


if __name__ == "__main__":
    main()
