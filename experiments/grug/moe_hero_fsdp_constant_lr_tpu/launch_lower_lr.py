# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the bracketed low-LR follow-up to the #7856 d512 constant-LR sweep."""

from collections.abc import Sequence

import click
from marin.execution.lazy import ArtifactStep
from marin.execution.step_runner import StepRunner
from marin.training.training import LevanterCheckpoint

from experiments.grug.moe_hero_fsdp_constant_lr_tpu.launch import (
    D512_STEPS,
    D512_TOKEN_MULTIPLES,
    MAX_CONCURRENT_RUNS,
    D512ConstantLrPoint,
    build_d512_constant_lr_run,
)

LOW_LR_EXPERIMENT_PREFIX = "AUG-LRC-LOW"
LOW_LR_EXPERIMENT_VERSION = "2026.08.27"
LOW_LR_WANDB_GROUP = "issue-7856-d512-constant-lr-low-tpu"
LOW_LR_MULTIPLIERS = (0.1, 0.2, 0.32, 0.45, 0.7)
NEW_LOW_LR_MULTIPLIERS = LOW_LR_MULTIPLIERS[:-1]


def _new_lower_lr_coordinates() -> tuple[tuple[int, float], ...]:
    reused_budgets = D512_TOKEN_MULTIPLES[:-1]
    new_coordinates = tuple(
        (token_multiple, lr_multiplier)
        for token_multiple in reused_budgets
        for lr_multiplier in NEW_LOW_LR_MULTIPLIERS
    )
    unfinished_600x_coordinates = tuple((600, lr_multiplier) for lr_multiplier in LOW_LR_MULTIPLIERS)
    return new_coordinates + unfinished_600x_coordinates


D512_NEW_LOW_LR_POINTS = tuple(
    D512ConstantLrPoint(
        experiment_id=f"{LOW_LR_EXPERIMENT_PREFIX}-{index:03d}",
        token_multiple=token_multiple,
        lr_multiplier=lr_multiplier,
        num_train_steps=D512_STEPS[token_multiple],
    )
    for index, (token_multiple, lr_multiplier) in enumerate(_new_lower_lr_coordinates(), start=1)
)


def select_d512_lower_lr_points(
    *,
    token_multiples: Sequence[int] = (),
    lr_multipliers: Sequence[float] = (),
) -> tuple[D512ConstantLrPoint, ...]:
    """Select an exact subset of the new low-LR cells."""
    selected = tuple(
        point
        for point in D512_NEW_LOW_LR_POINTS
        if (not token_multiples or point.token_multiple in token_multiples)
        and (not lr_multipliers or point.lr_multiplier in lr_multipliers)
    )
    if not selected:
        raise ValueError("the d512 low-LR selection is empty")
    return selected


def build_d512_lower_lr_run(
    point: D512ConstantLrPoint,
    *,
    version: str = LOW_LR_EXPERIMENT_VERSION,
) -> ArtifactStep[LevanterCheckpoint]:
    """Build one uniquely named low-LR follow-up cell."""
    return build_d512_constant_lr_run(
        point,
        version=version,
        wandb_group=LOW_LR_WANDB_GROUP,
        wandb_sweep_tag=LOW_LR_EXPERIMENT_PREFIX,
    )


@click.command()
@click.option(
    "--token-multiple",
    "token_multiples",
    multiple=True,
    type=click.Choice([str(value) for value in D512_TOKEN_MULTIPLES]),
    help="Select one or more token budgets. Omit to select every new cell.",
)
@click.option(
    "--lr-multiplier",
    "lr_multipliers",
    multiple=True,
    type=click.Choice([f"{value:g}" for value in LOW_LR_MULTIPLIERS]),
    help="Select one or more low-LR multipliers. Omit to select every new value.",
)
@click.option(
    "--version",
    default=LOW_LR_EXPERIMENT_VERSION,
    show_default=True,
    help="Artifact version shared by the low-LR matrix and exact retries.",
)
@click.option(
    "--max-concurrent",
    type=click.IntRange(min=1),
    default=MAX_CONCURRENT_RUNS,
    show_default=True,
    help="Maximum TPU cells materialized concurrently by this parent.",
)
def main(
    token_multiples: tuple[str, ...],
    lr_multipliers: tuple[str, ...],
    version: str,
    max_concurrent: int,
) -> None:
    """Materialize the 21 new cells needed to bracket the constant-LR optimum."""
    points = select_d512_lower_lr_points(
        token_multiples=tuple(int(value) for value in token_multiples),
        lr_multipliers=tuple(float(value) for value in lr_multipliers),
    )
    StepRunner().run(
        [build_d512_lower_lr_run(point, version=version).lower() for point in points],
        max_concurrent=min(max_concurrent, len(points)),
    )


if __name__ == "__main__":
    main()
