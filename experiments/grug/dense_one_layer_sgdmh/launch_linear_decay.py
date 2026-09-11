# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the one-layer dense d512 linear-decay LR sweep with SGD-MH."""

import click
from marin.execution.step_runner import StepRunner

from experiments.grug.dense_one_layer_sgdh.launch import (
    D512_LR_MULTIPLIERS,
    D512_STEPS,
    D512_TOKEN_MULTIPLES,
    MAX_CONCURRENT_RUNS,
    DenseSGDHExperiment,
    DenseSGDHPoint,
    build_dense_sgdh_run,
)

EXPERIMENT = DenseSGDHExperiment(
    experiment_prefix="AUG-LIN0-1L-DENSE-SGDMH",
    experiment_version="2026.09.11.1",
    wandb_group="issue-7856-d512-linear-decay-zero-tail-one-layer-dense-sgdmh-tpu",
    lr_schedule="linear",
    schedule_tag="linear-decay-lr",
    optimizer_tag="sgdmh",
    momentum=0.95,
    nesterov=True,
    min_lr_ratio=0.0,
)

SWEEP_POINTS = tuple(
    DenseSGDHPoint(
        experiment_id=f"{EXPERIMENT.experiment_prefix}-{index:03d}",
        token_multiple=token_multiple,
        lr_multiplier=lr_multiplier,
        num_train_steps=D512_STEPS[token_multiple],
    )
    for index, (token_multiple, lr_multiplier) in enumerate(
        (
            (token_multiple, lr_multiplier)
            for token_multiple in D512_TOKEN_MULTIPLES
            for lr_multiplier in D512_LR_MULTIPLIERS
        ),
        start=1,
    )
)


@click.command()
@click.option("--version", default=EXPERIMENT.experiment_version, show_default=True)
@click.option("--max-concurrent", type=click.IntRange(min=1), default=MAX_CONCURRENT_RUNS, show_default=True)
def main(version: str, max_concurrent: int) -> None:
    """Materialize the 25-cell one-layer dense SGD-MH linear-decay sweep."""
    StepRunner().run(
        [build_dense_sgdh_run(point, experiment=EXPERIMENT, version=version).lower() for point in SWEEP_POINTS],
        max_concurrent=min(max_concurrent, len(SWEEP_POINTS)),
    )


if __name__ == "__main__":
    main()
