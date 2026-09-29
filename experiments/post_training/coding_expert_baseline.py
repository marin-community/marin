# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the pinned Snowball parent on the coding expert benchmark set."""

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.experiment.cli import build_options

from experiments.evaluation.pipeline import EvaluationResult
from experiments.post_training.coding_expert import EXPERIMENT_NAME, baseline_collateral_step, baseline_step


@click.command(help=__doc__)
@click.option("--suite", type=click.Choice(("code", "collateral", "all")), default="all", show_default=True)
@build_options
def main(suite: str) -> ArtifactStep[EvaluationResult] | dict[str, ArtifactStep[EvaluationResult]]:
    version = resolve_version(EXPERIMENT_NAME, None)
    steps = {"code": baseline_step(version), "collateral": baseline_collateral_step(version)}
    if suite == "all":
        return steps
    return steps[suite]


if __name__ == "__main__":
    main()
