# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Simple example of a two-step lazy-artifact pipeline.

Steps:
1. Generate numbers 0 through n-1 and write them to a file.
2. Read the file and compute summary statistics (sum, min, max).

This is the simplest illustration of the Marin lazy-artifact pattern: each step is an
:class:`~marin.execution.artifact.Artifact` built by :class:`~marin.execution.lazy.ArtifactStep`.
Its ``build_config`` receives a :class:`~marin.execution.lazy.StepContext` that resolves
output paths at run time. :func:`~marin.execution.step_runner.StepRunner` executes the
dependency graph in topological order — the data step runs first, then the stats step.
"""

import json
import logging
import os
from dataclasses import dataclass

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, lower
from marin.execution.step_runner import StepRunner
from rigging.filesystem import open_url

logger = logging.getLogger(__name__)

N = 100


@dataclass(frozen=True)
class GenerateDataConfig:
    n: int
    """Number of data points to generate."""

    output_path: str
    """Where to write the numbers."""


@dataclass(frozen=True)
class ComputeStatsConfig:
    input_path: str
    """Path to the file with numbers."""

    output_path: str
    """Where to write the stats."""


def generate_data(config: GenerateDataConfig) -> None:
    """Generate numbers from 0 to ``config.n - 1`` and write them to ``config.output_path``."""
    numbers = list(range(config.n))
    numbers_path = os.path.join(config.output_path, "numbers.json")
    with open_url(numbers_path, "w") as f:
        json.dump(numbers, f)


def compute_stats(config: ComputeStatsConfig) -> None:
    """Compute sum/min/max of the numbers in ``config.input_path`` and write stats to ``config.output_path``."""
    numbers_path = os.path.join(config.input_path, "numbers.json")
    with open_url(numbers_path) as f:
        numbers = json.load(f)
    stats = {
        "sum": sum(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }
    stats_path = os.path.join(config.output_path, "stats.json")
    with open_url(stats_path, "w") as f:
        json.dump(stats, f)


# Step 1: generate the data. build_config resolves ctx.output_path to the artifact's output dir.
_data = ArtifactStep(
    name="hello_world/data",
    version="dev",
    artifact_type=Artifact,
    run=generate_data,
    build_config=lambda ctx: GenerateDataConfig(n=N, output_path=ctx.output_path),
)

# Step 2: compute statistics over the generated data. ctx.artifact_path(_data) gives the path
# of step 1's output; deps=(_data,) ensures it materializes before this step runs.
_stats = ArtifactStep(
    name="hello_world/stats",
    version="dev",
    artifact_type=Artifact,
    run=compute_stats,
    build_config=lambda ctx: ComputeStatsConfig(input_path=ctx.artifact_path(_data), output_path=ctx.output_path),
    deps=(_data,),
)


def build() -> ArtifactStep[Artifact]:
    """The stats computation as a lazy artifact, with its data dependency declared."""
    return _stats


if __name__ == "__main__":
    # Lower the artifact graph to StepSpecs and run it: data generates first,
    # then stats computes.
    StepRunner().run([lower(build())])
