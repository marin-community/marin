"""
Simple example of an experiment, which has two steps:
1. Outputs numbers 0 through n - 1.
2. Sum them.
"""

import json
import logging
import os
from dataclasses import dataclass

import fsspec

from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path

logger = logging.getLogger("ray")


@dataclass(frozen=True)
class GenerateDataConfig:
    n: int
    """Number of data points to generate."""

    output_path: str
    """Where to write the numbers."""


def generate_data(config: GenerateDataConfig):
    """Generate numbers from 0 to `n` - 1 and write them to `output_path`."""
    numbers = list(range(config.n))

    # Write to file
    numbers_path = os.path.join(config.output_path, "numbers.json")
    with fsspec.open(numbers_path, "w") as f:
        json.dump(numbers, f)


@dataclass(frozen=True)
class ComputeStatsConfig:
    input_path: str
    """Path to the file with numbers."""

    output_path: str
    """Where to write the stats."""


def compute_stats(config: ComputeStatsConfig):
    """Compute the sum of numbers in the input file and write it to the output file."""
    # Read from file
    numbers_path = os.path.join(config.input_path, "numbers.json")
    with fsspec.open(numbers_path) as f:
        numbers = json.load(f)

    # Compute statistics
    stats = {
        "sum": sum(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }
    stats_path = os.path.join(config.output_path, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f)


n = 100

data = ExecutorStep(
    name="hello_world/data",
    description=f"Generate data from 0 to {n}-1.",
    fn=generate_data,
    config=GenerateDataConfig(
        n=n,
        output_path=this_output_path(),
    ),
)

stats = ExecutorStep(
    name="hello_world/stats",
    description="Compute stats of the generated data.",
    fn=compute_stats,
    config=ComputeStatsConfig(
        input_path=output_path_of(data),
        output_path=this_output_path(),
    ),
)

if __name__ == "__main__":
    executor_main(
        steps=[data, stats],
        description="Simple experiment to compute stats of some numbers.",
    )
