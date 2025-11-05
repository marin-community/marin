# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    with fsspec.open(stats_path, "w") as f:
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
