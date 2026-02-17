# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Simple example of an experiment, which has two steps:
1. Outputs numbers 0 through n - 1.
2. Sum them.
"""

import json
import logging
import os

import fsspec

from marin.execution.step_model import StepSpec
from marin.execution.step_runner import StepRunner

logger = logging.getLogger("ray")


def generate_data(n: int, output_path: str):
    """Generate numbers from 0 to `n` - 1 and write them to `output_path`."""
    numbers = list(range(n))

    # Write to file
    numbers_path = os.path.join(output_path, "numbers.json")
    with fsspec.open(numbers_path, "w") as f:
        json.dump(numbers, f)


def compute_stats(input_path: str, output_path: str):
    """Compute the sum of numbers in the input file and write it to the output file."""
    # Read from file
    numbers_path = os.path.join(input_path, "numbers.json")
    with fsspec.open(numbers_path) as f:
        numbers = json.load(f)

    # Compute statistics
    stats = {
        "sum": sum(numbers),
        "min": min(numbers),
        "max": max(numbers),
    }
    stats_path = os.path.join(output_path, "stats.json")
    with fsspec.open(stats_path, "w") as f:
        json.dump(stats, f)


n = 100

data = StepSpec(
    name="hello_world/data",
    hash_attrs={"n": n},
    fn=lambda output_path: generate_data(n=n, output_path=output_path),
)

stats = StepSpec(
    name="hello_world/stats",
    deps=[data],
    fn=lambda output_path: compute_stats(input_path=data.output_path, output_path=output_path),
)

if __name__ == "__main__":
    StepRunner().run(
        [data, stats],
    )
