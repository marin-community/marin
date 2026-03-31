# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize a single dataset by name. Usage: python experiments/tokenize_single.py --dataset <name>"""

import dataclasses
import sys

from marin.execution.executor import executor_main

# Import all the individual tokenization steps from the main experiment
from experiments.exp_nemotron_v2_finepdfs import ALL_COMPONENTS


@dataclasses.dataclass
class Config:
    dataset: str = ""


def main():
    # Parse --dataset arg manually to avoid draccus conflicts with executor_main
    dataset = None
    for i, arg in enumerate(sys.argv):
        if arg == "--dataset" and i + 1 < len(sys.argv):
            dataset = sys.argv[i + 1]
            # Remove these args so executor_main doesn't see them
            sys.argv.pop(i)
            sys.argv.pop(i)
            break

    if not dataset:
        print("Available datasets:")
        for name in sorted(ALL_COMPONENTS.keys()):
            print(f"  {name}")
        sys.exit(1)

    if dataset not in ALL_COMPONENTS:
        print(f"Unknown dataset: {dataset}")
        print(f"Available: {sorted(ALL_COMPONENTS.keys())}")
        sys.exit(1)

    step = ALL_COMPONENTS[dataset]
    executor_main(steps=[step])


if __name__ == "__main__":
    main()
