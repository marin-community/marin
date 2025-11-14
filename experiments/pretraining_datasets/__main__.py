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

"""Command-line interface for tokenizing pretraining datasets."""

import sys
from collections.abc import Sequence

from marin.execution.executor import ExecutorStep, executor_main

from experiments.pretraining_datasets import (
    MULTI_SPLIT_DATASETS,
    SIMPLE_TOKENIZED_DATASETS,
)


def get_tokenization_steps(dataset_names: Sequence[str]) -> list[ExecutorStep]:
    """
    Get tokenization steps for the specified dataset names.

    Args:
        dataset_names: List of dataset names. Can be:
            - Simple dataset names: "dclm_baseline", "slimpajama_6b", etc.
            - Multi-split with split name: "dolmino:dclm", "nemotron_cc:hq_actual"
            - Multi-split all: "dolmino:all", "nemotron_cc:all", "dolma:all"

    Returns:
        List of ExecutorSteps for tokenization
    """
    steps = []

    for name in dataset_names:
        # Check if it's a multi-split dataset
        if ":" in name:
            dataset_family, split = name.split(":", 1)

            if dataset_family not in MULTI_SPLIT_DATASETS:
                print(f"Error: Unknown dataset family '{dataset_family}'", file=sys.stderr)
                print(f"Available families: {', '.join(MULTI_SPLIT_DATASETS.keys())}", file=sys.stderr)
                sys.exit(1)

            dataset_info = MULTI_SPLIT_DATASETS[dataset_family]
            available_splits = dataset_info["splits"]
            tokenize_fn = dataset_info["tokenize_fn"]

            if split == "all":
                steps.extend(tokenize_fn().values())
            elif split in available_splits:
                steps.append(tokenize_fn()[f"{dataset_family}/{split}"])
            else:
                print(f"Error: Unknown {dataset_family} split '{split}'", file=sys.stderr)
                print(f"Available splits: {', '.join(available_splits.keys())}", file=sys.stderr)
                sys.exit(1)

        # Check if it's a simple dataset
        elif name in SIMPLE_TOKENIZED_DATASETS:
            steps.append(SIMPLE_TOKENIZED_DATASETS[name])

        else:
            print(f"Error: Unknown dataset '{name}'", file=sys.stderr)
            print(f"Available simple datasets: {', '.join(SIMPLE_TOKENIZED_DATASETS.keys())}", file=sys.stderr)
            print("For multi-split datasets, use: FAMILY:SPLIT or FAMILY:all", file=sys.stderr)
            print(f"Available families: {', '.join(MULTI_SPLIT_DATASETS.keys())}", file=sys.stderr)
            print("Use --list to see all available datasets and splits", file=sys.stderr)
            sys.exit(1)

    return steps


def list_datasets():
    """Print all available datasets and their splits."""
    print("=" * 60)
    print("SIMPLE TOKENIZED DATASETS")
    print("=" * 60)
    for name in sorted(SIMPLE_TOKENIZED_DATASETS.keys()):
        print(f"  {name}")

    for family, info in sorted(MULTI_SPLIT_DATASETS.items()):
        print(f"\n{'=' * 60}")
        print(f"{family.upper()} SPLITS")
        print("=" * 60)
        print(f"  Use: {family}:SPLIT or {family}:all")
        print()
        for split in sorted(info["splits"].keys()):
            print(f"    {split}")


def main():
    """Command-line interface for tokenizing datasets."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m experiments.pretraining_datasets",
        description="Tokenize pretraining datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available datasets
  python -m experiments.pretraining_datasets --list

  # Tokenize simple datasets
  python -m experiments.pretraining_datasets --datasets slimpajama_6b dclm_baseline

  # Tokenize all splits of dolmino
  python -m experiments.pretraining_datasets --datasets dolmino:all

  # Tokenize specific splits
  python -m experiments.pretraining_datasets --datasets dolmino:dclm nemotron_cc:hq_actual dolma:c4
        """,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset names to tokenize. Use --list to see available datasets.",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets and splits",
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if not args.datasets:
        parser.print_help()
        print("\nError: --datasets is required (or use --list to see available datasets)", file=sys.stderr)
        sys.exit(1)

    steps = get_tokenization_steps(args.datasets)

    if not steps:
        print("Error: No tokenization steps found", file=sys.stderr)
        sys.exit(1)

    print(f"Running tokenization for {len(steps)} dataset(s)...")
    executor_main(steps=steps, description=f"Tokenize: {', '.join(args.datasets)}")


if __name__ == "__main__":
    main()
