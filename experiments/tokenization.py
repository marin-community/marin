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
Convenience script for tokenizing datasets.

This module provides a simple interface to run tokenization jobs for any dataset
by name. Supports all datasets defined in pretraining_datasets.py.

Usage:
    # Tokenize a single simple dataset
    python -m experiments.tokenization --datasets slimpajama_6b

    # Tokenize multiple datasets
    python -m experiments.tokenization --datasets dclm_baseline starcoderdata proofpile_2

    # Tokenize all splits of a multi-split dataset
    python -m experiments.tokenization --datasets dolmino:all nemotron_cc:all dolma:all

    # Tokenize specific splits
    python -m experiments.tokenization --datasets dolmino:dclm dolmino:flan nemotron_cc:hq_actual

    # List all available datasets
    python -m experiments.tokenization --list
"""

import argparse
import sys
from typing import Sequence

from experiments.pretraining_datasets import (
    DOLMA_DATASETS,
    DOLMINO_DATASETS,
    NEMOTRON_DATASETS,
    dclm_baseline_tokenized_llama3,
    fineweb_edu_tokenized_llama3,
    proofpile_2_tokenized_llama3,
    slimpajama_6b_tokenized_llama3,
    starcoderdata_tokenized_llama3,
    tokenize_dolma_steps,
    tokenize_dolmino_steps,
    tokenize_nemotron_steps,
)
from marin.execution.executor import ExecutorStep, executor_main


# Mapping of simple dataset names to their tokenized steps
SIMPLE_DATASETS = {
    "dclm_baseline": dclm_baseline_tokenized_llama3,
    "starcoderdata": starcoderdata_tokenized_llama3,
    "proofpile_2": proofpile_2_tokenized_llama3,
    "slimpajama_6b": slimpajama_6b_tokenized_llama3,
    "fineweb_edu": fineweb_edu_tokenized_llama3,
}


def get_tokenization_steps(dataset_names: Sequence[str]) -> list[ExecutorStep]:
    """
    Get tokenization steps for the specified dataset names.

    Args:
        dataset_names: List of dataset names. Can be:
            - Simple dataset names: "dclm_baseline", "slimpajama_6b", etc.
            - Multi-split with split name: "dolmino:dclm", "nemotron_cc:hq_actual"
            - Multi-split all: "dolmino:all", "nemotron_cc:all"

    Returns:
        List of ExecutorSteps for tokenization
    """
    steps = []

    for name in dataset_names:
        # Check if it's a multi-split dataset
        if ":" in name:
            dataset_family, split = name.split(":", 1)

            if dataset_family == "dolmino":
                if split == "all":
                    steps.extend(tokenize_dolmino_steps().values())
                elif split in DOLMINO_DATASETS:
                    steps.append(tokenize_dolmino_steps()[f"dolmino/{split}"])
                else:
                    print(f"Error: Unknown dolmino split '{split}'", file=sys.stderr)
                    print(f"Available splits: {', '.join(DOLMINO_DATASETS.keys())}", file=sys.stderr)
                    sys.exit(1)

            elif dataset_family == "nemotron_cc":
                if split == "all":
                    steps.extend(tokenize_nemotron_steps().values())
                elif split in NEMOTRON_DATASETS:
                    steps.append(tokenize_nemotron_steps()[f"nemotron_cc/{split}"])
                else:
                    print(f"Error: Unknown nemotron_cc split '{split}'", file=sys.stderr)
                    print(f"Available splits: {', '.join(NEMOTRON_DATASETS.keys())}", file=sys.stderr)
                    sys.exit(1)

            elif dataset_family == "dolma":
                if split == "all":
                    steps.extend(tokenize_dolma_steps().values())
                elif split in DOLMA_DATASETS:
                    steps.append(tokenize_dolma_steps()[f"dolma/{split}"])
                else:
                    print(f"Error: Unknown dolma split '{split}'", file=sys.stderr)
                    print(f"Available splits: {', '.join(DOLMA_DATASETS.keys())}", file=sys.stderr)
                    sys.exit(1)

            else:
                print(f"Error: Unknown dataset family '{dataset_family}'", file=sys.stderr)
                print("Available families: dolmino, nemotron_cc, dolma", file=sys.stderr)
                sys.exit(1)

        # Check if it's a simple dataset
        elif name in SIMPLE_DATASETS:
            steps.append(SIMPLE_DATASETS[name])

        else:
            print(f"Error: Unknown dataset '{name}'", file=sys.stderr)
            print(f"Available simple datasets: {', '.join(SIMPLE_DATASETS.keys())}", file=sys.stderr)
            print("For multi-split datasets, use: dolmino:SPLIT, nemotron_cc:SPLIT, dolma:SPLIT", file=sys.stderr)
            print("Use --list to see all available datasets and splits", file=sys.stderr)
            sys.exit(1)

    return steps


def list_datasets():
    """Print all available datasets and their splits."""
    print("=== Simple Datasets ===")
    for name in sorted(SIMPLE_DATASETS.keys()):
        print(f"  {name}")

    print("\n=== Dolmino Splits ===")
    print("  Use: dolmino:SPLIT or dolmino:all")
    for split in sorted(DOLMINO_DATASETS.keys()):
        print(f"    {split}")

    print("\n=== Nemotron CC Splits ===")
    print("  Use: nemotron_cc:SPLIT or nemotron_cc:all")
    for split in sorted(NEMOTRON_DATASETS.keys()):
        print(f"    {split}")

    print("\n=== Dolma Splits ===")
    print("  Use: dolma:SPLIT or dolma:all")
    for split in sorted(DOLMA_DATASETS.keys()):
        print(f"    {split}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize pretraining datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tokenize a single simple dataset
  python -m experiments.tokenization --datasets slimpajama_6b

  # Tokenize multiple datasets
  python -m experiments.tokenization --datasets dclm_baseline starcoderdata

  # Tokenize all dolmino splits
  python -m experiments.tokenization --datasets dolmino:all

  # Tokenize specific splits
  python -m experiments.tokenization --datasets dolmino:dclm nemotron_cc:hq_actual

  # List all available datasets
  python -m experiments.tokenization --list
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
    executor_main(steps=steps, description=f"Tokenize datasets: {', '.join(args.datasets)}")


if __name__ == "__main__":
    main()
