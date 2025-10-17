#!/usr/bin/env python
"""
Run tokenization for (one or more) Common Pile components via Marin's Executor.

Defaults to a single dataset (wikimedia). You can pass a substring to select
one or more components, or "all" to run for all Common Pile datasets.

This is a convenience launcher to kick off tokenization "now" on your current
Ray cluster. If you're targeting a TPU v4-8 pod, ensure your environment is
already connected to that cluster (or that your Ray runtime routes jobs there).

Examples:
  # Tokenize wikimedia only
  MARIN_PREFIX=gs://marin-us-central2 uv run experiments/memorize/tokenize_one_of.py

  # Tokenize pre_1929_books only
  MARIN_PREFIX=gs://marin-us-central2 uv run experiments/memorize/tokenize_one_of.py --dataset pre_1929

  # Tokenize all Common Pile components
  MARIN_PREFIX=gs://marin-us-central2 uv run experiments/memorize/tokenize_one_of.py --dataset all
"""

import os

from experiments.common_pile.tokenize_common_pile import common_pile_tokenized
from experiments.llama import llama3_tokenizer
from marin.execution.executor import ExecutorStep, executor_main


# Macro-like constant: restricts tokenization to a single component
# Change this string if you want a different Common Pile dataset
DATASET_PATTERN = "pre_1929"  # hard-coded per request
DEFAULT_DESCRIPTION = "Tokenize Common Pile (pre_1929) on v4-8 using Marin Executor"


def _select_steps(pattern: str) -> list[ExecutorStep]:
    """Filter Common Pile tokenization steps by substring pattern or 'all'."""
    steps = common_pile_tokenized(tokenizer=llama3_tokenizer)
    if pattern.lower() == "all":
        selected = list(steps.values())
    else:
        selected = [step for name, step in steps.items() if pattern in name]
        if not selected:
            raise SystemExit(f"No datasets matched pattern '{pattern}'.")
    return selected


def main():
    prefix = os.environ.get("MARIN_PREFIX")
    if not prefix:
        raise SystemExit("MARIN_PREFIX must be set (e.g., gs://<bucket>)")

    # Pick only the pre_1929 dataset's tokenization step and run it via the Executor framework.
    steps = common_pile_tokenized(tokenizer=llama3_tokenizer)
    selected: list[ExecutorStep] = [step for name, step in steps.items() if DATASET_PATTERN in name]
    if not selected:
        raise SystemExit(f"No Common Pile dataset matched pattern '{DATASET_PATTERN}'")
    if len(selected) > 1:
        # Be explicit to avoid surprises
        names = ", ".join(s.name for s in selected)
        raise SystemExit(f"Pattern '{DATASET_PATTERN}' matched multiple datasets: {names}")

    executor_main(steps=selected, description=DEFAULT_DESCRIPTION)


if __name__ == "__main__":
    main()
