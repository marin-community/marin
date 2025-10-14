#!/usr/bin/env python
"""
Quick sanity-check utility:

- Enumerate tokenization steps for the COMMA Common Pile mixture
- Use the Executor to compute the concrete output paths (without running anything)
- Print the derived tokenized cache paths and check existence via fsspec

Usage:
  MARIN_PREFIX=gs://marin-us-central2 uv run experiments/memorize/debug_tokenize_paths.py 

Optional env vars:
  DATASET_FILTER: substring to filter dataset names (e.g., "pre_1929" or "wikimedia")

Notes:
- This does not execute downloads or tokenization; it only computes paths using the Executor’s versioning logic.
- Existence checks require appropriate GCS credentials if MARIN_PREFIX is a GCS path.
"""

from __future__ import annotations

import json
import os
from typing import Dict

from marin.execution.executor import Executor
from levanter.utils import fsspec_utils

# We reuse the same tokenization component mapping used by memorize utils
from experiments.common_pile.tokenize_common_pile import common_pile_tokenized
from experiments.llama import llama3_tokenizer


def main():
    prefix = os.environ.get("MARIN_PREFIX")
    if not prefix:
        raise SystemExit("MARIN_PREFIX must be set (e.g., gs://<bucket>) to compute output paths.")

    dataset_filter = os.environ.get("DATASET_FILTER", "").strip()

    # Build the tokenization steps (ExecutorSteps) for COMMA Common Pile using the Llama 3 tokenizer
    tokenized: Dict[str, object] = common_pile_tokenized(tokenizer=llama3_tokenizer)

    if dataset_filter:
        tokenized = {k: v for k, v in tokenized.items() if dataset_filter in k}

    # Instantiate an Executor but do not run steps; just compute versions and output paths
    exec_info_base = os.path.join(prefix, "experiments")
    ex = Executor(prefix=prefix, executor_info_base_path=exec_info_base)

    # Register all steps with the executor to compute versions/paths
    for step in tokenized.values():
        ex.compute_version(step, is_pseudo_dep=False)

    # Materialize info structs (populates ex.output_paths et al.)
    ex.get_infos()

    results = []
    print(f"Computed output paths under prefix: {prefix}")
    for name, step in tokenized.items():
        step_canon = ex.canonicalize(step)
        out_path = ex.output_paths[step_canon]

        # Check existence of the derived cache root and its 'train' subdir
        exists_root = False
        exists_train = False
        try:
            exists_root = fsspec_utils.exists(out_path)
            exists_train = fsspec_utils.exists(os.path.join(out_path, "train"))
        except Exception as e:
            # Capture any auth/network errors as a string marker
            exists_root = f"error: {e}"  # type: ignore[assignment]
            exists_train = False

        print(f"- {name}:\n    step_name={step.name}\n    cache_dir={out_path}\n    exists(root)={exists_root} exists(train)={exists_train}")
        results.append(
            {
                "dataset": name,
                "step_name": step.name,
                "cache_dir": out_path,
                "exists_root": exists_root,
                "exists_train": exists_train,
            }
        )

    # Emit machine-readable summary last
    print("\nJSON summary:\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

