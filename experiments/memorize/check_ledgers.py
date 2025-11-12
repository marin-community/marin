#!/usr/bin/env python
"""
Check tokenization step output paths and report ledger status for each.

It uses the Executor to compute concrete output paths for the Common Pile
tokenization steps (same as the memorize runs), then inspects
<cache_dir>/train/shard_ledger.json to determine whether each cache is finished.

Usage:
  MARIN_PREFIX=gs://marin-us-central2 uv run experiments/memorize/check_ledgers.py

Optional:
  DATASET_FILTER: substring to filter datasets (e.g., "pre_1929" or "wikimedia")

Output:
  - For each dataset: prints cache_dir, ledger_exists, is_finished, and the
    executor status file path to delete if you want to re-run just that step.
  - Emits a JSON summary at the end (one object per dataset).
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any

import fsspec

from marin.execution.executor import Executor, get_status_path
from experiments.common_pile.tokenize_common_pile import common_pile_tokenized
from experiments.llama import llama3_tokenizer


def _read_json(path: str) -> dict[str, Any] | None:
    fs, _ = fsspec.core.url_to_fs(path)
    if not fs.exists(path):
        return None
    with fsspec.open(path, "r") as f:
        try:
            txt = f.read()
            return json.loads(txt)
        except Exception:
            return None


def main():
    prefix = os.environ.get("MARIN_PREFIX")
    if not prefix:
        raise SystemExit("MARIN_PREFIX must be set (e.g., gs://<bucket>)")

    dataset_filter = os.environ.get("DATASET_FILTER", "").strip()

    tokenized: Dict[str, object] = common_pile_tokenized(tokenizer=llama3_tokenizer)
    if dataset_filter:
        tokenized = {k: v for k, v in tokenized.items() if dataset_filter in k}

    exec_info_base = os.path.join(prefix, "experiments")
    ex = Executor(prefix=prefix, executor_info_base_path=exec_info_base)

    # Compute versions/paths but do not run
    for step in tokenized.values():
        ex.compute_version(step, is_pseudo_dep=False)
    ex.get_infos()

    results = []
    print(f"Scanning {len(tokenized)} tokenized datasets under prefix={prefix}")

    for name, step in tokenized.items():
        step_canon = ex.canonicalize(step)
        cache_dir = ex.output_paths[step_canon]
        # shard ledger path for training split
        ledger_path = os.path.join(cache_dir, "train", "shard_ledger.json")
        status_path = get_status_path(cache_dir)

        fs, _ = fsspec.core.url_to_fs(cache_dir)
        exists_root = fs.exists(cache_dir)
        exists_train = fs.exists(os.path.join(cache_dir, "train"))
        exists_ledger = fs.exists(ledger_path)

        ledger_json = _read_json(ledger_path) if exists_ledger else None
        is_finished = None
        if ledger_json is not None:
            # tolerate either camelCase or snake_case, but current writer uses snake_case
            is_finished = ledger_json.get("is_finished")
            if is_finished is None:
                is_finished = ledger_json.get("isFinished")

        print(
            f"- {name}:\n"
            f"    cache_dir={cache_dir}\n"
            f"    exists(root)={exists_root} exists(train)={exists_train} exists(ledger)={exists_ledger}\n"
            f"    is_finished={is_finished} executor_status={status_path}"
        )

        results.append(
            dict(
                dataset=name,
                cache_dir=cache_dir,
                exists_root=bool(exists_root),
                exists_train=bool(exists_train),
                exists_ledger=bool(exists_ledger),
                is_finished=is_finished,
                executor_status=status_path,
            )
        )

    print("\nJSON summary:\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

