"""Materialize Evalchemy's pinned NUPA sample as row-oriented JSONL.

Example:
    uv run --extra nupa-loose python -m eval.chat_benchmarks.NUPA-Loose.data_prep.flatten_hf_dataset \
        --output /tmp/nupa_test.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..eval_instruct import (
    DEFAULT_NUM_EACH,
    DEFAULT_RANDOM_SEED,
    DEFAULT_SPLIT,
    SOURCE_DATASET_NAME,
    SOURCE_DATASET_REVISION,
    download_nupa_source,
    iter_nupa_source_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=SOURCE_DATASET_NAME)
    parser.add_argument("--revision", default=SOURCE_DATASET_REVISION)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--num-each", type=int, default=DEFAULT_NUM_EACH)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()

    source = args.source_file or download_nupa_source(
        dataset_name=args.dataset_name,
        dataset_revision=args.revision,
        split=args.split,
    )
    count = convert_file(
        source,
        args.output,
        split=args.split,
        num_each=args.num_each,
        random_seed=args.random_seed,
    )
    print(f"Wrote {count} flattened NUPA records to {args.output}")


def convert_file(
    source: Path,
    output: Path,
    *,
    split: str,
    num_each: int = DEFAULT_NUM_EACH,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> int:
    """Write the deterministic NUPA evaluation sample and return its row count."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as output_file:
        for record in iter_nupa_source_records(
            source,
            split=split,
            num_each=num_each,
            random_seed=random_seed,
        ):
            output_file.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    main()
