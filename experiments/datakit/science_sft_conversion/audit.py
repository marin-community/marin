# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify that every science SFT source batch has a complete chat Parquet output."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from marin.datakit.chat_normalize import CHAT_SCHEMA
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.datakit.science_sft_conversion.conversion import (
    INPUT_BATCH_SIZE,
    OUTPUT_ROOT,
    WorkItem,
    _output_path,
    _work_items,
)


def audit(work: list[WorkItem], output_root: str, workers: int) -> dict:
    """Check output coverage, schema, and the minimum record count per batch."""
    expected: dict[str, int] = {}
    for item in work:
        for batch_index, offset in enumerate(range(0, item.rows, INPUT_BATCH_SIZE)):
            output_url = _output_path(item.source, item.url, item.row_group, batch_index, output_root)
            expected[filesystem_for(output_url)[1]] = min(INPUT_BATCH_SIZE, item.rows - offset)

    fs, output_path = filesystem_for(prefix_join(output_root, "outputs/main"))
    found = set(fs.ls(output_path, detail=False)) if fs.exists(output_path) else set()
    missing = expected.keys() - found
    unexpected = {path for path in found - expected.keys() if path.endswith(".parquet")}

    def inspect(path: str) -> tuple[str, int, bool]:
        with fs.open(path, "rb") as stream:
            metadata = pq.ParquetFile(stream).metadata
        return path, metadata.num_rows, metadata.schema.to_arrow_schema() == CHAT_SCHEMA

    short_batches = []
    wrong_schemas = []
    output_rows = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path, rows, valid_schema in pool.map(inspect, sorted(found & expected.keys())):
            output_rows += rows
            if rows < expected[path]:
                short_batches.append(path)
            if not valid_schema:
                wrong_schemas.append(path)

    result = {
        "expected_batches": len(expected),
        "found_batches": len(found & expected.keys()),
        "source_rows": sum(expected.values()),
        "output_rows": output_rows,
        "missing_batches": len(missing),
        "unexpected_batches": len(unexpected),
        "short_batches": len(short_batches),
        "wrong_schemas": len(wrong_schemas),
        "examples": {
            "missing": sorted(missing)[:5],
            "unexpected": sorted(unexpected)[:5],
            "short": short_batches[:5],
            "wrong_schema": wrong_schemas[:5],
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=32, help="Concurrent Parquet footer reads")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    result = audit(_work_items(), OUTPUT_ROOT, args.workers)
    print(json.dumps(result, indent=2))
    if any(result[key] for key in ("missing_batches", "unexpected_batches", "short_batches", "wrong_schemas")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
