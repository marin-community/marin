# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare two fuzzy-dedup artifacts record-for-record.

Used by the issue #6854 research run to prove that the combined-shard launcher
preserves the per-source launcher's logical MinHash/dedup output.
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from marin.execution.artifact import read_record
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArtifactCounts:
    sources: int
    shards: int
    marker_rows: int
    canonical_rows: int


def _result(path: str) -> dict[str, Any]:
    record = read_record(path)
    if record is None or not isinstance(record.result, dict):
        raise FileNotFoundError(f"No artifact result at {path}")
    return record.result


def _read_table(path: str) -> pa.Table:
    with StoragePath(path).open("rb") as fh:
        parquet_file = pq.ParquetFile(fh)
        if parquet_file.metadata.num_rows == 0:
            return pa.table({})
        return parquet_file.read(columns=["id", "attributes"])


def _shards(attr_dir: str) -> dict[str, str]:
    return {str(path).rsplit("/", 1)[-1]: str(path) for path in StoragePath(f"{attr_dir.rstrip('/')}/*.parquet").glob()}


def compare_artifacts(left_path: str, right_path: str) -> ArtifactCounts:
    """Assert identical source/shard marker tables and return exact row counts."""
    left_sources = _result(left_path)["sources"]
    right_sources = _result(right_path)["sources"]
    if left_sources.keys() != right_sources.keys():
        raise AssertionError(
            f"Source mismatch: left-only={sorted(left_sources.keys() - right_sources.keys())}, "
            f"right-only={sorted(right_sources.keys() - left_sources.keys())}"
        )

    shards = 0
    marker_rows = 0
    canonical_rows = 0
    for source_main_dir in sorted(left_sources):
        left_shards = _shards(left_sources[source_main_dir]["attr_dir"])
        right_shards = _shards(right_sources[source_main_dir]["attr_dir"])
        if left_shards.keys() != right_shards.keys():
            raise AssertionError(
                f"Shard mismatch for {source_main_dir}: "
                f"left-only={sorted(left_shards.keys() - right_shards.keys())}, "
                f"right-only={sorted(right_shards.keys() - left_shards.keys())}"
            )
        for basename in sorted(left_shards):
            left_table = _read_table(left_shards[basename])
            right_table = _read_table(right_shards[basename])
            if not left_table.equals(right_table):
                raise AssertionError(
                    f"Marker mismatch for {source_main_dir}/{basename}: "
                    f"left_rows={left_table.num_rows}, right_rows={right_table.num_rows}"
                )
            shards += 1
            marker_rows += left_table.num_rows
            if left_table.num_rows:
                canonical_rows += sum(
                    bool(attributes["is_cluster_canonical"]) for attributes in left_table["attributes"].to_pylist()
                )

    return ArtifactCounts(
        sources=len(left_sources),
        shards=shards,
        marker_rows=marker_rows,
        canonical_rows=canonical_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="left dedup step output path")
    parser.add_argument("--right", required=True, help="right dedup step output path")
    parser.add_argument("--output", help="optional JSON result path")
    args = parser.parse_args()
    configure_logging(logging.INFO)

    counts = compare_artifacts(args.left, args.right)
    payload = {"equal": True, **asdict(counts), "left": args.left, "right": args.right}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        StoragePath(args.output).write_text(text)
    logger.info("%s", text)


if __name__ == "__main__":
    main()
