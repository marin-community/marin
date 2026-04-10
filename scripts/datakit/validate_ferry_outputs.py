# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate datakit smoke ferry outputs.

Run after the iris job for the datakit smoke ferry has completed. Resolves
the output prefix via ``MARIN_PREFIX`` (falling back to
``marin_temp_bucket(ttl_days=1, prefix="datakit-smoke")`` — same default as
the ferry entrypoint) and asserts that each pipeline stage produced non-empty
output with correct schemas and plausible row counts.
"""

import logging
import os
import sys

import pyarrow.parquet as pq
from levanter.store.cache import CacheLedger
from marin.utils import fsspec_glob
from rigging.filesystem import marin_temp_bucket, url_to_fs
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

STAGES = ("normalize", "dedup", "consolidate", "tokens")

# fineweb-edu sample/10BT has 14 shards, normalize produces 106 output files
MIN_NORMALIZE_FILES = 100
MIN_NORMALIZE_ROWS = 5_000_000  # ~9.2M expected
NORMALIZE_REQUIRED_COLUMNS = {"id", "text", "url", "source_id"}

DEDUP_REQUIRED_COLUMNS = {"id", "attributes"}
MIN_CONSOLIDATE_FILES = 1


def _assert_non_empty(path: str) -> int:
    """Assert path exists and is non-empty. Returns entry count."""
    fs, fs_path = url_to_fs(path)
    if not fs.exists(fs_path):
        raise SystemExit(f"Missing output directory: {path}")
    entries = fs.find(fs_path)
    if not entries:
        raise SystemExit(f"Output directory is empty: {path}")
    logger.info("OK %s (%d entries)", path, len(entries))
    return len(entries)


def _count_parquet_rows(files: list[str]) -> int:
    """Sum row counts from parquet file metadata (no data read)."""
    fs, _ = url_to_fs(files[0])
    total = 0
    for path in files:
        with fs.open(path, "rb") as f:
            total += pq.ParquetFile(f).metadata.num_rows
    return total


def _check_parquet_schema(path: str, required_columns: set[str]) -> None:
    """Verify a parquet file contains the required columns."""
    fs, _ = url_to_fs(path)
    with fs.open(path, "rb") as f:
        schema = pq.ParquetFile(f).schema_arrow
    actual = set(schema.names)
    missing = required_columns - actual
    if missing:
        raise SystemExit(f"Schema mismatch in {path}: missing columns {missing}")


def _validate_normalize(base: str) -> int:
    """Validate normalize output. Returns total row count."""
    norm_path = f"{base}/normalize"
    _assert_non_empty(norm_path)

    files = fsspec_glob(f"{norm_path}/*.parquet")
    if len(files) < MIN_NORMALIZE_FILES:
        raise SystemExit(f"Normalize: expected >= {MIN_NORMALIZE_FILES} files, got {len(files)}")

    _check_parquet_schema(files[0], NORMALIZE_REQUIRED_COLUMNS)

    total_rows = _count_parquet_rows(files)
    if total_rows < MIN_NORMALIZE_ROWS:
        raise SystemExit(f"Normalize: expected >= {MIN_NORMALIZE_ROWS} rows, got {total_rows}")

    logger.info("Normalize OK: %d files, %d rows", len(files), total_rows)
    return total_rows


def _validate_dedup(base: str, normalize_rows: int) -> int:
    """Validate dedup output. Returns number of flagged duplicates."""
    dedup_data_path = f"{base}/dedup/data"
    _assert_non_empty(dedup_data_path)

    files = fsspec_glob(f"{dedup_data_path}/*.parquet")
    _check_parquet_schema(files[0], DEDUP_REQUIRED_COLUMNS)

    dedup_rows = _count_parquet_rows(files)
    if dedup_rows <= 0:
        raise SystemExit("Dedup: no duplicate documents flagged")
    if dedup_rows >= normalize_rows:
        raise SystemExit(
            f"Dedup: flagged {dedup_rows} dups >= {normalize_rows} total docs — "
            "something is wrong"
        )

    logger.info("Dedup OK: %d files, %d flagged duplicates (%.1f%% of %d docs)",
                len(files), dedup_rows, 100 * dedup_rows / normalize_rows, normalize_rows)
    return dedup_rows


def _validate_consolidate(base: str, normalize_rows: int, dedup_rows: int) -> int:
    """Validate consolidate output. Returns consolidated row count."""
    consol_path = f"{base}/consolidate"
    _assert_non_empty(consol_path)

    files = fsspec_glob(f"{consol_path}/*.parquet")
    if len(files) < MIN_CONSOLIDATE_FILES:
        raise SystemExit(f"Consolidate: expected >= {MIN_CONSOLIDATE_FILES} files, got {len(files)}")

    consol_rows = _count_parquet_rows(files)
    expected_max = normalize_rows
    expected_min = normalize_rows - dedup_rows
    # Allow some tolerance — consolidate may drop a few extra docs
    if consol_rows > expected_max:
        raise SystemExit(
            f"Consolidate: {consol_rows} rows > {expected_max} normalize rows — "
            "more rows after dedup removal is impossible"
        )
    if consol_rows < expected_min * 0.9:
        raise SystemExit(
            f"Consolidate: {consol_rows} rows much less than expected ~{expected_min} — "
            "too many docs removed"
        )

    logger.info("Consolidate OK: %d files, %d rows (removed %d docs)",
                len(files), consol_rows, normalize_rows - consol_rows)
    return consol_rows


def main() -> None:
    configure_logging()
    prefix = os.environ.get("MARIN_PREFIX") or marin_temp_bucket(ttl_days=1)
    prefix = prefix.rstrip("/")
    run_id = os.environ["SMOKE_RUN_ID"]
    base = f"{prefix}/datakit-smoke/{run_id}"

    normalize_rows = _validate_normalize(base)
    dedup_rows = _validate_dedup(base, normalize_rows)
    consol_rows = _validate_consolidate(base, normalize_rows, dedup_rows)

    # Tokens
    _assert_non_empty(f"{base}/tokens")
    train_dir = f"{base}/tokens/train"
    ledger = CacheLedger.load(train_dir)
    if not ledger.is_finished:
        raise SystemExit(f"Tokenizer cache ledger not finished: {train_dir}")
    if ledger.total_num_rows <= 0:
        raise SystemExit(f"Tokenizer cache ledger has 0 rows: {train_dir}")
    logger.info("Tokenizer cache OK: %d rows at %s", ledger.total_num_rows, train_dir)

    logger.info(
        "All checks passed: normalize=%d, dedup_flagged=%d, consolidate=%d, tokens=%d",
        normalize_rows, dedup_rows, consol_rows, ledger.total_num_rows,
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        sys.exit(1)
