# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate datakit smoke ferry outputs.

Run after the iris job for the datakit smoke ferry has completed.
``MARIN_PREFIX`` must be set to the GCS prefix the ferry wrote to
(read from ``ferry_run_status.json`` by the workflow).

Checks the full pipeline chain:
  download (14 files, ~9.7M rows)
  → normalize (106 files, ~9.3M rows, some rows filtered)
  → dedup (106 attribute files, ~286K flagged dups)
  → consolidate (106 files, normalize - dedup rows)
  → tokenize (cache ledger rows == consolidate rows)
"""

import logging
import os
import sys

import pyarrow.parquet as pq
from levanter.store.cache import CacheLedger
from marin.utils import fsspec_glob
from rigging.filesystem import url_to_fs
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

# --- Download: fineweb-edu sample/10BT has exactly 14 parquet shards ---
DOWNLOAD_EXPECTED_FILES = 14
DOWNLOAD_MIN_ROWS = 9_000_000  # observed: 9,672,101

# --- Normalize: scatter produces 106 output files ---
NORMALIZE_EXPECTED_FILES = 106
NORMALIZE_MIN_ROWS = 8_000_000  # observed: 9,268,156
NORMALIZE_REQUIRED_COLUMNS = {"id", "text", "url", "source_id", "token_count"}

# --- Dedup: one attribute file per normalize shard ---
DEDUP_EXPECTED_FILES = 106
DEDUP_REQUIRED_COLUMNS = {"id", "attributes"}
# Dedup should flag a non-trivial fraction but not everything.
# Observed: 286,263 / 9,268,156 = 3.1%.
DEDUP_MIN_FRACTION = 0.005  # at least 0.5% flagged
DEDUP_MAX_FRACTION = 0.50  # at most 50% flagged

# --- Consolidate: same file count, strictly fewer rows than normalize ---
CONSOLIDATE_REQUIRED_COLUMNS = NORMALIZE_REQUIRED_COLUMNS


def _list_parquet(path: str) -> list[str]:
    """Glob for parquet files under path; raise if none found."""
    files = fsspec_glob(f"{path}/*.parquet")
    if not files:
        raise SystemExit(f"No parquet files found under {path}")
    return files


def _count_parquet_rows(files: list[str]) -> int:
    """Sum row counts from parquet file metadata (no data read)."""
    fs, _ = url_to_fs(files[0])
    total = 0
    for path in files:
        with fs.open(path, "rb") as f:
            total += pq.ParquetFile(f).metadata.num_rows
    return total


def _check_schema(path: str, required: set[str]) -> list[str]:
    """Verify a parquet file contains the required columns. Returns actual column names."""
    fs, _ = url_to_fs(path)
    with fs.open(path, "rb") as f:
        names = pq.ParquetFile(f).schema_arrow.names
    missing = required - set(names)
    if missing:
        raise SystemExit(f"Schema mismatch in {path}: missing {missing}")
    return names


def _validate_download(base: str) -> int:
    dl_path = f"{base}/download"
    files = fsspec_glob(f"{dl_path}/**/*.parquet")
    if not files:
        raise SystemExit(f"No download parquet files under {dl_path}")
    if len(files) != DOWNLOAD_EXPECTED_FILES:
        raise SystemExit(f"Download: expected {DOWNLOAD_EXPECTED_FILES} files, got {len(files)}")

    rows = _count_parquet_rows(files)
    if rows < DOWNLOAD_MIN_ROWS:
        raise SystemExit(f"Download: expected >= {DOWNLOAD_MIN_ROWS} rows, got {rows}")

    logger.info("Download OK: %d files, %d rows", len(files), rows)
    return rows


def _validate_normalize(base: str, download_rows: int) -> int:
    files = _list_parquet(f"{base}/normalize")
    if len(files) != NORMALIZE_EXPECTED_FILES:
        raise SystemExit(f"Normalize: expected {NORMALIZE_EXPECTED_FILES} files, got {len(files)}")

    _check_schema(files[0], NORMALIZE_REQUIRED_COLUMNS)

    rows = _count_parquet_rows(files)
    if rows < NORMALIZE_MIN_ROWS:
        raise SystemExit(f"Normalize: expected >= {NORMALIZE_MIN_ROWS} rows, got {rows}")
    if rows > download_rows:
        raise SystemExit(
            f"Normalize: {rows} rows > download {download_rows} rows — "
            "normalize cannot produce more rows than download"
        )

    logger.info("Normalize OK: %d files, %d rows (%.1f%% of download)", len(files), rows, 100 * rows / download_rows)
    return rows


def _validate_dedup(base: str, normalize_rows: int) -> int:
    files = _list_parquet(f"{base}/dedup/data")
    if len(files) != DEDUP_EXPECTED_FILES:
        raise SystemExit(f"Dedup: expected {DEDUP_EXPECTED_FILES} files, got {len(files)}")

    _check_schema(files[0], DEDUP_REQUIRED_COLUMNS)

    flagged = _count_parquet_rows(files)
    fraction = flagged / normalize_rows if normalize_rows > 0 else 0

    if fraction < DEDUP_MIN_FRACTION:
        raise SystemExit(
            f"Dedup: only {flagged} dups flagged ({fraction:.1%} of {normalize_rows}) — "
            f"expected >= {DEDUP_MIN_FRACTION:.1%}"
        )
    if fraction > DEDUP_MAX_FRACTION:
        raise SystemExit(
            f"Dedup: {flagged} dups flagged ({fraction:.1%} of {normalize_rows}) — "
            f"expected <= {DEDUP_MAX_FRACTION:.0%}, something is wrong"
        )

    logger.info(
        "Dedup OK: %d files, %d flagged duplicates (%.1f%% of %d docs)",
        len(files),
        flagged,
        100 * fraction,
        normalize_rows,
    )
    return flagged


def _validate_consolidate(base: str, normalize_rows: int, dedup_rows: int) -> int:
    files = _list_parquet(f"{base}/consolidate")

    _check_schema(files[0], CONSOLIDATE_REQUIRED_COLUMNS)

    rows = _count_parquet_rows(files)

    # Core invariant: consolidate must have strictly fewer rows than normalize
    if rows >= normalize_rows:
        raise SystemExit(
            f"Consolidate: {rows} rows >= normalize {normalize_rows} rows — " "dedup removal did not reduce row count"
        )

    # The expected count is exactly normalize - dedup
    expected = normalize_rows - dedup_rows
    if rows != expected:
        raise SystemExit(
            f"Consolidate: {rows} rows, expected exactly {expected} "
            f"(normalize {normalize_rows} - dedup {dedup_rows})"
        )

    removed = normalize_rows - rows
    logger.info(
        "Consolidate OK: %d files, %d rows (removed %d, %.1f%%)",
        len(files),
        rows,
        removed,
        100 * removed / normalize_rows,
    )
    return rows


def _validate_tokens(base: str, consolidate_rows: int) -> int:
    train_dir = f"{base}/tokens/train"
    ledger = CacheLedger.load(train_dir)
    if not ledger.is_finished:
        raise SystemExit(f"Tokenizer cache ledger not finished: {train_dir}")
    if ledger.total_num_rows <= 0:
        raise SystemExit(f"Tokenizer cache ledger has 0 rows: {train_dir}")

    # Token rows should match consolidate rows exactly
    if ledger.total_num_rows != consolidate_rows:
        raise SystemExit(
            f"Tokens: {ledger.total_num_rows} rows != consolidate {consolidate_rows} rows — "
            "tokenizer should process every consolidated document"
        )

    logger.info("Tokens OK: %d rows (matches consolidate)", ledger.total_num_rows)
    return ledger.total_num_rows


def main() -> None:
    configure_logging()
    prefix = os.environ.get("MARIN_PREFIX")
    if not prefix:
        raise SystemExit("MARIN_PREFIX must be set to the GCS prefix the ferry wrote to")
    run_id = os.environ["SMOKE_RUN_ID"]
    prefix = prefix.rstrip("/")
    base = f"{prefix}/datakit-smoke/{run_id}"

    download_rows = _validate_download(base)
    normalize_rows = _validate_normalize(base, download_rows)
    dedup_rows = _validate_dedup(base, normalize_rows)
    consolidate_rows = _validate_consolidate(base, normalize_rows, dedup_rows)
    token_rows = _validate_tokens(base, consolidate_rows)

    logger.info(
        "All checks passed: download=%d → normalize=%d → dedup_flagged=%d → consolidate=%d → tokens=%d",
        download_rows,
        normalize_rows,
        dedup_rows,
        consolidate_rows,
        token_rows,
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        sys.exit(1)
