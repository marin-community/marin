# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the outputs of the tier-1 datakit ferry.

Run after the Iris job for the tier-1 ferry has completed, in the ferry's region.
``FERRY_OUTPUT_PREFIX`` must be set to the output prefix the ferry logged (also the
``marin_prefix`` of its run status). Leave ``MARIN_PREFIX`` as the regional bucket:
artifact paths are stored relative to it.

Checks the persisted output invariants:
  download (14 files, ~9.7M rows)
  → normalize (106 files under outputs/main, ~9.3M rows)
  → store (every normalized record enters the store, fuzzy dedup removes some
    of them, and the finished leaf caches of its buckets hold exactly the
    records it keeps)

Successful ferry completion covers the reference stages between normalize and
the store, which each write their own report under ``datakit/report``.
"""

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from levanter.store.cache import CacheLedger
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging

from experiments.datakit.store.datakit_store import ClusteredStoreData

logger = logging.getLogger(__name__)

# --- Download: fineweb-edu sample/10BT has exactly 14 parquet shards ---
DOWNLOAD_EXPECTED_FILES = 14
DOWNLOAD_MIN_ROWS = 9_000_000  # observed: 9,672,101

# --- Normalize: scatter produces 106 output files ---
NORMALIZE_EXPECTED_FILES = 106
NORMALIZE_MIN_ROWS = 8_000_000  # observed: 9,268,156
NORMALIZE_REQUIRED_COLUMNS = frozenset({"id", "text", "url", "source_id", "token_count"})

# --- Store: duplicate and contamination filters drop some records, never most ---
STORE_DROP_MAX_FRACTION = 0.50
LEDGER_READ_THREADS = 32


def _list_parquet(path: str) -> list[str]:
    """Glob for parquet files under path; raise if none found."""
    files = [str(m) for m in StoragePath(f"{path}/*.parquet").glob()]
    if not files:
        raise SystemExit(f"No parquet files found under {path}")
    return files


def _count_parquet_rows(files: list[str]) -> int:
    """Sum row counts from parquet file metadata (no data read)."""
    total = 0
    for path in files:
        with StoragePath(path).open("rb") as f:
            total += pq.ParquetFile(f).metadata.num_rows
    return total


def _check_schema(path: str, required: frozenset[str]) -> list[str]:
    """Verify a parquet file contains the required columns. Returns actual column names."""
    with StoragePath(path).open("rb") as f:
        names = pq.ParquetFile(f).schema_arrow.names
    missing = required - set(names)
    if missing:
        raise SystemExit(f"Schema mismatch in {path}: missing {missing}")
    return names


def _validate_download(base: str) -> int:
    dl_path = f"{base}/download"
    files = [str(m) for m in StoragePath(f"{dl_path}/**/*.parquet").glob()]
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
    # Normalize writes main records to {base}/normalize/outputs/main and duplicates
    # (when exact dedup is enabled) to {base}/normalize/outputs/dups. We load the
    # artifact to resolve the paths rather than hard-coding the layout.
    normalized = read_artifact(f"{base}/normalize", NormalizedData)
    files = _list_parquet(normalized.main_output_dir)
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


def _bucket_records(bucket_path: str) -> int:
    """Return the records in a bucket after checking each leaf cache's own ledger."""
    ledger = CacheLedger.load(bucket_path)
    if not ledger.is_finished:
        raise SystemExit(f"Store: bucket ledger not finished: {bucket_path}")

    def leaf_rows(shard: str) -> int:
        leaf = CacheLedger.load(prefix_join(bucket_path, shard))
        if not leaf.is_finished:
            raise SystemExit(f"Store: leaf cache not finished: {bucket_path}/{shard}")
        return leaf.total_num_rows

    shards = sorted(ledger.shard_rows)
    with ThreadPoolExecutor(LEDGER_READ_THREADS) as pool:
        rows = dict(zip(shards, pool.map(leaf_rows, shards), strict=True))
    if rows != ledger.shard_rows:
        raise SystemExit(f"Store: leaf cache rows differ from the bucket ledger in {bucket_path}")
    return ledger.total_num_rows


def _validate_store(base: str, normalize_rows: int) -> int:
    store_dirs = [str(m) for m in StoragePath(prefix_join(base, "datakit/store_*")).glob()]
    if len(store_dirs) != 1:
        raise SystemExit(f"Store: expected one store under {base}/datakit, got {store_dirs}")
    store = read_artifact(store_dirs[0], ClusteredStoreData)

    records_in = int(store.counters["datakit_store/records_in"])
    records_out = int(store.counters["datakit_store/records_out"])
    if records_in != normalize_rows:
        raise SystemExit(f"Store: read {records_in} records, normalize wrote {normalize_rows}")
    if not 0 < records_out < records_in:
        raise SystemExit(f"Store: kept {records_out} of {records_in} records; expected some but not all")
    dropped_fraction = 1 - records_out / records_in
    if dropped_fraction > STORE_DROP_MAX_FRACTION:
        raise SystemExit(f"Store: dropped {dropped_fraction:.1%} of records (max {STORE_DROP_MAX_FRACTION:.0%})")

    if int(store.counters["datakit_store/fuzzy_duplicate_dropped"]) <= 0:
        raise SystemExit("Store: fuzzy dedup removed no records")

    bucket_records = sum(_bucket_records(bucket.path) for bucket in store.buckets)
    if bucket_records != records_out:
        raise SystemExit(f"Store: bucket caches hold {bucket_records} records, store kept {records_out}")

    logger.info(
        "Store OK: %d records in %d buckets (%.2f%% dropped: %d fuzzy, %d exact, %d contaminated)",
        records_out,
        len(store.buckets),
        100 * dropped_fraction,
        store.counters["datakit_store/fuzzy_duplicate_dropped"],
        store.counters["datakit_store/exact_duplicate_dropped"],
        store.counters["datakit_store/contaminated_dropped"],
    )
    return records_out


def main() -> None:
    configure_logging()
    base = os.environ["FERRY_OUTPUT_PREFIX"].rstrip("/")

    download_rows = _validate_download(base)
    normalize_rows = _validate_normalize(base, download_rows)
    store_rows = _validate_store(base, normalize_rows)

    logger.info("All checks passed: download=%d → normalize=%d → store=%d", download_rows, normalize_rows, store_rows)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        logger.error("Validation failed: %s", exc)
        sys.exit(1)
