#!/usr/bin/env python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark parquet reading: pyarrow.dataset vs iter_parquet_row_groups.

Compares memory usage (RSS, Arrow pool) and wall-time for reading
large Parquet files via the old pyarrow.dataset API and the new
row-group-by-row-group reader.

Usage:
    uv run python tests/benchmark_parquet_reader.py                     # defaults: 2 GB, all modes
    uv run python tests/benchmark_parquet_reader.py --size-gb 4         # 4 GB file
    uv run python tests/benchmark_parquet_reader.py --modes dataset     # only dataset API
    uv run python tests/benchmark_parquet_reader.py --keep-file         # don't delete the test file
"""

import gc
import logging
import os
import tempfile
import time
from typing import Any

import click
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Number of string columns with random-ish text to inflate row size
_NUM_TEXT_COLS = 5


def _generate_batch(batch_idx: int, rows_per_batch: int) -> pa.RecordBatch:
    """Generate a single Arrow RecordBatch of synthetic data.

    Uses os.urandom for text columns so Parquet/Snappy cannot compress
    them away — on-disk size closely tracks uncompressed size.
    """
    import base64
    import random

    base = batch_idx * rows_per_batch
    ids = list(range(base, base + rows_per_batch))
    scores = [random.random() * 100 for _ in range(rows_per_batch)]
    category = [random.choice(["A", "B", "C", "D"]) for _ in range(rows_per_batch)]

    arrays: dict[str, Any] = {
        "id": pa.array(ids, type=pa.int64()),
        "score": pa.array(scores, type=pa.float64()),
        "category": pa.array(category, type=pa.string()),
    }
    for col_idx in range(_NUM_TEXT_COLS):
        # ~200 bytes of base64-encoded random data per cell — incompressible
        texts = [base64.b85encode(os.urandom(150)).decode("ascii") for _ in range(rows_per_batch)]
        arrays[f"text_{col_idx}"] = pa.array(texts, type=pa.string())

    return pa.RecordBatch.from_pydict(arrays)


def write_test_file(path: str, target_bytes: int, row_group_mb: int = 100) -> int:
    """Write a Parquet file of approximately target_bytes. Returns actual row count.

    Args:
        path: Output file path.
        target_bytes: Target on-disk file size in bytes.
        row_group_mb: Approximate size per row group in MB.
    """
    logger.info("Generating test file: %s (target %.1f GB, row_group ~%d MB)", path, target_bytes / 1e9, row_group_mb)

    # Estimate rows needed from a sample batch written to disk
    sample = _generate_batch(0, 1000)
    sample_table = pa.Table.from_batches([sample])

    # Write sample to a temp file to measure actual on-disk bytes per row
    import tempfile as _tf

    with _tf.NamedTemporaryFile(suffix=".parquet") as tmp:
        pq.write_table(sample_table, tmp.name)
        disk_bytes_per_row = os.path.getsize(tmp.name) / len(sample_table)

    estimated_rows = int(target_bytes / disk_bytes_per_row)
    row_group_rows = max(1000, int(row_group_mb * 1e6 / disk_bytes_per_row))
    logger.info(
        "Disk bytes/row: %.0f, total rows: %d, rows/row_group: %d",
        disk_bytes_per_row,
        estimated_rows,
        row_group_rows,
    )

    total_rows = 0
    writer = pq.ParquetWriter(path, sample.schema)
    batch_size = min(row_group_rows, 50_000)
    batch_idx = 0

    while total_rows < estimated_rows:
        remaining = estimated_rows - total_rows
        n = min(batch_size, remaining)
        batch = _generate_batch(batch_idx, n)
        writer.write_batch(batch)
        total_rows += n
        batch_idx += 1
        if batch_idx % 10 == 0:
            logger.info("  wrote %d / %d rows (%.0f%%)", total_rows, estimated_rows, 100 * total_rows / estimated_rows)

    writer.close()
    file_size = os.path.getsize(path)
    logger.info("Wrote %s: %d rows, %.2f GB on disk", path, total_rows, file_size / 1e9)
    return total_rows


def _get_memory_stats() -> dict[str, float]:
    proc = psutil.Process()
    rss_gb = proc.memory_info().rss / 1e9
    arrow_mb = pa.default_memory_pool().bytes_allocated() / 1e6
    return {"rss_gb": rss_gb, "arrow_pool_mb": arrow_mb}


def _reset_memory():
    gc.collect()
    pa.default_memory_pool().release_unused()


# ---------------------------------------------------------------------------
# Reader implementations
# ---------------------------------------------------------------------------


def read_dataset_api(path: str, limit: int | None = None) -> dict[str, Any]:
    """Read via pyarrow.dataset.to_batches (the old approach)."""
    import pyarrow.dataset as pads

    _reset_memory()
    mem_before = _get_memory_stats()
    t0 = time.monotonic()

    dataset = pads.dataset(path, format="parquet")
    count = 0
    for batch in dataset.to_batches():
        count += len(batch)
        if limit and count >= limit:
            break

    elapsed = time.monotonic() - t0
    mem_after = _get_memory_stats()
    return {
        "reader": "dataset",
        "rows": count,
        "wall_sec": elapsed,
        "rss_before_gb": mem_before["rss_gb"],
        "rss_after_gb": mem_after["rss_gb"],
        "rss_delta_gb": mem_after["rss_gb"] - mem_before["rss_gb"],
        "arrow_pool_before_mb": mem_before["arrow_pool_mb"],
        "arrow_pool_after_mb": mem_after["arrow_pool_mb"],
    }


def read_parquet_file_api(path: str, limit: int | None = None) -> dict[str, Any]:
    """Read via pq.ParquetFile.read_row_group (the new approach)."""
    _reset_memory()
    mem_before = _get_memory_stats()
    t0 = time.monotonic()

    pf = pq.ParquetFile(path)
    count = 0
    for i in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(i)
        count += len(table)
        del table
        if limit and count >= limit:
            break

    elapsed = time.monotonic() - t0
    mem_after = _get_memory_stats()
    return {
        "reader": "parquet_file",
        "rows": count,
        "wall_sec": elapsed,
        "rss_before_gb": mem_before["rss_gb"],
        "rss_after_gb": mem_after["rss_gb"],
        "rss_delta_gb": mem_after["rss_gb"] - mem_before["rss_gb"],
        "arrow_pool_before_mb": mem_before["arrow_pool_mb"],
        "arrow_pool_after_mb": mem_after["arrow_pool_mb"],
    }


def read_iter_row_groups(path: str, limit: int | None = None) -> dict[str, Any]:
    """Read via iter_parquet_row_groups (the shared utility)."""
    from zephyr.readers import iter_parquet_row_groups

    _reset_memory()
    mem_before = _get_memory_stats()
    t0 = time.monotonic()

    count = 0
    for table in iter_parquet_row_groups(path):
        count += len(table)
        del table
        if limit and count >= limit:
            break

    elapsed = time.monotonic() - t0
    mem_after = _get_memory_stats()
    return {
        "reader": "iter_row_groups",
        "rows": count,
        "wall_sec": elapsed,
        "rss_before_gb": mem_before["rss_gb"],
        "rss_after_gb": mem_after["rss_gb"],
        "rss_delta_gb": mem_after["rss_gb"] - mem_before["rss_gb"],
        "arrow_pool_before_mb": mem_before["arrow_pool_mb"],
        "arrow_pool_after_mb": mem_after["arrow_pool_mb"],
    }


def read_dataset_with_filter(path: str, limit: int | None = None) -> dict[str, Any]:
    """Read via pyarrow.dataset with a filter (old approach)."""
    import pyarrow.compute as pc
    import pyarrow.dataset as pads

    _reset_memory()
    mem_before = _get_memory_stats()
    t0 = time.monotonic()

    dataset = pads.dataset(path, format="parquet")
    filt = pc.field("score") > 50.0
    count = 0
    for batch in dataset.to_batches(filter=filt, columns=["id", "score"]):
        count += len(batch)
        if limit and count >= limit:
            break

    elapsed = time.monotonic() - t0
    mem_after = _get_memory_stats()
    return {
        "reader": "dataset+filter",
        "rows": count,
        "wall_sec": elapsed,
        "rss_before_gb": mem_before["rss_gb"],
        "rss_after_gb": mem_after["rss_gb"],
        "rss_delta_gb": mem_after["rss_gb"] - mem_before["rss_gb"],
        "arrow_pool_before_mb": mem_before["arrow_pool_mb"],
        "arrow_pool_after_mb": mem_after["arrow_pool_mb"],
    }


def read_row_groups_with_filter(path: str, limit: int | None = None) -> dict[str, Any]:
    """Read via iter_parquet_row_groups with a filter (new approach)."""
    import pyarrow.compute as pc

    from zephyr.readers import iter_parquet_row_groups

    _reset_memory()
    mem_before = _get_memory_stats()
    t0 = time.monotonic()

    filt = pc.field("score") > 50.0
    count = 0
    for table in iter_parquet_row_groups(path, columns=["id", "score"], row_filter=filt):
        count += len(table)
        del table
        if limit and count >= limit:
            break

    elapsed = time.monotonic() - t0
    mem_after = _get_memory_stats()
    return {
        "reader": "row_groups+filter",
        "rows": count,
        "wall_sec": elapsed,
        "rss_before_gb": mem_before["rss_gb"],
        "rss_after_gb": mem_after["rss_gb"],
        "rss_delta_gb": mem_after["rss_gb"] - mem_before["rss_gb"],
        "arrow_pool_before_mb": mem_before["arrow_pool_mb"],
        "arrow_pool_after_mb": mem_after["arrow_pool_mb"],
    }


READERS = {
    "dataset": read_dataset_api,
    "parquet_file": read_parquet_file_api,
    "iter_row_groups": read_iter_row_groups,
    "dataset+filter": read_dataset_with_filter,
    "row_groups+filter": read_row_groups_with_filter,
}


def print_results(results: list[dict[str, Any]]) -> None:
    header = f"{'Reader':<22} {'Rows':>12} {'Wall (s)':>10} {'RSS delta (GB)':>15} {'Arrow pool (MB)':>16}"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['reader']:<22} {r['rows']:>12,} {r['wall_sec']:>10.2f} "
            f"{r['rss_delta_gb']:>15.3f} {r['arrow_pool_after_mb']:>16.1f}"
        )
    print()


@click.command()
@click.option("--size-gb", default=2.0, help="Target on-disk file size in GB")
@click.option("--row-group-mb", default=100, help="Approximate row group size in MB")
@click.option("--modes", default=None, help="Comma-separated reader modes (default: all)")
@click.option("--keep-file", is_flag=True, help="Don't delete the test file after benchmark")
@click.option("--file", "file_path", default=None, help="Use an existing Parquet file instead of generating one")
def main(size_gb: float, row_group_mb: int, modes: str | None, keep_file: bool, file_path: str | None):
    """Benchmark parquet reading strategies."""
    if modes:
        selected = [m.strip() for m in modes.split(",")]
        for m in selected:
            if m not in READERS:
                raise click.BadParameter(f"Unknown mode: {m}. Available: {', '.join(READERS)}")
    else:
        selected = list(READERS)

    if file_path:
        path = file_path
        logger.info("Using existing file: %s", path)
        tmpdir = None
    else:
        tmpdir = tempfile.mkdtemp(prefix="bench_parquet_")
        path = os.path.join(tmpdir, "bench.parquet")
        write_test_file(path, int(size_gb * 1e9), row_group_mb=row_group_mb)

    try:
        results = []
        for mode in selected:
            logger.info("Running reader: %s", mode)
            _reset_memory()
            result = READERS[mode](path)
            results.append(result)
            logger.info(
                "  %s: %d rows in %.2fs, RSS delta %.3f GB, Arrow pool %.1f MB",
                result["reader"],
                result["rows"],
                result["wall_sec"],
                result["rss_delta_gb"],
                result["arrow_pool_after_mb"],
            )

        print_results(results)
    finally:
        if tmpdir and not keep_file:
            import shutil

            shutil.rmtree(tmpdir)
            logger.info("Cleaned up %s", tmpdir)
        elif tmpdir:
            logger.info("Kept test file at %s", path)


if __name__ == "__main__":
    main()
