#!/usr/bin/env python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark parquet reading: pyarrow.dataset vs iter_parquet_row_groups.

Tests two scenarios that match real zephyr workloads:

1. **Full scan** — read every row (ETL pipelines: load_parquet).
2. **Scatter read** — select one (shard_idx, chunk_idx) out of many
   row groups, each containing exactly one pair (the real scatter layout).

Usage:
    uv run python tests/benchmark_parquet_reader.py --size-gb 1
    uv run python tests/benchmark_parquet_reader.py --size-gb 1 --modes dataset,iter_row_groups
"""

import gc
import logging
import os
import sys
import tempfile
import time
from typing import Any

import click
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_NUM_TEXT_COLS = 5
_NUM_SHARDS = 10
_CHUNKS_PER_SHARD = 3  # each source shard writes 3 sorted chunks


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _generate_batch(rows: int, shard_idx: int, chunk_idx: int, offset: int = 0) -> pa.RecordBatch:
    """One batch = one row group = one (shard_idx, chunk_idx) pair.

    Uses os.urandom text so Parquet compression can't hide the real size.
    """
    import base64
    import random

    ids = list(range(offset, offset + rows))
    scores = [random.random() * 100 for _ in range(rows)]
    arrays: dict[str, Any] = {
        "shard_idx": pa.array([shard_idx] * rows, type=pa.int32()),
        "chunk_idx": pa.array([chunk_idx] * rows, type=pa.int32()),
        "item": pa.array(
            [base64.b85encode(os.urandom(150)).decode("ascii") for _ in range(rows)],
            type=pa.string(),
        ),
        "id": pa.array(ids, type=pa.int64()),
        "score": pa.array(scores, type=pa.float64()),
    }
    for col_idx in range(_NUM_TEXT_COLS):
        arrays[f"text_{col_idx}"] = pa.array(
            [base64.b85encode(os.urandom(150)).decode("ascii") for _ in range(rows)],
            type=pa.string(),
        )
    return pa.RecordBatch.from_pydict(arrays)


def write_test_file(path: str, target_bytes: int, row_group_mb: int = 100) -> dict[str, int]:
    """Write a scatter-style Parquet file.

    Layout: ``_NUM_SHARDS`` shards x ``_CHUNKS_PER_SHARD`` chunks.
    Each row group has exactly one ``(shard_idx, chunk_idx)`` — statistics
    are exact (min == max) for both columns, just like the real scatter writer.

    Returns dict with file metadata for the benchmark harness.
    """
    logger.info("Generating scatter file: %s (target %.1f GB, rg ~%d MB)", path, target_bytes / 1e9, row_group_mb)

    # Calibrate rows per row group from a sample
    sample = _generate_batch(1000, 0, 0)
    sample_table = pa.Table.from_batches([sample])
    import tempfile as _tf

    with _tf.NamedTemporaryFile(suffix=".parquet") as tmp:
        pq.write_table(sample_table, tmp.name)
        disk_bytes_per_row = os.path.getsize(tmp.name) / len(sample_table)

    total_row_groups = _NUM_SHARDS * _CHUNKS_PER_SHARD
    rows_per_rg = max(1000, int(row_group_mb * 1e6 / disk_bytes_per_row))

    # Adjust rows_per_rg so total ≈ target_bytes
    total_rows_target = int(target_bytes / disk_bytes_per_row)
    rows_per_rg = max(1000, total_rows_target // total_row_groups)

    logger.info(
        "disk bytes/row: %.0f, rows/rg: %d, row_groups: %d (%d shards x %d chunks), total rows: %d",
        disk_bytes_per_row,
        rows_per_rg,
        total_row_groups,
        _NUM_SHARDS,
        _CHUNKS_PER_SHARD,
        rows_per_rg * total_row_groups,
    )

    writer = pq.ParquetWriter(path, sample.schema)
    total_rows = 0
    offset = 0
    for shard_idx in range(_NUM_SHARDS):
        for chunk_idx in range(_CHUNKS_PER_SHARD):
            batch = _generate_batch(rows_per_rg, shard_idx, chunk_idx, offset)
            writer.write_table(pa.Table.from_batches([batch]))
            total_rows += rows_per_rg
            offset += rows_per_rg
            if (shard_idx * _CHUNKS_PER_SHARD + chunk_idx + 1) % 5 == 0:
                logger.info("  wrote %d / %d rows", total_rows, rows_per_rg * total_row_groups)

    writer.close()
    file_size = os.path.getsize(path)
    pf = pq.ParquetFile(path)
    logger.info(
        "Wrote %s: %d rows, %d row groups, %.2f GB on disk",
        path,
        total_rows,
        pf.metadata.num_row_groups,
        file_size / 1e9,
    )
    return {
        "total_rows": total_rows,
        "num_row_groups": pf.metadata.num_row_groups,
        "rows_per_rg": rows_per_rg,
    }


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------


def _measure(fn) -> dict[str, Any]:
    gc.collect()
    pa.default_memory_pool().release_unused()
    mem_before = _mem()
    t0 = time.monotonic()
    count = fn()
    elapsed = time.monotonic() - t0
    mem_after = _mem()
    return {
        "rows": count,
        "wall_sec": elapsed,
        "rss_delta_gb": mem_after["rss"] - mem_before["rss"],
        "arrow_pool_mb": mem_after["arrow"],
    }


def _mem() -> dict[str, float]:
    return {
        "rss": psutil.Process().memory_info().rss / 1e9,
        "arrow": pa.default_memory_pool().bytes_allocated() / 1e6,
    }


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

# -- Full scan (load_parquet path) -----------------------------------------


def read_dataset_full(path: str, limit: int | None = None) -> dict[str, Any]:
    """Full scan via pyarrow.dataset.to_batches."""
    import pyarrow.dataset as pads

    def run():
        ds = pads.dataset(path, format="parquet")
        n = 0
        for batch in ds.to_batches():
            n += len(batch)
        return n

    return {"reader": "dataset_full", **_measure(run)}


def read_rowgroups_full(path: str, limit: int | None = None) -> dict[str, Any]:
    """Full scan via iter_parquet_row_groups."""
    from zephyr.readers import iter_parquet_row_groups

    def run():
        n = 0
        for table in iter_parquet_row_groups(path):
            n += len(table)
            del table
        return n

    return {"reader": "rowgroups_full", **_measure(run)}


# -- Scatter read (ScatterParquetIterator path) ----------------------------


def read_dataset_scatter(path: str, limit: int | None = None) -> dict[str, Any]:
    """Read one (shard, chunk) via dataset Scanner — the old scatter reader."""
    import pyarrow.compute as pc
    import pyarrow.dataset as pads

    target_shard = _NUM_SHARDS // 2
    target_chunk = _CHUNKS_PER_SHARD // 2

    def run():
        ds = pads.dataset(path, format="parquet")
        scanner = ds.scanner(
            columns=["item"],
            filter=(pc.field("shard_idx") == target_shard) & (pc.field("chunk_idx") == target_chunk),
            use_threads=False,
        )
        n = 0
        for batch in scanner.to_batches():
            n += len(batch)
        return n

    return {"reader": "dataset_scatter", **_measure(run)}


def read_rowgroups_scatter(path: str, limit: int | None = None) -> dict[str, Any]:
    """Read one (shard, chunk) via iter_parquet_row_groups — the new scatter reader.

    Uses equality_predicates for row-group-level skipping via statistics.
    No row_filter needed because each row group has exactly one (shard, chunk)
    pair — statistics are exact (min == max), so every row in a matching row
    group is a hit.
    """
    from zephyr.readers import iter_parquet_row_groups

    target_shard = _NUM_SHARDS // 2
    target_chunk = _CHUNKS_PER_SHARD // 2

    def run():
        n = 0
        for table in iter_parquet_row_groups(
            path,
            columns=["item"],
            equality_predicates={"shard_idx": target_shard, "chunk_idx": target_chunk},
        ):
            n += len(table)
            del table
        return n

    return {"reader": "rowgroups_scatter", **_measure(run)}


def read_rowgroups_scatter_no_skip(path: str, limit: int | None = None) -> dict[str, Any]:
    """Read one (shard, chunk) via iter_parquet_row_groups WITHOUT statistics skipping.

    Reads every row group and filters post-hoc. Shows the cost of not having
    row-group-level predicate pushdown.
    """
    import pyarrow.compute as pc
    from zephyr.readers import iter_parquet_row_groups

    target_shard = _NUM_SHARDS // 2
    target_chunk = _CHUNKS_PER_SHARD // 2

    def run():
        filt = (pc.field("shard_idx") == target_shard) & (pc.field("chunk_idx") == target_chunk)
        n = 0
        for table in iter_parquet_row_groups(path):
            table = table.filter(filt).select(["item"])
            n += len(table)
            del table
        return n

    return {"reader": "rowgroups_no_skip", **_measure(run)}


READERS = {
    "dataset_full": read_dataset_full,
    "rowgroups_full": read_rowgroups_full,
    "dataset_scatter": read_dataset_scatter,
    "rowgroups_scatter": read_rowgroups_scatter,
    "rowgroups_no_skip": read_rowgroups_scatter_no_skip,
}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_results(results: list[dict[str, Any]]) -> None:
    header = f"{'Reader':<22} {'Rows':>12} {'Wall (s)':>10} {'RSS delta (GB)':>15} {'Arrow pool (MB)':>16}"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['reader']:<22} {r['rows']:>12,} {r['wall_sec']:>10.2f} "
            f"{r['rss_delta_gb']:>15.3f} {r['arrow_pool_mb']:>16.1f}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_in_subprocess(mode: str, path: str, script_path: str) -> dict[str, Any]:
    import json
    import subprocess

    cmd = [sys.executable, script_path, "--file", path, "--modes", mode, "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess for {mode} failed:\n{proc.stderr}")
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"No JSON output from subprocess for {mode}:\n{proc.stdout}")


@click.command()
@click.option("--size-gb", default=2.0, help="Target on-disk file size in GB")
@click.option("--row-group-mb", default=100, help="Approximate row group size in MB")
@click.option("--modes", default=None, help="Comma-separated reader modes (default: all)")
@click.option("--keep-file", is_flag=True, help="Don't delete the test file after benchmark")
@click.option("--file", "file_path", default=None, help="Use an existing Parquet file")
@click.option("--json", "json_output", is_flag=True, hidden=True)
def main(
    size_gb: float,
    row_group_mb: int,
    modes: str | None,
    keep_file: bool,
    file_path: str | None,
    json_output: bool,
):
    """Benchmark parquet reading strategies.

    Each reader runs in a separate subprocess for isolated memory measurement.
    """
    if modes:
        selected = [m.strip() for m in modes.split(",")]
        for m in selected:
            if m not in READERS:
                raise click.BadParameter(f"Unknown mode: {m}. Available: {', '.join(READERS)}")
    else:
        selected = list(READERS)

    if json_output:
        assert file_path and len(selected) == 1
        import json

        result = READERS[selected[0]](file_path)
        print(json.dumps(result))
        return

    if file_path:
        path = file_path
        logger.info("Using existing file: %s", path)
        tmpdir = None
    else:
        tmpdir = tempfile.mkdtemp(prefix="bench_parquet_")
        path = os.path.join(tmpdir, "bench.parquet")
        write_test_file(path, int(size_gb * 1e9), row_group_mb=row_group_mb)

    script_path = os.path.abspath(__file__)

    try:
        results = []
        for mode in selected:
            logger.info("Running: %s (subprocess)", mode)
            result = _run_in_subprocess(mode, path, script_path)
            results.append(result)
            logger.info(
                "  %s: %d rows in %.2fs, RSS delta %.3f GB, Arrow pool %.1f MB",
                result["reader"],
                result["rows"],
                result["wall_sec"],
                result["rss_delta_gb"],
                result["arrow_pool_mb"],
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
