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

# Number of string columns with random-ish text to inflate row size
_NUM_TEXT_COLS = 5


def _generate_batch(
    batch_idx: int,
    rows_per_batch: int,
    shard_idx: int = 0,
    chunk_idx: int = 0,
) -> pa.RecordBatch:
    """Generate a single Arrow RecordBatch of synthetic data.

    Uses os.urandom for text columns so Parquet/Snappy cannot compress
    them away — on-disk size closely tracks uncompressed size.

    ``shard_idx`` and ``chunk_idx`` are constant for the entire batch,
    mimicking scatter files where each row group belongs to exactly one
    (shard, chunk) pair.
    """
    import base64
    import random

    base = batch_idx * rows_per_batch
    ids = list(range(base, base + rows_per_batch))
    scores = [random.random() * 100 for _ in range(rows_per_batch)]

    arrays: dict[str, Any] = {
        "id": pa.array(ids, type=pa.int64()),
        "shard_idx": pa.array([shard_idx] * rows_per_batch, type=pa.int32()),
        "chunk_idx": pa.array([chunk_idx] * rows_per_batch, type=pa.int32()),
        "score": pa.array(scores, type=pa.float64()),
    }
    for col_idx in range(_NUM_TEXT_COLS):
        # ~200 bytes of base64-encoded random data per cell — incompressible
        texts = [base64.b85encode(os.urandom(150)).decode("ascii") for _ in range(rows_per_batch)]
        arrays[f"text_{col_idx}"] = pa.array(texts, type=pa.string())

    return pa.RecordBatch.from_pydict(arrays)


# Scatter-style layout: single shard_idx per file, N chunks per row group
_SHARD_IDX = 0
_CHUNKS_PER_ROW_GROUP = 3


def write_test_file(path: str, target_bytes: int, row_group_mb: int = 100) -> int:
    """Write a scatter-style Parquet file of approximately target_bytes.

    Layout mirrors zephyr scatter files:
    - Single ``shard_idx`` (= 0) for the whole file.
    - Each row group contains ``_CHUNKS_PER_ROW_GROUP`` consecutive
      ``chunk_idx`` values, each written as a separate batch (so Parquet
      statistics for chunk_idx span a small range per row group).

    Args:
        path: Output file path.
        target_bytes: Target on-disk file size in bytes.
        row_group_mb: Approximate size per row group in MB.
    """
    logger.info("Generating test file: %s (target %.1f GB, row_group ~%d MB)", path, target_bytes / 1e9, row_group_mb)

    # Estimate rows needed from a sample batch written to disk
    sample = _generate_batch(0, 1000)
    sample_table = pa.Table.from_batches([sample])

    import tempfile as _tf

    with _tf.NamedTemporaryFile(suffix=".parquet") as tmp:
        pq.write_table(sample_table, tmp.name)
        disk_bytes_per_row = os.path.getsize(tmp.name) / len(sample_table)

    estimated_rows = int(target_bytes / disk_bytes_per_row)
    row_group_rows = max(1000, int(row_group_mb * 1e6 / disk_bytes_per_row))
    rows_per_chunk = row_group_rows // _CHUNKS_PER_ROW_GROUP
    total_chunks = max(1, estimated_rows // rows_per_chunk)

    logger.info(
        "Disk bytes/row: %.0f, total rows: %d, rows/row_group: %d, "
        "chunks/row_group: %d, rows/chunk: %d, total chunks: %d",
        disk_bytes_per_row,
        estimated_rows,
        row_group_rows,
        _CHUNKS_PER_ROW_GROUP,
        rows_per_chunk,
        total_chunks,
    )

    total_rows = 0
    writer = pq.ParquetWriter(path, sample.schema)
    batch_idx = 0

    # Write _CHUNKS_PER_ROW_GROUP chunk batches as one row group (via write_table).
    # Each row group thus has chunk_idx values [base, base+1, base+2], and
    # Parquet statistics for chunk_idx span that small range.
    rg_batches: list[pa.RecordBatch] = []
    for chunk_idx in range(total_chunks):
        batch = _generate_batch(batch_idx, rows_per_chunk, shard_idx=_SHARD_IDX, chunk_idx=chunk_idx)
        rg_batches.append(batch)
        batch_idx += 1
        if len(rg_batches) == _CHUNKS_PER_ROW_GROUP:
            table = pa.Table.from_batches(rg_batches)
            writer.write_table(table)
            total_rows += len(table)
            rg_batches = []
            if batch_idx % 10 == 0:
                logger.info(
                    "  wrote %d / %d rows (%.0f%%)", total_rows, estimated_rows, 100 * total_rows / estimated_rows
                )
    if rg_batches:
        table = pa.Table.from_batches(rg_batches)
        writer.write_table(table)
        total_rows += len(table)

    writer.close()
    file_size = os.path.getsize(path)
    pf = pq.ParquetFile(path)
    logger.info(
        "Wrote %s: %d rows, %d row groups, %d total chunks, %.2f GB on disk",
        path,
        total_rows,
        pf.metadata.num_row_groups,
        total_chunks,
        file_size / 1e9,
    )
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


def _pick_target_chunk(path: str) -> int:
    """Pick a chunk_idx near the middle of the file for a representative read."""
    pf = pq.ParquetFile(path)
    # Read chunk_idx stats from the middle row group
    mid = pf.metadata.num_row_groups // 2
    rg = pf.metadata.row_group(mid)
    for col_idx in range(rg.num_columns):
        if rg.column(col_idx).path_in_schema == "chunk_idx":
            return rg.column(col_idx).statistics.min
    return 0


def read_dataset_scatter(path: str, limit: int | None = None) -> dict[str, Any]:
    """Read one (shard_idx, chunk_idx) via pyarrow.dataset Scanner — the old scatter path."""
    import pyarrow.compute as pc
    import pyarrow.dataset as pads

    target_chunk = _pick_target_chunk(path)

    _reset_memory()
    mem_before = _get_memory_stats()
    t0 = time.monotonic()

    dataset = pads.dataset(path, format="parquet")
    scanner = dataset.scanner(
        columns=["id", "score"],
        filter=(pc.field("shard_idx") == _SHARD_IDX) & (pc.field("chunk_idx") == target_chunk),
        use_threads=False,
    )
    count = 0
    for batch in scanner.to_batches():
        count += len(batch)
        if limit and count >= limit:
            break

    elapsed = time.monotonic() - t0
    mem_after = _get_memory_stats()
    return {
        "reader": "dataset+scatter",
        "rows": count,
        "wall_sec": elapsed,
        "rss_before_gb": mem_before["rss_gb"],
        "rss_after_gb": mem_after["rss_gb"],
        "rss_delta_gb": mem_after["rss_gb"] - mem_before["rss_gb"],
        "arrow_pool_before_mb": mem_before["arrow_pool_mb"],
        "arrow_pool_after_mb": mem_after["arrow_pool_mb"],
    }


def read_row_groups_scatter(path: str, limit: int | None = None) -> dict[str, Any]:
    """Read one (shard_idx, chunk_idx) via iter_parquet_row_groups with equality_predicates."""
    import pyarrow.compute as pc

    from zephyr.readers import iter_parquet_row_groups

    target_chunk = _pick_target_chunk(path)

    _reset_memory()
    mem_before = _get_memory_stats()
    t0 = time.monotonic()

    # equality_predicates skips non-matching row groups via statistics.
    # row_filter does the row-level filtering within qualifying row groups
    # (needed when multiple chunk_idx values share a row group).
    count = 0
    for table in iter_parquet_row_groups(
        path,
        columns=["id", "score"],
        equality_predicates={"shard_idx": _SHARD_IDX, "chunk_idx": target_chunk},
        row_filter=(pc.field("shard_idx") == _SHARD_IDX) & (pc.field("chunk_idx") == target_chunk),
    ):
        count += len(table)
        del table
        if limit and count >= limit:
            break

    elapsed = time.monotonic() - t0
    mem_after = _get_memory_stats()
    return {
        "reader": "row_groups+scatter",
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
    "dataset+scatter": read_dataset_scatter,
    "row_groups+scatter": read_row_groups_scatter,
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


def _run_single(mode: str, path: str) -> dict[str, Any]:
    """Run a single reader in the current process and return its result dict."""
    result = READERS[mode](path)
    return result


def _run_in_subprocess(mode: str, path: str, script_path: str) -> dict[str, Any]:
    """Run a single reader in a fresh subprocess for accurate memory measurement."""
    import json
    import subprocess

    cmd = [
        sys.executable,
        script_path,
        "--file",
        path,
        "--modes",
        mode,
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess for {mode} failed:\n{proc.stderr}")
    # Parse the last JSON line from stdout
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
@click.option("--file", "file_path", default=None, help="Use an existing Parquet file instead of generating one")
@click.option("--json", "json_output", is_flag=True, hidden=True, help="Print single result as JSON (internal)")
def main(
    size_gb: float,
    row_group_mb: int,
    modes: str | None,
    keep_file: bool,
    file_path: str | None,
    json_output: bool,
):
    """Benchmark parquet reading strategies.

    Each reader runs in a separate subprocess for accurate, isolated
    memory measurements.
    """
    if modes:
        selected = [m.strip() for m in modes.split(",")]
        for m in selected:
            if m not in READERS:
                raise click.BadParameter(f"Unknown mode: {m}. Available: {', '.join(READERS)}")
    else:
        selected = list(READERS)

    # --json mode: run a single reader in-process and print result as JSON.
    # Used by the subprocess spawner above.
    if json_output:
        assert file_path, "--json requires --file"
        assert len(selected) == 1, "--json requires exactly one --modes"
        import json

        result = _run_single(selected[0], file_path)
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
            logger.info("Running reader: %s (subprocess)", mode)
            result = _run_in_subprocess(mode, path, script_path)
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
