#!/usr/bin/env python
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark comparing synchronous vs threaded Levanter cache writes in Zephyr.

Generates synthetic tokenized records through a map-only pipeline and measures
throughput for direct (synchronous) writes vs ThreadedBatchWriter.

Usage:
    uv run python lib/zephyr/tests/benchmark_threaded_writer.py
    uv run python lib/zephyr/tests/benchmark_threaded_writer.py --num-records 50000 --seq-length 2048
"""

import itertools
import logging
import shutil
import tempfile
import time
from collections.abc import Iterable, Iterator
from typing import Any

import click
import fsspec

from zephyr.writers import ThreadedBatchWriter, batchify

logger = logging.getLogger(__name__)


def generate_synthetic_records(num_records: int, seq_length: int) -> list[dict[str, list[int]]]:
    """Generate synthetic tokenized records mimicking a real tokenization pipeline.

    Each record contains input_ids and attention_mask of the given sequence length,
    similar to what a HuggingFace tokenizer produces.
    """
    records = []
    for i in range(num_records):
        records.append({
            "input_ids": list(range(i, i + seq_length)),
            "attention_mask": [1] * seq_length,
        })
    return records


def apply_map_pipeline(records: Iterable[dict]) -> Iterator[dict]:
    """Simulate a map-only pipeline that transforms each record.

    Adds a simple computed field, similar to what a real tokenization or
    post-processing map would do.
    """
    for record in records:
        # Simulate light compute work (e.g. adding special tokens, truncation)
        ids = record["input_ids"]
        yield {
            "input_ids": [0] + ids[:-1],  # prepend BOS, truncate to same length
            "attention_mask": record["attention_mask"],
        }


def _write_synchronous(
    records: Iterator[dict[str, Any]],
    output_path: str,
    metadata: dict[str, Any],
    batch_size: int,
) -> dict:
    """Write levanter cache synchronously (no background thread) for baseline comparison."""
    from levanter.store.cache import CacheMetadata, SerialCacheWriter

    from zephyr.writers import _promote_tmp_cache, ensure_parent_dir

    ensure_parent_dir(output_path)
    record_iter = iter(records)
    tmp_path = f"{output_path}.tmp"
    fs = fsspec.core.url_to_fs(output_path)[0]

    try:
        exemplar = next(record_iter)
    except StopIteration:
        return {"path": output_path, "count": 0}

    count = 1

    with SerialCacheWriter(
        tmp_path, exemplar, shard_name=output_path, metadata=CacheMetadata(metadata), mode="w"
    ) as writer:
        writer.write_batch([exemplar])
        for batch in batchify(record_iter, n=batch_size):
            writer.write_batch(batch)
            count += len(batch)

    _promote_tmp_cache(fs, tmp_path, output_path)

    with fsspec.open(f"{output_path}/.success", "w") as f:
        f.write("")

    return {"path": output_path, "count": count}


def _write_threaded(
    records: Iterator[dict[str, Any]],
    output_path: str,
    metadata: dict[str, Any],
    batch_size: int,
) -> dict:
    """Write levanter cache with ThreadedBatchWriter (the PR's approach)."""
    from levanter.store.cache import CacheMetadata, SerialCacheWriter

    from zephyr.writers import _promote_tmp_cache, ensure_parent_dir

    ensure_parent_dir(output_path)
    record_iter = iter(records)
    tmp_path = f"{output_path}.tmp"
    fs = fsspec.core.url_to_fs(output_path)[0]

    try:
        exemplar = next(record_iter)
    except StopIteration:
        return {"path": output_path, "count": 0}

    count = 1

    with SerialCacheWriter(
        tmp_path, exemplar, shard_name=output_path, metadata=CacheMetadata(metadata), mode="w"
    ) as writer:
        with ThreadedBatchWriter(writer.write_batch) as threaded:
            threaded.submit([exemplar])
            for batch in batchify(record_iter, n=batch_size):
                threaded.submit(batch)
                count += len(batch)

    _promote_tmp_cache(fs, tmp_path, output_path)

    with fsspec.open(f"{output_path}/.success", "w") as f:
        f.write("")

    return {"path": output_path, "count": count}


def run_single_benchmark(
    name: str,
    write_fn,
    records: list[dict],
    output_dir: str,
    batch_size: int,
) -> dict[str, Any]:
    """Run a single benchmark iteration and return timing metrics."""
    output_path = f"{output_dir}/{name}"

    # Apply the map pipeline to get a fresh iterator each time
    mapped = apply_map_pipeline(iter(records))

    start = time.perf_counter()
    result = write_fn(mapped, output_path, metadata={}, batch_size=batch_size)
    elapsed = time.perf_counter() - start

    return {
        "name": name,
        "count": result["count"],
        "elapsed_s": elapsed,
        "records_per_sec": result["count"] / elapsed if elapsed > 0 else float("inf"),
    }


@click.command()
@click.option("--num-records", type=int, default=20_000, help="Number of synthetic records to generate")
@click.option("--seq-length", type=int, default=2048, help="Sequence length per record (tokens)")
@click.option("--batch-size", type=int, default=1024, help="Batch size for batchify")
@click.option("--warmup-rounds", type=int, default=1, help="Warmup rounds (discarded)")
@click.option("--bench-rounds", type=int, default=3, help="Measured rounds")
def main(
    num_records: int,
    seq_length: int,
    batch_size: int,
    warmup_rounds: int,
    bench_rounds: int,
) -> None:
    """Benchmark synchronous vs threaded Levanter cache writes."""
    # Import levanter early to fail fast
    try:
        import levanter.store.cache  # noqa: F401
    except ImportError:
        print("ERROR: levanter is not installed. Install it to run this benchmark.")
        raise SystemExit(1)

    print("=" * 70)
    print("Threaded Levanter Writer Benchmark")
    print("=" * 70)
    print(f"  Records:      {num_records:,}")
    print(f"  Seq length:   {seq_length:,}")
    print(f"  Batch size:   {batch_size:,}")
    print(f"  Warmup:       {warmup_rounds} round(s)")
    print(f"  Bench:        {bench_rounds} round(s)")
    print()

    print("Generating synthetic records...")
    gen_start = time.perf_counter()
    records = generate_synthetic_records(num_records, seq_length)
    gen_elapsed = time.perf_counter() - gen_start
    total_tokens = num_records * seq_length
    print(f"  Generated {num_records:,} records ({total_tokens:,} tokens) in {gen_elapsed:.2f}s")
    print()

    approaches = [
        ("sync", _write_synchronous),
        ("threaded", _write_threaded),
    ]

    # Warmup
    if warmup_rounds > 0:
        print(f"Warmup ({warmup_rounds} round(s))...")
        for _round in range(warmup_rounds):
            for name, write_fn in approaches:
                tmpdir = tempfile.mkdtemp(prefix=f"zephyr_bench_warmup_{name}_")
                try:
                    run_single_benchmark(name, write_fn, records, tmpdir, batch_size)
                finally:
                    shutil.rmtree(tmpdir, ignore_errors=True)
        print()

    # Benchmark
    all_results: dict[str, list[dict]] = {name: [] for name, _ in approaches}

    for round_idx in range(bench_rounds):
        print(f"Round {round_idx + 1}/{bench_rounds}:")
        for name, write_fn in approaches:
            tmpdir = tempfile.mkdtemp(prefix=f"zephyr_bench_{name}_")
            try:
                result = run_single_benchmark(name, write_fn, records, tmpdir, batch_size)
                all_results[name].append(result)
                print(f"  {name:>10s}: {result['elapsed_s']:.3f}s  ({result['records_per_sec']:,.0f} records/s)")
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

    # Summary
    print()
    print("=" * 70)
    print(f"{'Approach':<12} {'Mean (s)':<12} {'Min (s)':<12} {'Max (s)':<12} {'Mean rec/s':<15}")
    print("-" * 70)

    summary = {}
    for name, results in all_results.items():
        times = [r["elapsed_s"] for r in results]
        rates = [r["records_per_sec"] for r in results]
        mean_time = sum(times) / len(times)
        mean_rate = sum(rates) / len(rates)
        summary[name] = {"mean_time": mean_time, "mean_rate": mean_rate}
        print(f"{name:<12} {mean_time:<12.3f} {min(times):<12.3f} {max(times):<12.3f} {mean_rate:<15,.0f}")

    if "sync" in summary and "threaded" in summary:
        speedup = summary["sync"]["mean_time"] / summary["threaded"]["mean_time"]
        print("-" * 70)
        print(f"Speedup (threaded vs sync): {speedup:.2f}x")

    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    main()
