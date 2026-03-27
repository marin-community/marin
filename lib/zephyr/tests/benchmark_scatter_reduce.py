#!/usr/bin/env python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark scatter/reduce in isolation: serialization, sort, and Arrow merge.

Directly exercises _write_parquet_scatter and _arrow_reduce_gen with synthetic
data to measure the performance of the scatter/reduce code paths without the
overhead of file loading, mapping, or writing final output.

Usage:
    cd lib/zephyr
    uv run python tests/benchmark_scatter_reduce.py
    uv run python tests/benchmark_scatter_reduce.py --num-items 1000000 --num-shards 128
"""

import logging
import os
import random
import resource
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass

import click

from zephyr.plan import _arrow_reduce_gen
from zephyr.shuffle import (
    _build_scatter_shard_from_manifest,
    _write_parquet_scatter,
    _write_scatter_manifest,
)

WORDS = """
the be to of and a in that have I it for not on with he as you do at this but his by from
they we say her she or an will my one all would there their what so up out if about who get
which go me when make can like time no just him know take people into year your good some
could them see other than then now look only come its over think also back after use two how
our work first well way even new want because any these give day most us data system process
compute memory network storage algorithm function variable method class object interface
""".split()


def generate_items(n: int, num_keys: int = 1000) -> list[dict]:
    """Generate n items resembling real dedup records (~150 bytes each).

    Realistic shape: hash key, document ID, file index, and a short field.
    Matches the typical item size in exact/fuzzy dedup pipelines.
    """
    return [
        {
            "key": random.randint(0, num_keys - 1),
            "id": f"doc-{i:08d}",
            "file_idx": i % 100,
            "score": random.random(),
        }
        for i in range(n)
    ]


def peak_rss_mb() -> float:
    """Return peak RSS in MB (macOS returns bytes, Linux returns KB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    return usage / 1024


@dataclass
class BenchmarkResult:
    name: str
    scatter_time_s: float
    reduce_time_s: float
    peak_rss_mb: float
    total_items: int
    num_shards: int
    num_keys: int
    unique_keys_found: int
    scatter_file_bytes: int

    @property
    def scatter_items_per_sec(self) -> float:
        return self.total_items / self.scatter_time_s if self.scatter_time_s > 0 else 0

    @property
    def reduce_items_per_sec(self) -> float:
        return self.total_items / self.reduce_time_s if self.reduce_time_s > 0 else 0


def _key_fn(item: dict) -> int:
    return item["key"]


def run_scatter(items: list[dict], tmp_dir: str, num_shards: int) -> tuple[str, float, int]:
    """Scatter items, return (manifest_path, elapsed_s, file_bytes)."""
    t0 = time.monotonic()
    parquet_path = f"{tmp_dir}/shard-0000.parquet"
    list_shard = _write_parquet_scatter(
        iter(items),
        source_shard=0,
        parquet_path=parquet_path,
        key_fn=_key_fn,
        num_output_shards=num_shards,
    )
    seg_paths = list(list_shard)
    manifest_path = f"{tmp_dir}/scatter_metadata"
    _write_scatter_manifest(seg_paths, manifest_path)
    elapsed = time.monotonic() - t0

    file_bytes = sum(os.path.getsize(p) for p in seg_paths if os.path.exists(p))
    return manifest_path, elapsed, file_bytes


def _keep_first(_key: int, items: Iterator) -> dict:
    return next(items)


def run_reduce(manifest_path: str, num_shards: int) -> tuple[int, float]:
    """Reduce all shards (keep-first per key) using Arrow merge, return (unique_keys, elapsed_s)."""
    t0 = time.monotonic()
    count = 0
    for shard_idx in range(num_shards):
        shard = _build_scatter_shard_from_manifest(manifest_path, shard_idx)
        for _item in _arrow_reduce_gen(shard, _keep_first):
            count += 1
    elapsed = time.monotonic() - t0
    return count, elapsed


@click.command()
@click.option("--num-items", default=500_000, help="Total items to scatter")
@click.option("--num-shards", default=64, help="Number of output shards")
@click.option("--num-keys", default=1000, help="Number of unique keys")
@click.option("--seed", default=42, help="Random seed for reproducibility")
def benchmark(num_items: int, num_shards: int, num_keys: int, seed: int) -> None:
    """Benchmark scatter/reduce performance in isolation."""
    random.seed(seed)

    print(f"\nGenerating {num_items:,} items ({num_keys} unique keys)...")
    gen_start = time.monotonic()
    items = generate_items(num_items, num_keys)
    gen_time = time.monotonic() - gen_start
    print(f"Generated in {gen_time:.2f}s")

    tmp_dir = tempfile.mkdtemp(prefix="zephyr_bench_scatter_")
    try:
        print(f"\nScattering to {num_shards} shards...")
        manifest_path, scatter_time, file_bytes = run_scatter(items, tmp_dir, num_shards)

        print("Reducing with Arrow merge (keep-first per key)...")
        arrow_keys, arrow_reduce_time = run_reduce(manifest_path, num_shards)

        print(f"\n{'=' * 60}")
        print("Scatter/Reduce Benchmark Results")
        print(f"{'=' * 60}")
        print(f"  Items:              {num_items:>12,}")
        print(f"  Shards:             {num_shards:>12,}")
        print(f"  Unique keys:        {num_keys:>12,}")
        print(f"  Keys found:         {arrow_keys:>12,}")
        print(f"  Scatter file size:  {file_bytes / (1024*1024):>12.1f} MB")
        print(f"{'─' * 60}")
        print(f"  Scatter time:       {scatter_time:>12.2f} s")
        print(f"  Scatter throughput: {num_items / scatter_time:>12,.0f} items/s")
        print(f"{'─' * 60}")
        print(f"  Reduce (Arrow):     {arrow_reduce_time:>12.2f} s  ({num_items / arrow_reduce_time:>10,.0f} items/s)")
        print(f"{'=' * 60}\n")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    benchmark()
