#!/usr/bin/env python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end shuffle benchmark for ``zephyr.shuffle``.

Generates ~``--target-bytes`` of synthetic items (no I/O at the source) and
runs a ``group_by`` so the full scatter + reduce path executes. Measures
total walltime, per-stage walltime via Zephyr counters, and throughput.

Each input shard generates its items in-memory via ``map_shard``, so the
benchmark exercises the shuffle layer (scatter writes, reduce reads, k-way
merge) without spending walltime on input parsing.

Examples:
    # Local (small) — sanity check
    uv run python lib/zephyr/tests/benchmark_shuffle.py \\
        --num-input-shards 8 --items-per-shard 50000 --item-bytes 200

    # On marin-dev cluster (~10 GB)
    SMOKE_RUN_ID="shuffle-bench-$(date +%s)" \\
    uv run iris --cluster=marin-dev job run --no-wait \\
        --memory=2G --disk=8G --cpu=1 --extra=cpu \\
        -e SMOKE_RUN_ID "$SMOKE_RUN_ID" \\
        -- python lib/zephyr/tests/benchmark_shuffle.py \\
           --num-input-shards 64 --items-per-shard 600000 --item-bytes 250

    # Skewed: 90% of items routed to a single hot reducer (shard 0)
    ... --hot-shard-frac 0.9 --hot-key-pool 128 ...

Output: prints a single JSON line ``RESULT: {...}`` for easy log scraping.
"""

from __future__ import annotations

import json
import logging
import os
import random
import string
import time
from collections.abc import Iterator

import click
from rigging.log_setup import configure_logging

from fray import ResourceConfig
from zephyr import Dataset
from zephyr.dataset import ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.plan import deterministic_hash

logger = logging.getLogger(__name__)


def _make_payload(rnd: random.Random, n: int) -> str:
    """Random ASCII payload of length n. Uses random chars so zstd cannot trivialise."""
    return "".join(rnd.choices(string.ascii_letters + string.digits, k=n))


def _hot_keys_for_shard(target_shard: int, num_output_shards: int, count: int) -> list[int]:
    """Find the first ``count`` integer keys whose hash routes to ``target_shard``.

    Used by the skewed benchmark to bias most items toward one reducer.
    """
    keys: list[int] = []
    k = 0
    while len(keys) < count:
        if deterministic_hash(k) % num_output_shards == target_shard:
            keys.append(k)
        k += 1
    return keys


def _gen_shard(
    _items: Iterator,
    info: ShardInfo,
    items_per_shard: int,
    item_bytes: int,
    num_keys: int,
    hot_shard_frac: float,
    hot_keys: list[int],
):
    """Generate ``items_per_shard`` synthetic dicts for this input shard.

    Each dict has a routing ``key`` (drawn from ``num_keys`` distinct values)
    and a payload of approximately ``item_bytes`` random characters. With
    ``hot_shard_frac > 0``, that fraction of items is biased to keys routing
    to a single hot reducer, the rest are uniform.
    """
    rnd = random.Random(info.shard_idx)
    payload_size = max(0, item_bytes - 32)  # leave headroom for dict + key overhead
    n_hot = len(hot_keys)
    for i in range(items_per_shard):
        if hot_shard_frac > 0 and rnd.random() < hot_shard_frac:
            key = hot_keys[rnd.randrange(n_hot)]
        else:
            key = rnd.randrange(num_keys)
        yield {
            "key": key,
            "seq": i,
            "src": info.shard_idx,
            "payload": _make_payload(rnd, payload_size),
        }


def _count_local(items: Iterator) -> int:
    return sum(1 for _ in items)


def _build_pipeline(
    num_input_shards: int,
    items_per_shard: int,
    item_bytes: int,
    num_keys: int,
    num_output_shards: int,
    hot_shard_frac: float,
    hot_keys: list[int],
) -> Dataset:
    """Empty seed -> generate items -> group_by -> count.

    The terminal ``reduce`` returns a single scalar (total item count) so the
    coordinator does not need to ship a large result back, but every item
    still flows through scatter + reduce.
    """
    seeds = list(range(num_input_shards))
    return (
        Dataset.from_list(seeds)
        .map_shard(
            lambda items, info: _gen_shard(items, info, items_per_shard, item_bytes, num_keys, hot_shard_frac, hot_keys)
        )
        .group_by(
            key=lambda x: x["key"],
            reducer=lambda key, items: {"key": key, "n": sum(1 for _ in items)},
            num_output_shards=num_output_shards,
        )
        .reduce(local_reducer=_count_local, global_reducer=sum)
    )


@click.command()
@click.option("--num-input-shards", type=int, default=8)
@click.option("--items-per-shard", type=int, default=50_000)
@click.option("--item-bytes", type=int, default=200, help="Approx bytes per generated item")
@click.option("--num-keys", type=int, default=10_000, help="Distinct group_by keys")
@click.option(
    "--num-output-shards",
    type=int,
    default=None,
    help="Number of output shards (defaults to num_input_shards)",
)
@click.option(
    "--hot-shard-frac",
    type=float,
    default=0.0,
    help="Fraction of items biased to keys routing to a single hot reducer (0 = uniform)",
)
@click.option(
    "--hot-key-pool",
    type=int,
    default=128,
    help="Number of distinct keys all routing to the hot shard (only used when --hot-shard-frac > 0)",
)
@click.option("--worker-cpu", type=int, default=1)
@click.option("--worker-ram", type=str, default="4g")
@click.option("--max-workers", type=int, default=None)
@click.option("--label", type=str, default="shuffle-bench")
def main(
    num_input_shards: int,
    items_per_shard: int,
    item_bytes: int,
    num_keys: int,
    num_output_shards: int | None,
    hot_shard_frac: float,
    hot_key_pool: int,
    worker_cpu: int,
    worker_ram: str,
    max_workers: int | None,
    label: str,
) -> None:
    configure_logging()

    n_out = num_output_shards if num_output_shards is not None else num_input_shards
    total_items = num_input_shards * items_per_shard
    target_gb = total_items * item_bytes / (1024**3)
    hot_keys = _hot_keys_for_shard(0, n_out, hot_key_pool) if hot_shard_frac > 0 else []

    logger.info(
        "Shuffle benchmark: %d shards x %d items x ~%d bytes = %.2f GB synthetic data, "
        "num_output_shards=%d, hot_shard_frac=%.2f (hot_keys=%d routing to shard 0)",
        num_input_shards,
        items_per_shard,
        item_bytes,
        target_gb,
        n_out,
        hot_shard_frac,
        len(hot_keys),
    )

    pipeline = _build_pipeline(num_input_shards, items_per_shard, item_bytes, num_keys, n_out, hot_shard_frac, hot_keys)

    ctx_kwargs: dict = {
        "name": label,
        "resources": ResourceConfig(cpu=worker_cpu, ram=worker_ram),
    }
    if max_workers is not None:
        ctx_kwargs["max_workers"] = max_workers

    ctx = ZephyrContext(**ctx_kwargs)

    t0 = time.monotonic()
    result = ctx.execute(pipeline)
    elapsed = time.monotonic() - t0

    counted = sum(result.results) if result.results else 0
    # Throughput is computed against the *input* item count, not the
    # post-aggregation count, so the number reflects the bytes pushed
    # through scatter+reduce.
    throughput_items = total_items / elapsed if elapsed > 0 else 0.0
    throughput_mb = (total_items * item_bytes) / (1024**2) / elapsed if elapsed > 0 else 0.0

    summary = {
        "label": label,
        "num_input_shards": num_input_shards,
        "items_per_shard": items_per_shard,
        "item_bytes": item_bytes,
        "num_keys": num_keys,
        "num_output_shards": n_out,
        "hot_shard_frac": hot_shard_frac,
        "hot_key_pool": len(hot_keys),
        "expected_items": total_items,
        "counted_items": counted,
        "elapsed_s": round(elapsed, 2),
        "items_per_sec": round(throughput_items, 1),
        "mb_per_sec": round(throughput_mb, 1),
        "target_gb": round(target_gb, 2),
        "counters": result.counters,
    }
    print("RESULT:", json.dumps(summary))

    status_path = os.environ.get("BENCH_STATUS_PATH")
    if status_path:
        from rigging.filesystem import url_to_fs

        fs, _ = url_to_fs(status_path)
        with fs.open(status_path, "w") as f:
            f.write(json.dumps(summary))


if __name__ == "__main__":
    main()
