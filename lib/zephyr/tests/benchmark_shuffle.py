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

    # Tier2-shaped item-size skew (lognormal + Pareto tail + mega docs),
    # mirroring scripts/datakit/generate_tier2_skewed.py at reduced scale
    ... --size-dist tier2 --item-bytes 5000 --mega-per-shard 2 ...

Output: prints a single JSON line ``RESULT: {...}`` for easy log scraping.
"""

import base64
import json
import logging
import math
import os
import random
import string
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import click
from fray.types import ResourceConfig
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging
from zephyr.dataset import Dataset, ShardInfo
from zephyr.execution import ZephyrContext
from zephyr.shard_keys import deterministic_hash

logger = logging.getLogger(__name__)


def _make_payload(rnd: random.Random, n: int) -> str:
    """Random ASCII payload of length n. Uses random chars so zstd cannot trivialise."""
    return "".join(rnd.choices(string.ascii_letters + string.digits, k=n))


@dataclass(frozen=True)
class SizeSkew:
    """Item-size distribution for the generated payloads.

    ``dist="fixed"`` keeps the historical behavior (every payload is
    ``item_bytes`` long). ``dist="tier2"`` mirrors
    ``scripts/datakit/generate_tier2_skewed.py`` at reduced scale:
    ``1 - heavy_frac`` lognormal centered at ``item_bytes`` (sigma 0.5),
    ``heavy_frac`` truncated Pareto with scale ``0.4 * item_bytes``, plus
    ``mega_per_shard`` forced docs uniform in [mega_min_bytes, mega_max_bytes]
    at random positions in each input shard.
    """

    dist: str = "fixed"
    heavy_frac: float = 0.3
    pareto_alpha: float = 1.1
    max_item_bytes: int = 32 * 1024 * 1024
    mega_per_shard: int = 0
    mega_min_bytes: int = 8 * 1024 * 1024
    mega_max_bytes: int = 16 * 1024 * 1024


def _make_random_ascii(n: int) -> str:
    """Random ASCII payload of length n, fast enough for multi-MB docs.

    base64 over urandom runs at native speed where per-char ``random.choices``
    would take seconds per mega doc. The result is incompressible (the
    conservative case for encode/upload timing). Content randomness is not
    seeded — only sizes and keys (both seeded) affect routing and grouping, so
    A/B runs still shuffle identically shaped data.
    """
    return base64.b64encode(os.urandom((n * 3) // 4 + 3)).decode("ascii")[:n]


def _sample_item_bytes(rnd: random.Random, mean_bytes: int, skew: SizeSkew) -> int:
    """Sample one payload size from the tier2-skewed mixture."""
    if rnd.random() < skew.heavy_frac:
        # Pareto(alpha, scale): X = scale * U^(-1/alpha) for U ~ Uniform(0,1).
        size = 0.4 * mean_bytes * (rnd.random() ** (-1.0 / skew.pareto_alpha))
    else:
        size = rnd.lognormvariate(math.log(mean_bytes), 0.5)
    return max(64, min(int(size), skew.max_item_bytes))


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
    skew: SizeSkew,
):
    """Generate ``items_per_shard`` synthetic dicts for this input shard.

    Each dict has a routing ``key`` (drawn from ``num_keys`` distinct values)
    and a payload sized per ``skew`` (fixed ``item_bytes``, or the tier2
    mixture with ``item_bytes`` as the lognormal mean). With
    ``hot_shard_frac > 0``, that fraction of items is biased to keys routing
    to a single hot reducer, the rest are uniform.
    """
    rnd = random.Random(info.shard_idx)
    payload_size = max(0, item_bytes - 32)  # leave headroom for dict + key overhead
    n_hot = len(hot_keys)
    mega_indices: set[int] = set()
    if skew.dist == "tier2" and skew.mega_per_shard > 0:
        mega_indices = set(rnd.sample(range(items_per_shard), min(skew.mega_per_shard, items_per_shard)))
    for i in range(items_per_shard):
        if hot_shard_frac > 0 and rnd.random() < hot_shard_frac:
            key = hot_keys[rnd.randrange(n_hot)]
        else:
            key = rnd.randrange(num_keys)
        if skew.dist == "tier2":
            if i in mega_indices:
                n = rnd.randint(skew.mega_min_bytes, skew.mega_max_bytes)
            else:
                n = _sample_item_bytes(rnd, item_bytes, skew)
            payload = _make_random_ascii(n)
        else:
            payload = _make_payload(rnd, payload_size)
        yield {
            "key": key,
            "seq": i,
            "src": info.shard_idx,
            "payload": payload,
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
    skew: SizeSkew,
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
            lambda items, info: _gen_shard(
                items, info, items_per_shard, item_bytes, num_keys, hot_shard_frac, hot_keys, skew
            )
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
@click.option(
    "--size-dist",
    type=click.Choice(["fixed", "tier2"]),
    default="fixed",
    show_default=True,
    help="Payload size distribution: fixed --item-bytes, or the tier2 lognormal+Pareto mixture "
    "with --item-bytes as the lognormal mean.",
)
@click.option("--size-heavy-frac", type=float, default=0.3, help="tier2: fraction of Pareto-tail items")
@click.option("--size-pareto-alpha", type=float, default=1.1, help="tier2: Pareto tail exponent")
@click.option("--max-item-bytes", type=int, default=32 * 1024 * 1024, help="tier2: cap on sampled payload size")
@click.option("--mega-per-shard", type=int, default=0, help="tier2: forced mega docs per input shard")
@click.option("--mega-min-bytes", type=int, default=8 * 1024 * 1024, help="tier2: mega doc lower bound")
@click.option("--mega-max-bytes", type=int, default=16 * 1024 * 1024, help="tier2: mega doc upper bound")
@click.option("--worker-cpu", type=int, default=1)
@click.option("--worker-ram", type=str, default="4g")
@click.option("--max-workers", type=int, default=None)
@click.option("--label", type=str, default="shuffle-bench")
@click.option(
    "--repeat",
    type=int,
    default=1,
    help="Run the shuffle this many times sequentially in the same process. "
    "Each iteration emits its own RESULT line (tagged with 'repeat').",
)
def main(
    num_input_shards: int,
    items_per_shard: int,
    item_bytes: int,
    num_keys: int,
    num_output_shards: int | None,
    hot_shard_frac: float,
    hot_key_pool: int,
    size_dist: str,
    size_heavy_frac: float,
    size_pareto_alpha: float,
    max_item_bytes: int,
    mega_per_shard: int,
    mega_min_bytes: int,
    mega_max_bytes: int,
    worker_cpu: int,
    worker_ram: str,
    max_workers: int | None,
    label: str,
    repeat: int,
) -> None:
    configure_logging()

    skew = SizeSkew(
        dist=size_dist,
        heavy_frac=size_heavy_frac,
        pareto_alpha=size_pareto_alpha,
        max_item_bytes=max_item_bytes,
        mega_per_shard=mega_per_shard,
        mega_min_bytes=mega_min_bytes,
        mega_max_bytes=mega_max_bytes,
    )
    n_out = num_output_shards if num_output_shards is not None else num_input_shards
    total_items = num_input_shards * items_per_shard
    # For tier2 sizes this is the nominal (lognormal-mean) volume, not the
    # realized one; use the zephyr/bytes_processed counter for actual bytes.
    target_gb = total_items * item_bytes / (1024**3)
    hot_keys = _hot_keys_for_shard(0, n_out, hot_key_pool) if hot_shard_frac > 0 else []

    logger.info(
        "Shuffle benchmark: %d shards x %d items x ~%d bytes = %.2f GB synthetic data, "
        "num_output_shards=%d, hot_shard_frac=%.2f (hot_keys=%d routing to shard 0), repeat=%d",
        num_input_shards,
        items_per_shard,
        item_bytes,
        target_gb,
        n_out,
        hot_shard_frac,
        len(hot_keys),
        repeat,
    )

    pipeline = _build_pipeline(
        num_input_shards, items_per_shard, item_bytes, num_keys, n_out, hot_shard_frac, hot_keys, skew
    )

    ctx_kwargs: dict = {
        "name": label,
        "resources": ResourceConfig(cpu=worker_cpu, ram=worker_ram),
    }
    if max_workers is not None:
        ctx_kwargs["max_workers"] = max_workers

    # Reuse one ZephyrContext across repeats so worker actors stay warm and
    # variance from coordinator/worker startup is isolated from shuffle time.
    ctx = ZephyrContext(**ctx_kwargs)

    for i in range(repeat):
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
            "repeat": i,
            "repeats": repeat,
            "num_input_shards": num_input_shards,
            "items_per_shard": items_per_shard,
            "item_bytes": item_bytes,
            "num_keys": num_keys,
            "num_output_shards": n_out,
            "hot_shard_frac": hot_shard_frac,
            "hot_key_pool": len(hot_keys),
            "size_skew": asdict(skew),
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
        StoragePath(status_path).write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
