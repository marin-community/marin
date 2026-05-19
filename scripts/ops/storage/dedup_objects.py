#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dedup raw scan parquets by (bucket, name) via a Zephyr group_by.

Belt-and-suspenders for the distributed scan: every parquet row is the
size+metadata for one GCS object, but the scan can in principle emit the
same object more than once (RPC retries, overlapping prefix scans, etc).
This pipeline collapses those into one row per (bucket, name).

Submitted as an Iris coordinator job; ZephyrContext internally spawns worker
actors against the same cluster.

Usage:
    uv run iris --cluster=marin job run --cpu 1 --memory 4GB -- \\
        uv run python scripts/ops/storage/dedup_objects.py \\
        --input-glob 'gs://marin-us-central2/tmp/storage-scan/objects_*.parquet' \\
        --output-dir gs://marin-us-central2/tmp/storage-scan/deduped \\
        --num-shards 64
"""

from __future__ import annotations

import click
from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext


@click.command()
@click.option("--input-glob", required=True, help="Glob pattern for raw scan parquets.")
@click.option("--output-dir", required=True, help="GCS dir for deduped parquets.")
@click.option("--num-shards", default=64, show_default=True, type=int)
@click.option("--worker-cpu", default=2, show_default=True, type=int)
@click.option("--worker-ram", default="8g", show_default=True)
def main(input_glob: str, output_dir: str, num_shards: int, worker_cpu: int, worker_ram: str) -> None:
    output_pattern = f"{output_dir.rstrip('/')}/objects-{{shard:05d}}.parquet"

    pipeline = (
        Dataset.from_files(input_glob)
        .load_parquet()
        .deduplicate(
            key=lambda r: (r["bucket"], r["name"]),
            num_output_shards=num_shards,
        )
        .write_parquet(output_pattern)
    )

    ctx = ZephyrContext(
        name="storage-dedup",
        resources=ResourceConfig(cpu=worker_cpu, ram=worker_ram),
    )
    ctx.execute(pipeline)


if __name__ == "__main__":
    main()
