# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-source flag-rate aggregator over the all-sources decon output.

Walks every parquet shard under each ``datakit/decon/<source>/`` dir,
reads only the ``attributes`` struct column, and counts
``contaminated`` true vs total. Reports one row per (source, sub-source)
plus a grand total. Output TSV next to the script.

Each decon output is small (3-20 MiB per source) so the per-shard
parquet reads are cheap. Concurrency = ``SOURCE_CONCURRENCY`` * shards
within each source.

Run on iris for in-region speed (recommended), or locally:

    MARIN_PREFIX=gs://marin-eu-west4 uv run python \\
        experiments/datakit/decontam/flag_rates.py \\
        --decon-root gs://marin-eu-west4/tmp/ttl=7d/rav/decon-all-sources-v1/datakit/decon/
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from pyarrow import fs as pa_fs
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)
REPORT_PATH = Path(__file__).with_name("flag_rates.tsv")

SOURCE_CONCURRENCY = 16
SHARD_CONCURRENCY = 8


def _list_parquet_recursive(root: str) -> list[str]:
    gcs = pa_fs.GcsFileSystem()
    bare = root.removeprefix("gs://").rstrip("/")
    entries = gcs.get_file_info(pa_fs.FileSelector(bare, recursive=True))
    return sorted(f"gs://{e.path}" for e in entries if e.path.endswith(".parquet"))


def _shard_flag_count(path: str) -> tuple[int, int]:
    """Return (total, flagged) for one decon parquet shard."""
    gcs = pa_fs.GcsFileSystem()
    bare = path.removeprefix("gs://")
    pf = pq.ParquetFile(bare, filesystem=gcs)
    total = 0
    flagged = 0
    for batch in pf.iter_batches(columns=["attributes"], batch_size=4096):
        attrs = batch.column("attributes").to_pylist()
        total += len(attrs)
        for a in attrs:
            if a and a.get("contaminated"):
                flagged += 1
    return (total, flagged)


def _aggregate_source(decon_dir: str) -> tuple[str, int, int]:
    """Return (sub-source key, total, flagged) for one decon source dir."""
    shards = _list_parquet_recursive(decon_dir)
    key = decon_dir.rstrip("/").removeprefix("gs://").rsplit("/datakit/decon/", 1)[-1]
    if not shards:
        return (key, 0, 0)
    total = 0
    flagged = 0
    with ThreadPoolExecutor(max_workers=SHARD_CONCURRENCY) as pool:
        for t, f in pool.map(_shard_flag_count, shards):
            total += t
            flagged += f
    return (key, total, flagged)


def _list_source_dirs(decon_root: str) -> list[str]:
    """All ``datakit/decon/<source>/<sub>/`` leaf dirs that contain .executor_status."""
    gcs = pa_fs.GcsFileSystem()
    bare = decon_root.removeprefix("gs://").rstrip("/")
    entries = gcs.get_file_info(pa_fs.FileSelector(bare, recursive=True))
    dirs: set[str] = set()
    for e in entries:
        if e.path.endswith("/.executor_status"):
            d = "gs://" + e.path.rsplit("/", 1)[0] + "/"
            dirs.add(d)
    return sorted(dirs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decon-root", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    sources = _list_source_dirs(args.decon_root)
    if args.limit:
        sources = sources[: args.limit]
    logger.info("aggregating across %d decon source dirs", len(sources))

    results: list[tuple[str, int, int]] = []
    with ThreadPoolExecutor(max_workers=SOURCE_CONCURRENCY) as pool:
        for i, r in enumerate(pool.map(_aggregate_source, sources), 1):
            results.append(r)
            if i % 10 == 0 or i == len(sources):
                logger.info("processed %d/%d sources", i, len(sources))

    results.sort(key=lambda r: -r[2])  # most-flagged first
    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        fh.write("source\ttotal\tflagged\trate\n")
        grand_total = grand_flagged = 0
        for key, total, flagged in results:
            rate = flagged / total if total else 0.0
            fh.write(f"{key}\t{total}\t{flagged}\t{rate:.6f}\n")
            grand_total += total
            grand_flagged += flagged
        fh.write(
            f"GRAND TOTAL\t{grand_total}\t{grand_flagged}\t{grand_flagged / grand_total if grand_total else 0:.6f}\n"
        )

    print()
    print(
        f"grand: {grand_flagged:,} flagged / {grand_total:,} records "
        f"= {grand_flagged / grand_total * 100 if grand_total else 0:.4f}%"
    )
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
