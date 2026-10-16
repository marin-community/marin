# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure FineStore commit cost under concurrent writers on a simulated S3 bucket.

Each writer thread opens its own ``DataStore`` (distinct writer ID), appends rows, and
publishes ``--commits`` flushes while the store's maintenance thread runs with the
given flush interval. The report lists elapsed wall time, total simulated requests,
compare-and-swap conflicts, abandoned compactions, and the final shard count and
manifest size, so layouts and commit strategies can be compared. ``--time-scale``
compresses latency and retry sleeps but not CPU work, so elapsed time is comparable
between runs at one scale and is not a fixed multiple of full-latency time.

Example:

    uv run --package marin-finestore lib/finestore/scripts/benchmark_multi_writer.py \\
        --writers 1 4 16 --commits 8 --time-scale 0.1

Arrow's thread pools may print "terminate called without an active exception" while the
interpreter exits after the report; the exit status is still zero.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum

from finestore.layout import FineStoreLayout
from finestore.reader import ReadView
from finestore.store import DataStore
from finestore.testing.simulated_s3 import DEFAULT_LATENCY, Request, simulated_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

ROWS_PER_COMMIT = 64
IDLE_INTERVAL = 1e9
COMPACTION_LOGGER = "finestore.compaction"


class Layout(StrEnum):
    """How the benchmark's writers share archive roots."""

    SHARED_ROOT = "shared-root"
    PER_WRITER_ROOT = "per-writer-root"


class _CompactionLog(logging.Handler):
    """Count compactions that were started and the ones abandoned after a lost race."""

    def __init__(self) -> None:
        super().__init__()
        self.started = 0
        self.abandoned = 0
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        with self._lock:
            if message.startswith("FineStore compacting"):
                self.started += 1
            elif message.startswith("FineStore abandoned compaction"):
                self.abandoned += 1


@dataclass(frozen=True)
class RunResult:
    writers: int
    layout: str
    wall_seconds: float
    requests: int
    cas_puts: int
    cas_conflicts: int
    compactions_started: int
    compactions_abandoned: int
    shards: int
    manifest_bytes: int
    point_lookup_requests: int


def _writer(root: str, writer_id: str, commits: int, flush_interval: float) -> None:
    with DataStore.open(root, writer_id=writer_id, flush_interval=flush_interval) as store:
        table = store.table("samples", primary_key=("writer", "row"))
        for commit in range(commits):
            table.extend(
                {"writer": writer_id, "row": commit * ROWS_PER_COMMIT + index, "score": float(index)}
                for index in range(ROWS_PER_COMMIT)
            )
            store.flush()


def run(
    *,
    writers: int,
    commits: int,
    latency: float,
    time_scale: float,
    flush_interval: float,
    layout: Layout,
    maintainers: int | None,
) -> RunResult:
    compaction_log = _CompactionLog()
    logging.getLogger(COMPACTION_LOGGER).addHandler(compaction_log)
    try:
        with simulated_s3(latency=latency, time_scale=time_scale) as fs:
            shared_root = "s3://bench/archive"
            roots = [
                prefix_join(shared_root, f"w{index}") if layout is Layout.PER_WRITER_ROOT else shared_root
                for index in range(writers)
            ]
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=writers) as pool:
                futures = [
                    pool.submit(
                        _writer,
                        root,
                        f"writer-{index}",
                        commits,
                        flush_interval * time_scale if maintainers is None or index < maintainers else IDLE_INTERVAL,
                    )
                    for index, root in enumerate(roots)
                ]
                for future in futures:
                    future.result()
            wall = time.monotonic() - started
            summary = fs.log.snapshot()
            shards = 0
            manifest_bytes = 0
            for root in dict.fromkeys(roots):
                view = ReadView(root)
                shards += len(view.list_shards("samples"))
                assert view.token is not None
                manifest_bytes += StoragePath(FineStoreLayout(root).manifest_path(view.token.commit_id)).size()
            before = fs.log.snapshot()
            ReadView(roots[0]).point("samples", writer="writer-0", row=0)
            lookup = fs.log.delta(before)
            return RunResult(
                writers=writers,
                layout=str(layout) + ("" if maintainers is None else f"/{maintainers}m"),
                wall_seconds=wall,
                requests=summary.total(),
                cas_puts=summary.requests[Request.CAS_PUT],
                cas_conflicts=summary.cas_conflicts,
                compactions_started=compaction_log.started,
                compactions_abandoned=compaction_log.abandoned,
                shards=shards,
                manifest_bytes=manifest_bytes,
                point_lookup_requests=lookup.total(),
            )
    finally:
        logging.getLogger(COMPACTION_LOGGER).removeHandler(compaction_log)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--writers", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--commits", type=int, default=8, help="flushes per writer")
    parser.add_argument("--latency", type=float, default=DEFAULT_LATENCY, help="simulated round trip in seconds")
    parser.add_argument("--time-scale", type=float, default=0.1, help="compress latency and backoff by this factor")
    parser.add_argument("--flush-interval", type=float, default=5.0, help="maintenance interval in unscaled seconds")
    parser.add_argument(
        "--layout",
        type=Layout,
        choices=list(Layout),
        default=Layout.SHARED_ROOT,
        help="whether writers share one archive root or each get their own",
    )
    parser.add_argument(
        "--maintainers",
        type=int,
        default=None,
        help="run background maintenance on only this many writers, emulating an elected compactor",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger(COMPACTION_LOGGER).setLevel(logging.INFO)
    logging.getLogger(COMPACTION_LOGGER).propagate = False

    header = (
        f"{'writers':>7} {'layout':>16} {'elapsed_s':>9} {'requests':>9} {'cas':>5} {'conflicts':>9} "
        f"{'compact':>8} {'abandon':>8} {'shards':>7} {'manifest_B':>11} {'lookup_req':>10}"
    )
    print(header)
    for writers in args.writers:
        result = run(
            writers=writers,
            commits=args.commits,
            latency=args.latency,
            time_scale=args.time_scale,
            flush_interval=args.flush_interval,
            layout=args.layout,
            maintainers=args.maintainers,
        )
        print(
            f"{result.writers:>7} {result.layout:>16} {result.wall_seconds:>9.1f} {result.requests:>9} "
            f"{result.cas_puts:>5} {result.cas_conflicts:>9} {result.compactions_started:>8} "
            f"{result.compactions_abandoned:>8} {result.shards:>7} {result.manifest_bytes:>11} "
            f"{result.point_lookup_requests:>10}"
        )


if __name__ == "__main__":
    main()
