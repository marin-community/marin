#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delete GCS objects matching delete_rules while respecting protect_rules.

This operates at the **object level** rather than the directory level. Each
individual object is judged on its own merits against the rule set — a tiny
STANDARD sentinel file in a directory does not veto deletion of its NEARLINE
neighbours.

Pipeline:
    parquet scan -> DuckDB streaming query (cursor.fetchmany)
                 -> feeder thread -> object_q
                 -> N delete workers (batched)

Dry-run mode skips the workers entirely and instead streams one TSV line per
object (bucket, name, size_bytes, storage_class) to stdout so callers can pipe
to head/wc/awk without buffering the full delete set.

Usage:
    uv run scripts/ops/storage/delete_files.py \\
        --scan-parquet gs://marin-us-central2/storage-scan/YYYY-MM-DD/ \\
        --protect-rules scripts/ops/storage/purge/protect_rules.json \\
        --delete-rules scripts/ops/storage/purge/delete_rules.json \\
        [--dry-run]
"""

from __future__ import annotations

import json
import logging
import queue
import random
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import click
import duckdb
from google.api_core.exceptions import DeadlineExceeded, NotFound, ServiceUnavailable, TooManyRequests
from google.cloud import storage
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from scripts.storage.constants import STORAGE_CLASS_PRICING, human_bytes
from scripts.storage.report import _download_gcs_parquet

log = logging.getLogger("delete_files")

# ---------------------------------------------------------------------------
# Tunables (mirrored from the old cleanup pipeline)
# ---------------------------------------------------------------------------

N_DELETE_WORKERS = 32
DELETE_BATCH_SIZE = 100
DELETE_BATCH_TIMEOUT = (10, 30)
DELETE_BATCH_MAX_ATTEMPTS = 9
DELETE_BATCH_INITIAL_BACKOFF = 2.0
DELETE_WORKER_QUEUE_TIMEOUT = 1.0

OBJECT_Q_MAXSIZE = 20_000
CURSOR_FETCH_SIZE = 10_000

STORAGE_CLASS_CACHE_ROOT = Path("/tmp/storage-scan-cache")

CONSOLE = Console(stderr=True)


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtectRule:
    bucket: str
    pattern: str


@dataclass(frozen=True)
class DeleteRule:
    pattern: str
    storage_class: str | None


def _load_protect_rules(path: Path) -> list[ProtectRule]:
    raw = json.loads(path.read_text())
    return [ProtectRule(bucket=r["bucket"], pattern=r["pattern"]) for r in raw]


def _load_delete_rules(path: Path) -> list[DeleteRule]:
    raw = json.loads(path.read_text())
    return [DeleteRule(pattern=r["pattern"], storage_class=r.get("storage_class")) for r in raw]


# ---------------------------------------------------------------------------
# DuckDB query construction
# ---------------------------------------------------------------------------


def _resolve_parquet_dir(scan_parquet: str) -> Path:
    """Return a local directory containing *.parquet for the scan input."""
    if scan_parquet.startswith("gs://"):
        subpath = scan_parquet.removeprefix("gs://").rstrip("/").replace("/", "_")
        local_cache = STORAGE_CLASS_CACHE_ROOT / subpath
        return _download_gcs_parquet(scan_parquet, local_cache)
    return Path(scan_parquet)


def _build_duckdb(
    protect_rules: list[ProtectRule],
    delete_rules: list[DeleteRule],
) -> duckdb.DuckDBPyConnection:
    """Build the DuckDB connection with rule tables but no `objects` view.

    The `objects` view is (re)bound per-parquet-file by the caller before each
    query, scoping the join/LIKE workload to one file at a time. This keeps
    peak memory bounded to a single file's working set (~50MB) instead of the
    full scan's 479M rows cross-joined against ~480 protect patterns.
    """
    conn = duckdb.connect(":memory:")
    conn.execute("SET threads=8")

    conn.execute(
        """
        CREATE TABLE storage_classes (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        )
        """
    )
    for sc_id, name, _, _ in STORAGE_CLASS_PRICING:
        conn.execute("INSERT INTO storage_classes VALUES (?, ?)", (sc_id, name))

    conn.execute("CREATE TABLE protect_rules (bucket TEXT NOT NULL, pattern TEXT NOT NULL)")
    conn.executemany(
        "INSERT INTO protect_rules VALUES (?, ?)",
        [(r.bucket, r.pattern) for r in protect_rules],
    )

    conn.execute("CREATE TABLE delete_rules (pattern TEXT NOT NULL, storage_class TEXT)")
    conn.executemany(
        "INSERT INTO delete_rules VALUES (?, ?)",
        [(r.pattern, r.storage_class) for r in delete_rules],
    )
    return conn


_DELETE_SET_SQL = """
SELECT o.bucket, o.name, o.size_bytes, sc.name AS storage_class
FROM objects o
JOIN storage_classes sc ON o.storage_class_id = sc.id
WHERE EXISTS (
    SELECT 1 FROM delete_rules dr
    WHERE o.name LIKE dr.pattern
      AND (dr.storage_class IS NULL OR sc.name = dr.storage_class)
)
AND NOT EXISTS (
    SELECT 1 FROM protect_rules p
    WHERE (p.bucket = '*' OR p.bucket = o.bucket)
      AND o.name LIKE p.pattern
)
"""


# ---------------------------------------------------------------------------
# Delete pipeline
# ---------------------------------------------------------------------------


class DeleteState:
    """Shared progress state: recent objects seen/deleted plus counters."""

    RECENT_MAXLEN = 100

    def __init__(self) -> None:
        self.start_time = time.time()
        self._lock = threading.Lock()
        self.recent_enqueued: deque[str] = deque(maxlen=self.RECENT_MAXLEN)
        self.recent_deleted: deque[str] = deque(maxlen=self.RECENT_MAXLEN)
        self.enqueued_count = 0
        self.enqueued_bytes = 0
        self.deleted_count = 0
        self.deleted_bytes = 0
        self.feeder_done = False
        self.files_done = 0
        self.files_total = 0

    def set_files_total(self, total: int) -> None:
        with self._lock:
            self.files_total = total

    def record_file_done(self) -> None:
        with self._lock:
            self.files_done += 1

    def record_enqueued(self, name: str, size: int) -> None:
        with self._lock:
            self.recent_enqueued.append(name)
            self.enqueued_count += 1
            self.enqueued_bytes += size

    def mark_feeder_done(self) -> None:
        with self._lock:
            self.feeder_done = True

    def record_deleted(self, names_and_sizes: list[tuple[str, int]]) -> None:
        with self._lock:
            for name, size in names_and_sizes:
                self.recent_deleted.append(name)
                self.deleted_count += 1
                self.deleted_bytes += size

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "recent_enqueued": list(self.recent_enqueued),
                "recent_deleted": list(self.recent_deleted),
                "enqueued_count": self.enqueued_count,
                "enqueued_bytes": self.enqueued_bytes,
                "deleted_count": self.deleted_count,
                "deleted_bytes": self.deleted_bytes,
                "feeder_done": self.feeder_done,
                "files_done": self.files_done,
                "files_total": self.files_total,
            }


def _raise_worker_failure(failures: queue.Queue[Exception]) -> None:
    try:
        exc = failures.get_nowait()
    except queue.Empty:
        return
    raise RuntimeError("Delete worker failed; aborting remaining work.") from exc


def _queue_put(
    q: queue.Queue,
    item,
    stop_event: threading.Event,
    failures: queue.Queue[Exception],
) -> None:
    while True:
        _raise_worker_failure(failures)
        if stop_event.is_set():
            raise RuntimeError("Delete aborted after a worker stopped.")
        try:
            q.put(item, timeout=DELETE_WORKER_QUEUE_TIMEOUT)
            return
        except queue.Full:
            continue


def _cursor_feeder(
    conn: duckdb.DuckDBPyConnection,
    parquet_files: list[Path],
    object_q: queue.Queue,
    n_workers: int,
    stop_event: threading.Event,
    failures: queue.Queue[Exception],
    state: DeleteState,
) -> None:
    """Stream rows from per-file DuckDB cursors onto ``object_q``.

    Each parquet segment is bound to the ``objects`` view in turn and queried
    in isolation. This bounds the NL-join working set to one file's rows while
    keeping the SQL (and bounded-queue backpressure) unchanged.
    """
    try:
        for segment in parquet_files:
            if stop_event.is_set():
                return
            path_lit = str(segment).replace("'", "''")
            conn.execute(
                f"CREATE OR REPLACE VIEW objects AS SELECT * FROM read_parquet('{path_lit}')",
            )
            cursor = conn.execute(_DELETE_SET_SQL)
            while True:
                rows = cursor.fetchmany(CURSOR_FETCH_SIZE)
                if not rows:
                    break
                for bucket, name, size_bytes, _storage_class in rows:
                    size = int(size_bytes or 0)
                    state.record_enqueued(name, size)
                    _queue_put(object_q, (bucket, name, size), stop_event, failures)
            state.record_file_done()
        for _ in range(n_workers):
            _queue_put(object_q, None, stop_event, failures)
    except Exception as exc:
        stop_event.set()
        failures.put(exc)
    finally:
        state.mark_feeder_done()


def _delete_batch(
    client: storage.Client,
    bucket_cache: dict[str, storage.Bucket],
    batch: list[tuple[str, str, int]],
    state: DeleteState,
) -> None:
    """Delete a heterogenous batch (potentially multi-bucket) via grouped batch calls."""
    # Group by bucket so each client.batch() hits a single bucket — simpler, and
    # avoids surprising per-object failure modes across buckets in one batch.
    by_bucket: dict[str, list[tuple[str, int]]] = {}
    for bucket, name, size in batch:
        by_bucket.setdefault(bucket, []).append((name, size))

    for bucket_name, entries in by_bucket.items():
        bucket_obj = bucket_cache.get(bucket_name)
        if bucket_obj is None:
            bucket_obj = client.bucket(bucket_name)
            bucket_cache[bucket_name] = bucket_obj

        for attempt in range(DELETE_BATCH_MAX_ATTEMPTS):
            try:
                with client.batch():
                    for name, _ in entries:
                        bucket_obj.blob(name).delete(timeout=DELETE_BATCH_TIMEOUT)
                state.record_deleted(entries)
                break
            except NotFound:
                # Idempotent: already-gone objects count as success.
                state.record_deleted(entries)
                break
            except (DeadlineExceeded, ServiceUnavailable, TooManyRequests) as exc:
                if attempt == DELETE_BATCH_MAX_ATTEMPTS - 1:
                    raise
                sleep = DELETE_BATCH_INITIAL_BACKOFF * (2**attempt) + random.uniform(0, 1)
                log.warning(
                    "%s from GCS batch delete (%s, %d objs); sleeping %.1fs (attempt %d/%d)",
                    exc.__class__.__name__,
                    bucket_name,
                    len(entries),
                    sleep,
                    attempt + 1,
                    DELETE_BATCH_MAX_ATTEMPTS,
                )
                time.sleep(sleep)


def _delete_worker(
    client: storage.Client,
    object_q: queue.Queue,
    state: DeleteState,
    stop_event: threading.Event,
    failures: queue.Queue[Exception],
) -> None:
    bucket_cache: dict[str, storage.Bucket] = {}
    batch: list[tuple[str, str, int]] = []

    try:
        while True:
            if stop_event.is_set():
                return
            try:
                item = object_q.get(timeout=DELETE_WORKER_QUEUE_TIMEOUT)
            except queue.Empty:
                continue
            if item is None:
                if batch:
                    _delete_batch(client, bucket_cache, batch, state)
                    batch.clear()
                return
            batch.append(item)
            if len(batch) >= DELETE_BATCH_SIZE:
                _delete_batch(client, bucket_cache, batch, state)
                batch.clear()
    except Exception as exc:
        stop_event.set()
        failures.put(exc)


# ---------------------------------------------------------------------------
# Live display
# ---------------------------------------------------------------------------


def _render_display(state: DeleteState, object_q_size: int) -> Group:
    snap = state.snapshot()
    deleted = snap["deleted_count"]
    deleted_bytes = snap["deleted_bytes"]
    enqueued = snap["enqueued_count"]
    enqueued_bytes = snap["enqueued_bytes"]

    elapsed = time.time() - state.start_time
    rate = deleted / elapsed if elapsed > 1 else 0.0

    # Once the feeder has finished, we know the true total and can give an ETA.
    if snap["feeder_done"] and rate > 0:
        remaining = max(0, enqueued - deleted)
        eta_s = int(remaining / rate)
        eta_str = f"{eta_s // 60}m {eta_s % 60}s"
    else:
        eta_str = "…"

    def item_table(items: list[str]) -> Table:
        t = Table(show_header=False, box=None, padding=(0, 0), expand=True)
        t.add_column("item", no_wrap=True, overflow="ellipsis")
        for item in items:
            t.add_row(item)
        return t

    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(
        Panel(item_table(snap["recent_enqueued"][-20:]), title="Recently Enqueued"),
        Panel(item_table(snap["recent_deleted"][-20:]), title="Recently Deleted"),
    )

    total_suffix = "" if snap["feeder_done"] else "+"
    files_done = snap["files_done"]
    files_total = snap["files_total"]
    status = Text(
        f" files {files_done:,}/{files_total:,}"
        f"  deleted {deleted:,}/{enqueued:,}{total_suffix}"
        f"  {human_bytes(deleted_bytes)}/{human_bytes(enqueued_bytes)}{total_suffix}"
        f"  object_q {object_q_size}"
        f"  {rate:,.0f} obj/s"
        f"  ETA {eta_str}",
        style="bold",
    )
    return Group(grid, status)


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


DRY_RUN_PROGRESS_INTERVAL = 50


def _run_dry_run(conn: duckdb.DuckDBPyConnection, parquet_files: list[Path]) -> None:
    """Stream the delete set to stdout as TSV, one line per object.

    Queries one parquet segment at a time to keep DuckDB memory bounded.
    Output is unbuffered-per-row so ``| head`` terminates promptly. A summary
    line is emitted to stderr when the stream completes.
    """
    total_objects = 0
    total_bytes = 0
    total_files = len(parquet_files)
    out = sys.stdout
    try:
        for file_idx, segment in enumerate(parquet_files, start=1):
            path_lit = str(segment).replace("'", "''")
            conn.execute(
                f"CREATE OR REPLACE VIEW objects AS SELECT * FROM read_parquet('{path_lit}')",
            )
            cursor = conn.execute(_DELETE_SET_SQL)
            while True:
                rows = cursor.fetchmany(CURSOR_FETCH_SIZE)
                if not rows:
                    break
                for bucket, name, size_bytes, storage_class in rows:
                    size = int(size_bytes or 0)
                    out.write(f"{bucket}\t{name}\t{size}\t{storage_class}\n")
                    total_objects += 1
                    total_bytes += size
                out.flush()
            if file_idx % DRY_RUN_PROGRESS_INTERVAL == 0 or file_idx == total_files:
                sys.stderr.write(f"# file {file_idx}/{total_files} processed\n")
                sys.stderr.flush()
    except BrokenPipeError:
        # Downstream (head, etc.) closed early. Exit cleanly without a traceback
        # or summary — the consumer intentionally stopped reading.
        try:
            sys.stdout.close()
        except Exception:
            pass
        return

    sys.stderr.write(f"# {total_objects:,} objects, {human_bytes(total_bytes)} would be deleted\n")
    sys.stderr.flush()


def _run_pipeline(
    conn: duckdb.DuckDBPyConnection,
    parquet_files: list[Path],
    project: str | None,
) -> None:
    client = storage.Client(project=project) if project else storage.Client()

    state = DeleteState()
    state.set_files_total(len(parquet_files))
    object_q: queue.Queue = queue.Queue(maxsize=OBJECT_Q_MAXSIZE)
    stop_event = threading.Event()
    failures: queue.Queue[Exception] = queue.Queue()

    feeder = threading.Thread(
        target=_cursor_feeder,
        args=(conn, parquet_files, object_q, N_DELETE_WORKERS, stop_event, failures, state),
        daemon=True,
    )
    workers = [
        threading.Thread(
            target=_delete_worker,
            args=(client, object_q, state, stop_event, failures),
            daemon=True,
        )
        for _ in range(N_DELETE_WORKERS)
    ]
    for t in [feeder, *workers]:
        t.start()

    try:
        with Live(console=CONSOLE, refresh_per_second=1) as live:
            while any(w.is_alive() for w in [feeder, *workers]):
                time.sleep(1.0)
                _raise_worker_failure(failures)
                live.update(_render_display(state, object_q.qsize()))
            live.update(_render_display(state, object_q.qsize()))
        for t in [feeder, *workers]:
            t.join()
        _raise_worker_failure(failures)
    except Exception:
        stop_event.set()
        raise


@click.command()
@click.option("--scan-parquet", required=True, help="GCS or local path to objects parquet directory.")
@click.option("--protect-rules", required=True, type=click.Path(exists=True), help="JSON protect rules.")
@click.option("--delete-rules", required=True, type=click.Path(exists=True), help="JSON delete rules.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Stream TSV (bucket, name, size, storage_class) of would-be deletions to stdout.",
)
@click.option("--delete-workers", default=N_DELETE_WORKERS, type=int, show_default=True)
@click.option("--project", default=None, help="GCP project override.")
def main(
    scan_parquet: str,
    protect_rules: str,
    delete_rules: str,
    dry_run: bool,
    delete_workers: int,
    project: str | None,
) -> None:
    """Delete GCS objects matching delete_rules but not protect_rules."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Override the module constant so the pipeline uses the CLI choice. A single
    # setting keeps feeder sentinels in sync with worker count.
    global N_DELETE_WORKERS
    N_DELETE_WORKERS = delete_workers

    protect = _load_protect_rules(Path(protect_rules))
    deletes = _load_delete_rules(Path(delete_rules))

    parquet_dir = _resolve_parquet_dir(scan_parquet)
    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        raise click.ClickException(f"No parquet files found under {parquet_dir}")
    conn = _build_duckdb(protect, deletes)

    if dry_run:
        # Dry run writes a clean TSV stream to stdout; keep Rich/log chatter
        # confined to stderr so `| head` and `| wc -l` stay usable.
        _run_dry_run(conn, parquet_files)
        return

    CONSOLE.print(f"loaded {len(protect)} protect rules, {len(deletes)} delete rules")
    CONSOLE.print(f"scanning {len(parquet_files)} parquet files one-at-a-time")
    CONSOLE.print(f"starting delete pipeline with {N_DELETE_WORKERS} workers ...")
    _run_pipeline(conn, parquet_files, project)
    CONSOLE.print("done.")


if __name__ == "__main__":
    main()
