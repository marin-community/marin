#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distributed object-storage scanning via Iris actors, over any Marin backend.

Backend-agnostic successor to the GCS-only scanner: ``filesystem_for`` attaches
backend-aware paged listing operations at ``fs.listing``, so the same
coordinator/worker machinery walks ``gs://`` and ``s3://`` buckets alike and
writes the parquet segments that ``render_report.py`` rolls up.

Architecture:
  - Coordinator actor: holds a task queue of (bucket_url, prefix_path) pairs.
    Workers pull items, scan them, stream objects back, and push any split-off
    sub-prefixes as new tasks. The coordinator accumulates objects in memory and
    writes consolidated ~100MB parquet segments to the staging dir.
  - Worker jobs: each runs WORKER_THREADS local threads. Each thread loops pulling
    prefixes from the coordinator, scanning them via a bucket-routed lister, and
    reporting results (object dicts + new prefixes) back.

Each task adaptively probes its prefix with a flat (no-delimiter) listing: a
subtree that exhausts below FLAT_PROBE_MAX_OBJECTS is consumed whole by that
one task, so small directories never fan out into per-prefix tasks and the task
count stays around objects/FLAT_PROBE_MAX_OBJECTS. Only larger subtrees split
one delimiter level into subtasks; past MAX_SPLIT_DEPTH the flat listing is
streamed to completion page by page, so worker memory stays bounded regardless
of prefix size.

Usage:
    uv run iris --cluster=marin job run \\
        --cpu 2 --memory 30GB --enable-extra-resources \\
        --target-cluster cw-rno2a -- \\
        uv run python scripts/ops/storage/scan_fs.py \\
        --staging-dir s3://marin-us-east-02a/tmp/storage-scan \\
        --buckets s3://marin-us-east-02a \\
        --workers 128
"""

import logging
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from typing import Any

import click
import pyarrow as pa
import pyarrow.parquet as pq
from iris.actor.client import ActorClient
from iris.actor.server import ActorServer
from iris.client.client import iris_ctx
from iris.cluster.client import get_job_info
from iris.cluster.types import Entrypoint, ResourceSpec
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from rigging.fsutil.listing import bucket_url, entry_mtime

from scripts.ops.storage.constants import (
    MARIN_BUCKETS,
    OBJECTS_ARROW_SCHEMA,
    STORAGE_CLASS_IDS,
    human_bytes,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKER_THREADS = 16

# Worker sends this many objects per RPC during long scans.
# Keeps worker memory bounded regardless of prefix size.
WORKER_STREAM_CHUNK = 5_000

# Flat-probe threshold for the adaptive splitter. A task first lists its whole
# subtree flat (no delimiter); if the listing exhausts before this many objects,
# the task consumes the subtree itself and creates no subtasks. Only larger
# subtrees split, keeping the task count around objects / this value.
FLAT_PROBE_MAX_OBJECTS = 25_000

# Stop splitting at this depth below the top-level prefixes and stream the flat
# listing to completion instead: deeper fan-out multiplies listing RPCs faster
# than it adds useful parallelism.
MAX_SPLIT_DEPTH = 2

# Coordinator flushes when buffer reaches this many objects.
# ~2M objects x ~150 bytes/row ≈ 300MB uncompressed, ~50-80MB zstd parquet.
# Coordinator runs with 30GB so this leaves plenty of headroom.
COORDINATOR_FLUSH_THRESHOLD = 2_000_000

# Abandon stragglers when the queue has been empty AND no progress has been
# reported for this long. "Progress" means either a task completed *or* a
# worker streamed objects to the coordinator. Workers scanning a huge flat
# prefix can take many minutes between task completions while still streaming
# steady chunks of objects, and we don't want to abandon those mid-flight.
# Truly hung workers (no RPCs at all) get timed out here; everything else is
# bounded by MAX_SCAN_SECONDS.
STRAGGLER_GRACE_SECONDS = 300

# Hard wall-clock cap on the whole scan. Beyond this we terminate workers
# and finalize whatever we have, even if some tasks are still in flight.
MAX_SCAN_SECONDS = 90 * 60

# Drain-based early finish. Once only a handful of tasks remain in flight
# (queue + active workers) and stay there for DRAIN_GRACE_SECONDS, finalize
# instead of waiting out MAX_SCAN_SECONDS. The straggler tail is a few huge
# flat prefixes that keep streaming objects (so the no-progress timeout never
# fires) but contribute marginally to a directory-level report.
DRAIN_TASK_THRESHOLD = 100
DRAIN_GRACE_SECONDS = 300

# Lower bound on elapsed wall-clock before the drain-based early finish is even
# considered. Early in a run the in-flight count can briefly dip to/below
# DRAIN_TASK_THRESHOLD before the queue fans back out, so without a floor we
# could finalize prematurely and miss large swaths of the namespace.
DRAIN_MIN_SCAN_SECONDS = 45 * 60


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanTask:
    """One prefix to scan, on the lister routed for its bucket URL.

    ``path`` is the protocol-stripped ``bucket/key`` form that
    :func:`rigging.filesystem.buckets.filesystem_for` returns. ``depth`` counts
    delimiter splits below the top-level prefixes; at MAX_SPLIT_DEPTH the task
    streams its subtree flat instead of splitting further.
    """

    bucket_url: str
    path: str
    depth: int = 0


@dataclass
class ColumnBuffer:
    """Column-oriented accumulator for scanned objects."""

    bucket: list[str] = dataclass_field(default_factory=list)
    name: list[str] = dataclass_field(default_factory=list)
    size_bytes: list[int] = dataclass_field(default_factory=list)
    storage_class_id: list[int] = dataclass_field(default_factory=list)
    created: list = dataclass_field(default_factory=list)
    updated: list = dataclass_field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.bucket)

    def extend(self, objects: list[dict]) -> int:
        """Append objects. Returns total bytes added."""
        total = 0
        for o in objects:
            self.bucket.append(o["bucket"])
            self.name.append(o["name"])
            self.size_bytes.append(o["size_bytes"])
            self.storage_class_id.append(o["storage_class_id"])
            self.created.append(o["created"])
            self.updated.append(o["updated"])
            total += o["size_bytes"]
        return total

    def to_arrow(self) -> pa.Table:
        return pa.table(
            {
                "bucket": pa.array(self.bucket, type=pa.string()),
                "name": pa.array(self.name, type=pa.string()),
                "size_bytes": pa.array(self.size_bytes, type=pa.int64()),
                "storage_class_id": pa.array(self.storage_class_id, type=pa.int32()),
                "created": pa.array(self.created, type=pa.timestamp("us", tz="UTC")),
                "updated": pa.array(self.updated, type=pa.timestamp("us", tz="UTC")),
            },
            schema=OBJECTS_ARROW_SCHEMA,
        )


def _write_parquet_segment(table: pa.Table, staging_dir: str) -> str:
    """Write an Arrow table as a zstd-compressed parquet segment to the staging dir."""

    segment_id = uuid.uuid4().hex[:12]
    path = f"{staging_dir}/objects_{segment_id}.parquet"
    with StoragePath(path).open("wb") as f:
        pq.write_table(table, f, compression="zstd")
    return path


def _truncate_staging_dir(staging_dir: str) -> None:
    """Delete all `objects_*.parquet` segments under staging_dir before a run.

    Segments are written under fresh UUIDs so reruns cannot overwrite prior
    files. Otherwise, every re-run appends and the consumer sees N-way
    duplicated (bucket, name) rows.
    """

    pattern = f"{staging_dir.rstrip('/')}/objects_*.parquet"
    existing = StoragePath(pattern).glob()
    if not existing:
        return
    print(f"Truncating {len(existing)} stale segments under {staging_dir}")
    for path in existing:
        path.rm()


# ---------------------------------------------------------------------------
# Entry normalization
# ---------------------------------------------------------------------------


def _bucket_name(url: str) -> str:
    return StoragePath(url).bucket


def _to_datetime(value: Any) -> datetime | None:
    """Coerce a backend timestamp (``datetime`` or ISO-8601 string) to UTC, else ``None``.

    gcsfs reports ``ctime``/``mtime`` as datetimes but ``timeCreated``/``updated`` as
    strings; s3fs reports ``LastModified`` as a datetime. This accepts either shape.
    """
    if isinstance(value, str) and value:
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return None


def _entry_to_object(entry: dict, bucket_name: str) -> dict:
    """Normalize one fsspec detail dict into an object row.

    ``created`` is the object's creation time where the backend reports one
    (gcsfs ``ctime``/``timeCreated``); S3 has no distinct creation timestamp, so it
    is left null and only ``updated`` (``LastModified``) carries a time.
    """
    key = entry["name"].removeprefix(f"{bucket_name}/")
    storage_class = entry.get("storageClass") or entry.get("StorageClass") or "STANDARD"
    return {
        "bucket": bucket_name,
        "name": key,
        "size_bytes": int(entry.get("size") or 0),
        "storage_class_id": STORAGE_CLASS_IDS.get(storage_class, STORAGE_CLASS_IDS["STANDARD"]),
        "created": _to_datetime(entry.get("ctime") or entry.get("timeCreated")),
        "updated": entry_mtime(entry),
    }


# ---------------------------------------------------------------------------
# Coordinator actor
# ---------------------------------------------------------------------------


class ScanCoordinatorActor:
    """Task queue + object accumulator. Workers pull tasks and push results.

    Incoming objects are buffered in a ColumnBuffer. When the buffer exceeds
    COORDINATOR_FLUSH_THRESHOLD, it is swapped out and written to the staging dir
    in a background thread so RPC handlers aren't blocked during the upload.
    """

    def __init__(self, staging_dir: str) -> None:
        self._staging_dir = staging_dir
        self._queue: deque[ScanTask] = deque()
        self._lock = threading.Lock()
        self._total_objects = 0
        self._total_bytes = 0
        self._tasks_completed = 0
        self._tasks_total = 0
        self._parquet_paths: list[str] = []
        self._errors: list[str] = []
        self._active_workers = 0
        self._buf = ColumnBuffer()
        self._flush_thread: threading.Thread | None = None
        # Wall-clock of the last forward-progress signal from any worker:
        # either a streamed batch of objects or a task completion. Used to
        # distinguish slow-but-progressing scans from genuinely hung workers
        # in the straggler-timeout check.
        self._last_progress_at: float | None = None

    def load_tasks(self, tasks: list[ScanTask]) -> None:
        with self._lock:
            self._queue.extend(tasks)
            self._tasks_total += len(tasks)

    def pull_task(self) -> ScanTask | None:
        with self._lock:
            if self._queue:
                self._active_workers += 1
                return self._queue.popleft()
            return None

    def report_objects(self, objects: list[dict]) -> None:
        """Worker streams scanned objects to the coordinator."""
        with self._lock:
            added_bytes = self._buf.extend(objects)
            self._total_objects += len(objects)
            self._total_bytes += added_bytes
            self._last_progress_at = time.monotonic()
            if self._buf.count >= COORDINATOR_FLUSH_THRESHOLD:
                self._swap_and_flush()

    def report_task_done(self, new_prefixes: list[ScanTask]) -> None:
        """Worker signals task complete and pushes any new sub-prefix tasks."""
        with self._lock:
            self._tasks_completed += 1
            self._active_workers -= 1
            self._last_progress_at = time.monotonic()
            if new_prefixes:
                self._queue.extend(new_prefixes)
                self._tasks_total += len(new_prefixes)

    def report_error(self, prefix: str, error: str) -> None:
        with self._lock:
            self._errors.append(f"{prefix}: {error}")
            self._tasks_completed += 1
            self._active_workers -= 1
            self._last_progress_at = time.monotonic()

    def flush(self) -> None:
        """Force-flush remaining buffered objects. Blocks until complete."""
        if self._flush_thread is not None:
            self._flush_thread.join()
        with self._lock:
            if self._buf.count > 0:
                self._swap_and_flush()
        if self._flush_thread is not None:
            self._flush_thread.join()

    def _swap_and_flush(self) -> None:
        """Swap buffer and write to the staging dir in background. Caller holds _lock."""
        snapshot = self._buf
        self._buf = ColumnBuffer()

        # If background thread is still writing, do a synchronous flush
        if self._flush_thread is not None and self._flush_thread.is_alive():
            table = snapshot.to_arrow()
            path = _write_parquet_segment(table, self._staging_dir)
            self._parquet_paths.append(path)
            return

        self._flush_thread = threading.Thread(
            target=self._bg_write,
            args=(snapshot,),
            daemon=True,
        )
        self._flush_thread.start()

    def _bg_write(self, buf: ColumnBuffer) -> None:
        table = buf.to_arrow()
        path = _write_parquet_segment(table, self._staging_dir)
        with self._lock:
            self._parquet_paths.append(path)

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            queue_empty = len(self._queue) == 0
            all_completed = queue_empty and self._active_workers == 0 and self._tasks_completed == self._tasks_total
            # Only declare stragglers timed out when the queue is empty AND
            # nothing has reported progress (objects or task completion) for
            # STRAGGLER_GRACE_SECONDS. Workers still streaming a huge flat prefix
            # keep _last_progress_at fresh via report_objects, so they won't be
            # killed mid-scan.
            stragglers_timed_out = (
                queue_empty
                and self._last_progress_at is not None
                and time.monotonic() - self._last_progress_at >= STRAGGLER_GRACE_SECONDS
            )
            return {
                "total_objects": self._total_objects,
                "total_bytes": self._total_bytes,
                "tasks_completed": self._tasks_completed,
                "tasks_total": self._tasks_total,
                "queue_size": len(self._queue),
                "active_workers": self._active_workers,
                "parquet_count": len(self._parquet_paths),
                "buffered": self._buf.count,
                "error_count": len(self._errors),
                "done": self._tasks_total > 0 and (all_completed or stragglers_timed_out),
            }

    def get_parquet_paths(self) -> list[str]:
        with self._lock:
            return list(self._parquet_paths)

    def get_errors(self) -> list[str]:
        with self._lock:
            return list(self._errors)


# ---------------------------------------------------------------------------
# Worker scanning logic
# ---------------------------------------------------------------------------


class _WorkerFilesystems:
    """Per-worker-thread cache of bucket-routed filesystems and bucket names."""

    def __init__(self) -> None:
        self._filesystems: dict[str, tuple[Any, str]] = {}

    def get(self, url: str) -> tuple[Any, str]:
        if url not in self._filesystems:
            fs, _ = filesystem_for(url)
            self._filesystems[url] = (fs, _bucket_name(url))
        return self._filesystems[url]


def _report_chunks(coordinator: Any, objects: list[dict]) -> None:
    for start in range(0, len(objects), WORKER_STREAM_CHUNK):
        coordinator.report_objects(objects[start : start + WORKER_STREAM_CHUNK])


def scan_one_prefix(
    filesystems: _WorkerFilesystems,
    task: ScanTask,
    coordinator: Any,
) -> list[ScanTask]:
    """Adaptively scan one prefix, streaming objects to the coordinator.

    Probes the subtree with a flat listing first: one that exhausts below
    FLAT_PROBE_MAX_OBJECTS is consumed whole here, so small directories never
    become per-prefix tasks. A larger subtree relists one delimiter level,
    discarding the probe objects to avoid double-counting, and returning the
    sub-prefixes as new tasks. When there are none, the level listing covered
    the whole subtree. At
    MAX_SPLIT_DEPTH the flat listing streams to completion instead. Every path
    reports page by page, keeping worker memory bounded.

    Returns new sub-prefix tasks for re-queuing (empty at a leaf).
    """
    fs, bucket_name = filesystems.get(task.bucket_url)
    pages = fs.listing.flat_pages(task.path)
    probe: list[dict] = []
    subtree_exhausted = True
    for page in pages:
        probe.extend(_entry_to_object(entry, bucket_name) for entry in page)
        if len(probe) >= FLAT_PROBE_MAX_OBJECTS:
            subtree_exhausted = False
            break

    if subtree_exhausted:
        _report_chunks(coordinator, probe)
        return []

    if task.depth < MAX_SPLIT_DEPTH:
        subdirs: list[str] = []
        for files, dirs in fs.listing.level_pages(task.path):
            _report_chunks(coordinator, [_entry_to_object(entry, bucket_name) for entry in files])
            subdirs.extend(dirs)
        return [ScanTask(bucket_url=task.bucket_url, path=sub, depth=task.depth + 1) for sub in subdirs]

    _report_chunks(coordinator, probe)
    for page in pages:
        _report_chunks(coordinator, [_entry_to_object(entry, bucket_name) for entry in page])
    return []


# ---------------------------------------------------------------------------
# Worker thread loop
# ---------------------------------------------------------------------------


def _worker_thread_loop(
    coordinator: Any,  # ActorClient or ScanCoordinatorActor
    stop_event: threading.Event,
    thread_id: str,
) -> None:
    """Single worker thread: pull tasks, scan, report results back."""
    filesystems = _WorkerFilesystems()
    idle_count = 0
    max_idle = 20

    while not stop_event.is_set():
        task = coordinator.pull_task()
        if task is None:
            idle_count += 1
            status = coordinator.get_status()
            if status["done"]:
                log.info("[%s] coordinator reports done, exiting", thread_id)
                return
            if idle_count > max_idle:
                log.info("[%s] idle too long, checking if done", thread_id)
                if status["queue_size"] == 0 and status["active_workers"] == 0:
                    return
                idle_count = 0
            time.sleep(0.5)
            continue

        idle_count = 0
        try:
            new_prefixes = scan_one_prefix(filesystems, task, coordinator)
            coordinator.report_task_done(new_prefixes)
        except Exception as e:
            log.exception("[%s] error scanning %s", thread_id, task.path)
            coordinator.report_error(task.path, str(e))


# ---------------------------------------------------------------------------
# Iris worker job entrypoint
# ---------------------------------------------------------------------------


def worker_job_entrypoint(coordinator_actor_name: str) -> None:
    """Iris job entrypoint for scan workers.

    Discovers the coordinator actor, then runs WORKER_THREADS threads
    that pull tasks and list prefixes.
    """

    ctx = iris_ctx()
    resolver = ctx.resolver
    coordinator = ActorClient(resolver, coordinator_actor_name, call_timeout=300.0)

    stop_event = threading.Event()
    threads = []
    for i in range(WORKER_THREADS):
        t = threading.Thread(
            target=_worker_thread_loop,
            args=(coordinator, stop_event, f"w{i}"),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    log.info("Worker job complete, all threads finished")


# ---------------------------------------------------------------------------
# Prefix discovery (runs on coordinator)
# ---------------------------------------------------------------------------


def _bucket_url(bucket: str) -> str:
    """Normalize a bucket identifier to a URL, routing bare names by declared backend."""
    return bucket if "://" in bucket else bucket_url(bucket)


def discover_top_level_prefixes(
    bucket_urls: Sequence[str],
    coordinator: ScanCoordinatorActor,
) -> list[ScanTask]:
    """List each bucket's root level, streaming root objects and returning sub-prefix tasks."""
    tasks: list[ScanTask] = []
    for url in bucket_urls:
        log.info("Discovering prefixes for %s...", url)
        fs, root_path = filesystem_for(url)
        bucket_name = _bucket_name(url)
        root_objects = 0
        subdirs: list[str] = []
        for files, dirs in fs.listing.level_pages(root_path):
            _report_chunks(coordinator, [_entry_to_object(entry, bucket_name) for entry in files])
            root_objects += len(files)
            subdirs.extend(dirs)
        tasks.extend(ScanTask(bucket_url=url, path=sub, depth=0) for sub in subdirs)
        log.info("  %s: %d top-level prefixes, %d root objects", url, len(subdirs), root_objects)
    return tasks


# ---------------------------------------------------------------------------
# Iris distributed execution
# ---------------------------------------------------------------------------


def run_distributed(
    buckets: Sequence[str],
    num_workers: int,
    staging_dir: str,
) -> None:
    """Run the scan as an Iris coordinator job.

    ``buckets`` are configured bucket names or full ``scheme://bucket`` URLs.
    The coordinator accumulates objects in memory and writes consolidated parquet
    segments (~100MB each) to ``staging_dir``.
    """

    # Iris captures the coordinator's stdout as a pipe, which block-buffers by
    # default and hides the periodic progress lines until the process exits.
    sys.stdout.reconfigure(line_buffering=True)

    ctx = iris_ctx()
    client = ctx.client

    bucket_urls = [_bucket_url(b) for b in buckets]

    _truncate_staging_dir(staging_dir)

    # Start coordinator actor
    coordinator = ScanCoordinatorActor(staging_dir)
    actor_name = "scan-coordinator"
    server = ActorServer(host="0.0.0.0")
    server.register(actor_name, coordinator)
    actual_port = server.serve_background()

    job_info = get_job_info()
    address = f"http://{job_info.advertise_host}:{actual_port}"
    ctx.registry.register(actor_name, address, {"role": "coordinator"})
    print(f"Coordinator actor registered at {address}")

    # Discover prefixes and load queue
    print(f"Discovering top-level prefixes for {len(bucket_urls)} buckets...")
    tasks = discover_top_level_prefixes(bucket_urls, coordinator)
    coordinator.load_tasks(tasks)
    print(f"Loaded {len(tasks)} initial tasks into queue")

    # Submit one worker job with N replicas. A bucket whose objects all live at
    # the root yields no tasks because discovery already streamed those objects.
    # In that case there is no worker job or queue to drain.
    worker_job = None
    if tasks:
        worker_job = client.submit(
            entrypoint=Entrypoint.from_callable(worker_job_entrypoint, actor_name),
            name="scan-workers",
            resources=ResourceSpec(cpu=2, memory="4GB"),
            replicas=num_workers,
        )
        print(f"Submitted worker job with {num_workers} replicas")

    # Monitor progress
    start_time = time.monotonic()
    # Monotonic time when the in-flight task count first dropped to the drain
    # threshold; reset whenever it climbs back above (new sub-prefixes queued).
    drained_since: float | None = None
    try:
        while tasks:
            status = coordinator.get_status()
            elapsed = time.monotonic() - start_time
            remaining = status["queue_size"] + status["active_workers"]

            print(
                f"[{elapsed:6.0f}s] "
                f"{status['tasks_completed']}/{status['tasks_total']} tasks | "
                f"{status['total_objects']:,} objects | "
                f"{human_bytes(status['total_bytes'])} | "
                f"queue={status['queue_size']} active={status['active_workers']} "
                f"buf={status['buffered']:,} parquets={status['parquet_count']} "
                f"errors={status['error_count']}"
            )

            if status["done"]:
                break

            # Only consider the drain-based early finish after a minimum
            # elapsed time, so a transient early dip in the in-flight count
            # can't finalize the scan prematurely.
            if elapsed >= DRAIN_MIN_SCAN_SECONDS and remaining <= DRAIN_TASK_THRESHOLD:
                if drained_since is None:
                    drained_since = time.monotonic()
                elif time.monotonic() - drained_since >= DRAIN_GRACE_SECONDS:
                    print(f"Only {remaining} tasks left for {DRAIN_GRACE_SECONDS}s; abandoning stragglers, finalizing")
                    break
            else:
                drained_since = None

            if elapsed >= MAX_SCAN_SECONDS:
                print(f"Wall-clock cap of {MAX_SCAN_SECONDS}s hit; abandoning stragglers and finalizing")
                break

            time.sleep(30)
    finally:
        if worker_job is not None:
            try:
                worker_job.terminate()
            except Exception:
                pass
        server.stop()

    # Flush remaining buffered objects
    coordinator.flush()

    elapsed = time.monotonic() - start_time
    final_status = coordinator.get_status()
    print(f"\nScan complete in {elapsed:.0f}s")
    print(f"  Objects: {final_status['total_objects']:,}")
    print(f"  Size: {human_bytes(final_status['total_bytes'])}")
    print(f"  Parquet segments: {final_status['parquet_count']}")

    errors = coordinator.get_errors()
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    {e}")

    print(f"\nParquet output: {staging_dir}")
    print(f"  Run report with: uv run scripts/ops/storage/render_report.py {staging_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--workers", default=4, type=int, show_default=True, help="Number of Iris worker replicas.")
@click.option("--staging-dir", required=True, help="Object-storage path (gs:// or s3://) for parquet output.")
@click.option("--buckets", help="Comma-separated bucket names or scheme://bucket URLs. Default: all MARIN_BUCKETS.")
def main(
    workers: int,
    staging_dir: str,
    buckets: str | None,
) -> None:
    """Run a distributed object-storage scan as an Iris coordinator job.

    Submit via iris job run (federate to a CoreWeave peer with --target-cluster):

        uv run iris --cluster=marin job run \\
            --cpu 2 --memory 30GB --enable-extra-resources \\
            --target-cluster cw-rno2a -- \\
            uv run python scripts/ops/storage/scan_fs.py \\
            --staging-dir s3://marin-us-east-02a/tmp/storage-scan \\
            --buckets s3://marin-us-east-02a \\
            --workers 128
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    bucket_list = buckets.split(",") if buckets else list(MARIN_BUCKETS)

    run_distributed(
        buckets=bucket_list,
        num_workers=workers,
        staging_dir=staging_dir,
    )


if __name__ == "__main__":
    main()
