#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distributed object-storage scanning via Iris actors, over any Marin backend.

Backend-agnostic successor to the GCS-only scanner: object listing goes through
``rigging.filesystem`` (fsspec/s3fs for CoreWeave and R2, gcsfs for GCS) instead
of the ``google.cloud.storage`` client, so the same coordinator/worker machinery
walks ``gs://`` and ``s3://`` buckets alike and writes the same parquet segments
that ``render_report.py`` rolls up.

Architecture:
  - Coordinator actor: holds a task queue of (bucket_url, prefix_path) pairs.
    Workers pull items, scan them, stream objects back, and push any split-off
    sub-prefixes as new tasks. The coordinator accumulates objects in memory and
    writes consolidated ~100MB parquet segments to the staging dir.
  - Worker jobs: each runs WORKER_THREADS local threads. Each thread loops pulling
    prefixes from the coordinator, scanning them via a bucket-routed lister, and
    reporting results (object dicts + new prefixes) back.

Each task adaptively probes its prefix with a flat (no-delimiter) listing: a
subtree that exhausts within FLAT_PROBE_MAX_OBJECTS is consumed whole by that
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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

import click
import google.auth
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage
from iris.actor.client import ActorClient
from iris.actor.server import ActorServer
from iris.client.client import iris_ctx
from iris.cluster.client import get_job_info
from iris.cluster.types import Entrypoint, ResourceSpec
from rigging.filesystem.storage_path import StoragePath
from rigging.fsutil.listing import (
    entry_mtime,
    flat_listing_page,
    is_s3_filesystem,
    iter_object_pages,
    listing_filesystem,
)

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

# Streaming chunk size — worker sends this many objects per RPC during long scans.
# Keeps worker memory bounded regardless of prefix size.
WORKER_STREAM_CHUNK = 5_000

# Flat-probe threshold for the adaptive splitter. A task first lists its whole
# subtree flat (no delimiter); if the listing exhausts within this many objects,
# the task consumes the subtree itself and creates no subtasks. Only larger
# subtrees split, keeping the task count around objects / this value.
FLAT_PROBE_MAX_OBJECTS = 25_000

# Stop splitting at this depth below the top-level prefixes and stream the flat
# listing to completion instead: deeper fan-out multiplies listing RPCs faster
# than it adds useful parallelism.
MAX_SPLIT_DEPTH = 2

# google.cloud.storage paging parameters for the GCS lister.
GCS_PAGE_SIZE = 5_000
GCS_LIST_TIMEOUT = 120
BLOB_FIELDS = "items(name,size,storageClass,timeCreated,updated),prefixes,nextPageToken"

# Coordinator flushes when buffer reaches this many objects.
# ~2M objects x ~150 bytes/row ≈ 300MB uncompressed, ~50-80MB zstd parquet.
# Coordinator runs with 30GB so this leaves plenty of headroom.
COORDINATOR_FLUSH_THRESHOLD = 2_000_000

# Abandon stragglers when the queue has been empty AND no progress has been
# reported for this long. "Progress" means either a task completed *or* a
# worker streamed objects to the coordinator — workers scanning a huge flat
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
    files — without this, every re-run strictly appends and the consumer
    sees N-way duplicated (bucket, name) rows.
    """

    pattern = f"{staging_dir.rstrip('/')}/objects_*.parquet"
    existing = StoragePath(pattern).glob()
    if not existing:
        return
    print(f"Truncating {len(existing)} stale segments under {staging_dir}")
    for path in existing:
        path.rm()


# ---------------------------------------------------------------------------
# Bucket-routed listers
# ---------------------------------------------------------------------------


def _bucket_name(bucket_url: str) -> str:
    """The bucket name (no scheme) for a ``scheme://bucket`` URL."""
    return StoragePath(bucket_url).bucket


def _entry_to_object(entry: dict, bucket_name: str) -> dict:
    """Normalize one s3fs detail dict into an object row.

    S3 reports no distinct creation timestamp, so ``created`` is null and only
    ``updated`` (``LastModified``) carries a time. List pages carry no storage
    class either; CoreWeave and R2 have a single class, priced as STANDARD.
    """
    key = entry["name"].removeprefix(f"{bucket_name}/")
    return {
        "bucket": bucket_name,
        "name": key,
        "size_bytes": int(entry.get("size") or 0),
        "storage_class_id": STORAGE_CLASS_IDS["STANDARD"],
        "created": None,
        "updated": entry_mtime(entry),
    }


class _S3Lister:
    """Lists an S3-backed bucket through its routed s3fs filesystem."""

    def __init__(self, fs: Any, bucket_name: str) -> None:
        self._fs = fs
        self._bucket_name = bucket_name

    def flat_pages(self, path: str) -> Iterator[list[dict]]:
        """Pages of object rows from a flat (recursive) listing of *path*."""
        token: str | None = None
        while True:
            entries, token = flat_listing_page(self._fs, path, token)
            yield [_entry_to_object(entry, self._bucket_name) for entry in entries]
            if token is None:
                return

    def delimiter_level(self, path: str) -> tuple[list[dict], list[str]]:
        """Object rows directly under *path* plus its immediate sub-prefix paths."""
        files: list[dict] = []
        subdirs: list[str] = []
        for page_files, page_dirs in iter_object_pages(self._fs, path):
            files.extend(_entry_to_object(entry, self._bucket_name) for entry in page_files)
            subdirs.extend(page_dirs)
        return files, subdirs


class _GcsLister:
    """Lists a GCS bucket through google.cloud.storage.

    gcsfs exposes only whole-level ``ls`` and all-at-once ``find``; neither pages
    a flat listing incrementally, so the adaptive probe would materialize entire
    subtrees in worker memory. The native client's ``list_blobs`` iterator pages
    both flat and delimiter listings.
    """

    def __init__(self, bucket_name: str) -> None:
        credentials, project = google.auth.default()
        self._client = storage.Client(project=project, credentials=credentials)
        self._bucket_name = bucket_name

    def flat_pages(self, path: str) -> Iterator[list[dict]]:
        """Pages of object rows from a flat (recursive) listing of *path*."""
        for page in self._list_blobs(path, delimiter=None).pages:
            yield [self._blob_to_object(blob) for blob in page]

    def delimiter_level(self, path: str) -> tuple[list[dict], list[str]]:
        """Object rows directly under *path* plus its immediate sub-prefix paths."""
        files: list[dict] = []
        subdirs: list[str] = []
        for page in self._list_blobs(path, delimiter="/").pages:
            files.extend(self._blob_to_object(blob) for blob in page)
            subdirs.extend(f"{self._bucket_name}/{prefix}" for prefix in page.prefixes)
        return files, subdirs

    def _list_blobs(self, path: str, delimiter: str | None):
        key = path.removeprefix(self._bucket_name).lstrip("/")
        if key and not key.endswith("/"):
            key += "/"
        return self._client.list_blobs(
            self._bucket_name,
            prefix=key,
            delimiter=delimiter,
            page_size=GCS_PAGE_SIZE,
            fields=BLOB_FIELDS,
            timeout=GCS_LIST_TIMEOUT,
        )

    def _blob_to_object(self, blob: Any) -> dict:
        storage_class = blob.storage_class or "STANDARD"
        return {
            "bucket": self._bucket_name,
            "name": blob.name,
            "size_bytes": int(blob.size or 0),
            "storage_class_id": STORAGE_CLASS_IDS.get(storage_class, STORAGE_CLASS_IDS["STANDARD"]),
            "created": blob.time_created,
            "updated": blob.updated,
        }


def make_lister(bucket_url: str) -> tuple[_S3Lister | _GcsLister, str]:
    """Build the lister for *bucket_url*'s declared backend: ``(lister, root_path)``."""
    fs, root_path = listing_filesystem(bucket_url, WORKER_THREADS)
    if is_s3_filesystem(fs):
        return _S3Lister(fs, _bucket_name(bucket_url)), root_path
    protocol = getattr(fs, "protocol", ())
    protocols = (protocol,) if isinstance(protocol, str) else tuple(protocol)
    if "gs" in protocols or "gcs" in protocols:
        return _GcsLister(_bucket_name(bucket_url)), root_path
    raise ValueError(f"No adaptive lister for {bucket_url} (filesystem protocol {protocols})")


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
        # Wall-clock of the last forward-progress signal from any worker —
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


class _WorkerListers:
    """Per-worker-thread cache of bucket-routed listers."""

    def __init__(self) -> None:
        self._listers: dict[str, _S3Lister | _GcsLister] = {}

    def get(self, bucket_url: str) -> _S3Lister | _GcsLister:
        if bucket_url not in self._listers:
            self._listers[bucket_url], _ = make_lister(bucket_url)
        return self._listers[bucket_url]


def _report_chunks(coordinator: Any, objects: list[dict]) -> None:
    """Stream *objects* to the coordinator in RPCs of at most WORKER_STREAM_CHUNK."""
    for start in range(0, len(objects), WORKER_STREAM_CHUNK):
        coordinator.report_objects(objects[start : start + WORKER_STREAM_CHUNK])


def scan_one_prefix(
    listers: _WorkerListers,
    task: ScanTask,
    coordinator: Any,
) -> list[ScanTask]:
    """Adaptively scan one prefix, streaming objects to the coordinator.

    Probes the subtree with a flat listing first: one that exhausts within
    FLAT_PROBE_MAX_OBJECTS is consumed whole here, so small directories never
    become per-prefix tasks. A larger subtree splits one delimiter level into
    subtasks — its probe objects are discarded, since the delimiter listing
    re-yields this level's files and the subtasks cover the rest. At
    MAX_SPLIT_DEPTH (or when there is nothing to split into) the flat listing
    streams to completion instead, keeping worker memory bounded.

    Returns new sub-prefix tasks for re-queuing (empty at a leaf).
    """
    lister = listers.get(task.bucket_url)
    pages = lister.flat_pages(task.path)
    probe: list[dict] = []
    subtree_exhausted = True
    for page in pages:
        probe.extend(page)
        if len(probe) >= FLAT_PROBE_MAX_OBJECTS:
            subtree_exhausted = False
            break

    if subtree_exhausted:
        _report_chunks(coordinator, probe)
        return []

    if task.depth < MAX_SPLIT_DEPTH:
        files, subdirs = lister.delimiter_level(task.path)
        if subdirs:
            _report_chunks(coordinator, files)
            return [ScanTask(bucket_url=task.bucket_url, path=sub, depth=task.depth + 1) for sub in subdirs]

    _report_chunks(coordinator, probe)
    for page in pages:
        _report_chunks(coordinator, page)
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
    listers = _WorkerListers()
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
            new_prefixes = scan_one_prefix(listers, task, coordinator)
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
    """Normalize a bucket identifier to a URL. Bare names are assumed GCS."""
    return bucket if "://" in bucket else f"gs://{bucket}"


def discover_top_level_prefixes(
    bucket_urls: Sequence[str],
    coordinator: ScanCoordinatorActor,
) -> list[ScanTask]:
    """List each bucket's root level, streaming root objects and returning sub-prefix tasks.

    Doing the root listing on the coordinator fans the queue out to every top-level
    prefix immediately, so the first workers do not serialize behind one root task.
    """
    tasks: list[ScanTask] = []
    for bucket_url in bucket_urls:
        log.info("Discovering prefixes for %s...", bucket_url)
        lister, root_path = make_lister(bucket_url)
        root_objects, subdirs = lister.delimiter_level(root_path)
        if root_objects:
            _report_chunks(coordinator, root_objects)
        tasks.extend(ScanTask(bucket_url=bucket_url, path=sub, depth=0) for sub in subdirs)
        log.info("  %s: %d top-level prefixes, %d root objects", bucket_url, len(subdirs), len(root_objects))
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

    ``buckets`` are bucket names (assumed ``gs://``) or full ``scheme://bucket`` URLs.
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

    # Submit one worker job with N replicas
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
        while True:
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
