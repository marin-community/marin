#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distributed GCS object scanning via Iris actors.

Architecture:
  - Coordinator actor: holds a task queue of (bucket, prefix) pairs.
    Workers pull individual items, scan objects, and return the results.
    The coordinator accumulates objects in memory and writes consolidated
    ~100MB parquet segments to GCS.
  - Worker jobs: each runs WORKER_THREADS local threads. Each thread
    loops pulling prefixes from the coordinator, scanning objects via GCS
    API, and reporting results (object dicts + new prefixes) back.

Usage:
    uv run iris --cluster=marin job run \\
        --cpu 2 --memory 10GB --enable-extra-resources -- \\
        uv run python scripts/storage/distributed_scan.py \\
        --staging-dir gs://marin-us-central2/tmp/storage-scan \\
        --workers 128
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field as dataclass_field
from typing import Any

import click
import google.auth
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage

from scripts.storage.constants import (
    ADAPTIVE_MAX_DEPTH,
    ADAPTIVE_SPLIT_THRESHOLD,
    BLOB_FIELDS,
    GCS_LIST_TIMEOUT,
    GCS_MAX_PAGE_SIZE,
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

# Coordinator flushes when buffer reaches this many objects.
# ~2M objects x ~150 bytes/row ≈ 300MB uncompressed, ~50-80MB zstd parquet.
# Coordinator runs with 30GB so this leaves plenty of headroom.
COORDINATOR_FLUSH_THRESHOLD = 2_000_000


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScanTask:
    bucket: str
    prefix: str
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


def _write_parquet_to_gcs(table: pa.Table, staging_dir: str) -> str:
    """Write an Arrow table as a zstd-compressed parquet segment to GCS."""
    import fsspec

    segment_id = uuid.uuid4().hex[:12]
    path = f"{staging_dir}/objects_{segment_id}.parquet"
    with fsspec.open(path, "wb") as f:
        pq.write_table(table, f, compression="zstd")
    return path


def _truncate_staging_dir(staging_dir: str) -> None:
    """Delete all `objects_*.parquet` segments under staging_dir before a run.

    Segments are written under fresh UUIDs so reruns cannot overwrite prior
    files — without this, every re-run strictly appends and the consumer
    sees N-way duplicated (bucket, name) rows.
    """
    import fsspec

    fs, _ = fsspec.core.url_to_fs(staging_dir)
    pattern = f"{staging_dir.rstrip('/')}/objects_*.parquet"
    existing = fs.glob(pattern)
    if not existing:
        return
    print(f"Truncating {len(existing)} stale segments under {staging_dir}")
    for path in existing:
        fs.rm(path)


# ---------------------------------------------------------------------------
# Coordinator actor
# ---------------------------------------------------------------------------


class ScanCoordinatorActor:
    """Task queue + object accumulator. Workers pull tasks and push results.

    Incoming objects are buffered in a ColumnBuffer. When the buffer exceeds
    COORDINATOR_FLUSH_THRESHOLD, it is swapped out and written to GCS in a
    background thread so RPC handlers aren't blocked during the upload.
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
            if self._buf.count >= COORDINATOR_FLUSH_THRESHOLD:
                self._swap_and_flush()

    def report_task_done(self, new_prefixes: list[ScanTask]) -> None:
        """Worker signals task complete and pushes any new sub-prefix tasks."""
        with self._lock:
            self._tasks_completed += 1
            self._active_workers -= 1
            if new_prefixes:
                self._queue.extend(new_prefixes)
                self._tasks_total += len(new_prefixes)

    def report_error(self, prefix: str, error: str) -> None:
        with self._lock:
            self._errors.append(f"{prefix}: {error}")
            self._tasks_completed += 1
            self._active_workers -= 1

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
        """Swap buffer and write to GCS in background. Caller holds _lock."""
        snapshot = self._buf
        self._buf = ColumnBuffer()

        # If background thread is still writing, do a synchronous flush
        if self._flush_thread is not None and self._flush_thread.is_alive():
            table = snapshot.to_arrow()
            path = _write_parquet_to_gcs(table, self._staging_dir)
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
        path = _write_parquet_to_gcs(table, self._staging_dir)
        with self._lock:
            self._parquet_paths.append(path)

    def get_status(self) -> dict[str, Any]:
        with self._lock:
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
                "done": (
                    len(self._queue) == 0
                    and self._active_workers == 0
                    and self._tasks_completed == self._tasks_total
                    and self._tasks_total > 0
                ),
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


def _make_storage_client(project: str | None) -> storage.Client:
    credentials, default_project = google.auth.default()
    return storage.Client(project=project or default_project, credentials=credentials)


def _blob_to_dict(blob: Any, bucket_name: str) -> dict:
    sc = blob.storage_class or "STANDARD"
    return {
        "bucket": bucket_name,
        "name": blob.name,
        "size_bytes": int(blob.size or 0),
        "storage_class_id": STORAGE_CLASS_IDS.get(sc, STORAGE_CLASS_IDS["STANDARD"]),
        "created": blob.time_created,
        "updated": blob.updated,
    }


def _scan_with_delimiter(
    client: storage.Client,
    bucket_name: str,
    prefix: str,
) -> tuple[list[dict], list[str]]:
    """List objects at this level + discover sub-prefixes."""
    root_objects: list[dict] = []
    sub_prefixes: list[str] = []
    for page in client.list_blobs(
        bucket_name,
        prefix=prefix,
        delimiter="/",
        page_size=GCS_MAX_PAGE_SIZE,
        fields=BLOB_FIELDS,
        timeout=GCS_LIST_TIMEOUT,
    ).pages:
        for blob in page:
            root_objects.append(_blob_to_dict(blob, bucket_name))
        sub_prefixes.extend(page.prefixes)
    return root_objects, sub_prefixes


# Streaming chunk size — worker sends this many objects per RPC during long scans.
# Keeps worker memory bounded regardless of prefix size.
WORKER_STREAM_CHUNK = 5_000


def _stream_pages(
    client: storage.Client,
    bucket_name: str,
    prefix: str,
    coordinator: Any,
    initial: list[dict],
) -> int:
    """Stream pages of a flat prefix to the coordinator in chunks.

    Returns total objects sent. Worker memory stays bounded at ~WORKER_STREAM_CHUNK
    objects regardless of prefix size.
    """
    chunk = list(initial)
    total = 0
    for page in client.list_blobs(
        bucket_name,
        prefix=prefix,
        page_size=GCS_MAX_PAGE_SIZE,
        fields=BLOB_FIELDS,
        timeout=GCS_LIST_TIMEOUT,
    ).pages:
        for blob in page:
            chunk.append(_blob_to_dict(blob, bucket_name))
            if len(chunk) >= WORKER_STREAM_CHUNK:
                coordinator.report_objects(chunk)
                total += len(chunk)
                chunk = []
    if chunk:
        coordinator.report_objects(chunk)
        total += len(chunk)
    return total


def scan_one_prefix(
    client: storage.Client,
    task: ScanTask,
    coordinator: Any,
) -> list[ScanTask]:
    """Scan a single prefix, streaming objects to the coordinator.

    Returns a list of new sub-prefix tasks for re-queuing (empty if leaf).
    Workers stream object data directly to the coordinator in chunks of
    WORKER_STREAM_CHUNK to keep memory bounded.
    """
    bucket_name = task.bucket
    prefix = task.prefix

    # Root-level: always use delimiter
    if prefix == "":
        root_objects, sub_prefixes = _scan_with_delimiter(client, bucket_name, "")
        if root_objects:
            coordinator.report_objects(root_objects)
        return [ScanTask(bucket=bucket_name, prefix=sp, depth=1) for sp in sub_prefixes]

    # Probe: flat scan up to threshold
    objects: list[dict] = []
    iterator = client.list_blobs(
        bucket_name,
        prefix=prefix,
        page_size=GCS_MAX_PAGE_SIZE,
        fields=BLOB_FIELDS,
        timeout=GCS_LIST_TIMEOUT,
    )
    pages_iter = iterator.pages
    is_small = False

    for page in pages_iter:
        page_items = [_blob_to_dict(blob, bucket_name) for blob in page]
        objects.extend(page_items)
        if len(page_items) < GCS_MAX_PAGE_SIZE:
            is_small = True
            break
        if len(objects) >= ADAPTIVE_SPLIT_THRESHOLD:
            break

    # Small prefix: send and done
    if is_small:
        if objects:
            coordinator.report_objects(objects)
        return []

    # Max depth: stream remaining pages directly (no buffering of full list)
    if task.depth >= ADAPTIVE_MAX_DEPTH:
        # Flush probe objects, then stream the rest page-by-page
        if objects:
            coordinator.report_objects(objects)
        chunk: list[dict] = []
        for page in pages_iter:
            for blob in page:
                chunk.append(_blob_to_dict(blob, bucket_name))
                if len(chunk) >= WORKER_STREAM_CHUNK:
                    coordinator.report_objects(chunk)
                    chunk = []
        if chunk:
            coordinator.report_objects(chunk)
        return []

    # Below max depth: try delimiter split
    del objects
    root_objects, sub_prefixes = _scan_with_delimiter(client, bucket_name, prefix)

    if sub_prefixes:
        if root_objects:
            coordinator.report_objects(root_objects)
        return [ScanTask(bucket=bucket_name, prefix=sp, depth=task.depth + 1) for sp in sub_prefixes]

    # No sub-prefixes (rare): stream the flat directory
    _stream_pages(client, bucket_name, prefix, coordinator, root_objects)
    return []


# ---------------------------------------------------------------------------
# Worker thread loop
# ---------------------------------------------------------------------------


def _worker_thread_loop(
    coordinator: Any,  # ActorClient or ScanCoordinatorActor
    project: str | None,
    stop_event: threading.Event,
    thread_id: str,
) -> None:
    """Single worker thread: pull tasks, scan, report results back."""
    client = _make_storage_client(project)
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
            new_prefixes = scan_one_prefix(client, task, coordinator)
            coordinator.report_task_done(new_prefixes)
        except Exception as e:
            log.exception("[%s] error scanning %s/%s", thread_id, task.bucket, task.prefix)
            coordinator.report_error(f"{task.bucket}/{task.prefix}", str(e))


# ---------------------------------------------------------------------------
# Iris worker job entrypoint
# ---------------------------------------------------------------------------


def worker_job_entrypoint(
    project: str | None,
    coordinator_actor_name: str,
) -> None:
    """Iris job entrypoint for scan workers.

    Discovers the coordinator actor, then runs WORKER_THREADS threads
    that pull tasks and scan prefixes.
    """
    from iris.actor.client import ActorClient
    from iris.client import iris_ctx

    ctx = iris_ctx()
    resolver = ctx.resolver
    coordinator = ActorClient(resolver, coordinator_actor_name, call_timeout=300.0)

    stop_event = threading.Event()
    threads = []
    for i in range(WORKER_THREADS):
        t = threading.Thread(
            target=_worker_thread_loop,
            args=(coordinator, project, stop_event, f"w{i}"),
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


def discover_top_level_prefixes(
    buckets: list[str],
    project: str | None,
) -> list[ScanTask]:
    """Discover top-level prefixes for each bucket via delimiter listing."""
    client = _make_storage_client(project)
    tasks: list[ScanTask] = []

    for bucket_name in buckets:
        log.info("Discovering prefixes for %s...", bucket_name)
        iterator = client.list_blobs(
            bucket_name,
            delimiter="/",
            fields="items(name),prefixes,nextPageToken",
            timeout=GCS_LIST_TIMEOUT,
        )
        has_root_objects = False
        prefixes: list[str] = []
        for page in iterator.pages:
            prefixes.extend(page.prefixes)
            if not has_root_objects:
                for _ in page:
                    has_root_objects = True
                    break

        if has_root_objects:
            tasks.append(ScanTask(bucket=bucket_name, prefix="", depth=0))
        for p in sorted(set(prefixes)):
            tasks.append(ScanTask(bucket=bucket_name, prefix=p, depth=0))

        log.info("  %s: %d top-level prefixes", bucket_name, len(prefixes) + (1 if has_root_objects else 0))

    return tasks


# ---------------------------------------------------------------------------
# Iris distributed execution
# ---------------------------------------------------------------------------


def run_distributed(
    buckets: list[str],
    num_workers: int,
    project: str | None,
    staging_dir: str,
) -> None:
    """Run the scan as an Iris coordinator job.

    Coordinator accumulates objects in memory and writes consolidated
    parquet segments (~100MB each) to staging_dir on GCS.
    """
    from iris.actor import ActorServer
    from iris.client import iris_ctx
    from iris.cluster.types import Entrypoint, ResourceSpec

    ctx = iris_ctx()
    client = ctx.client

    _truncate_staging_dir(staging_dir)

    # Start coordinator actor
    coordinator = ScanCoordinatorActor(staging_dir)
    actor_name = "scan-coordinator"
    server = ActorServer(host="0.0.0.0")
    server.register(actor_name, coordinator)
    actual_port = server.serve_background()

    from iris.cluster.client import get_job_info

    job_info = get_job_info()
    address = f"http://{job_info.advertise_host}:{actual_port}"
    ctx.registry.register(actor_name, address, {"role": "coordinator"})
    print(f"Coordinator actor registered at {address}")

    # Discover prefixes and load queue
    print(f"Discovering top-level prefixes for {len(buckets)} buckets...")
    tasks = discover_top_level_prefixes(buckets, project)
    coordinator.load_tasks(tasks)
    print(f"Loaded {len(tasks)} initial tasks into queue")

    # Submit one worker job with N replicas
    worker_job = client.submit(
        entrypoint=Entrypoint.from_callable(worker_job_entrypoint, project, actor_name),
        name="scan-workers",
        resources=ResourceSpec(cpu=2, memory="4GB"),
        replicas=num_workers,
    )
    print(f"Submitted worker job with {num_workers} replicas")

    # Monitor progress
    start_time = time.monotonic()
    try:
        while True:
            status = coordinator.get_status()
            elapsed = time.monotonic() - start_time

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
    print(f"  Run report with: uv run scripts/storage/cleanup.py report --parquet-dir {staging_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option("--workers", default=4, type=int, show_default=True, help="Number of Iris worker replicas.")
@click.option("--staging-dir", required=True, help="GCS path for parquet output.")
@click.option("--project", help="GCP project override.")
@click.option("--buckets", help="Comma-separated bucket names. Default: all MARIN_BUCKETS.")
def main(
    workers: int,
    staging_dir: str,
    project: str | None,
    buckets: str | None,
) -> None:
    """Run distributed GCS object scan as an Iris coordinator job.

    Submit via iris job run:

        uv run iris --cluster=marin job run \\
            --cpu 2 --memory 10GB --enable-extra-resources -- \\
            uv run python scripts/storage/distributed_scan.py \\
            --staging-dir gs://marin-us-central2/tmp/storage-scan \\
            --workers 128
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    bucket_list = buckets.split(",") if buckets else MARIN_BUCKETS

    run_distributed(
        buckets=bucket_list,
        num_workers=workers,
        project=project,
        staging_dir=staging_dir,
    )


if __name__ == "__main__":
    main()
