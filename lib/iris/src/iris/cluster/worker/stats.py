# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Finelog stats schemas used by iris.

- ``iris.worker`` / ``iris.task`` — worker-emitted host and per-attempt
  resource rows. Replace the controller's old in-memory history tables.
- ``iris.task_status`` — markdown status text pushed from inside a running
  task via ``RemoteClusterClient.report_task_status_text``.

The ``iris.profile`` schema lives in ``iris.cluster.runtime.profile`` next to
the capture machinery — see ``IrisProfile`` and ``PROFILE_NAMESPACE`` there.

Worker schemas are registered eagerly in ``Worker.start()``; the task-status
schema is registered lazily on first ``report_task_status_text`` call so a
CLI that never touches status text doesn't open a finelog connection.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import ClassVar

from finelog.client import StoragePolicy
from rigging.timing import Timestamp

from iris.rpc import job_pb2

WORKER_STATS_NAMESPACE = "iris.worker"
TASK_STATS_NAMESPACE = "iris.task"
TASK_STATUS_NAMESPACE = "iris.task_status"
TASK_EVENT_NAMESPACE = "iris.task_event"

# Task status rows are only useful while a task is still running — once
# the job ends the data is dead weight on the finelog server. Cap the
# namespace at ~1 hour of history or 100 MiB, whichever fires first.
# Worker / task stats keep the cluster-wide defaults so historical
# resource-usage queries continue to work.
TASK_STATUS_STORAGE_POLICY = StoragePolicy(
    max_bytes=100 * 1024 * 1024,
    max_age_seconds=3600,
)

# Scheduling/admission events are a diagnostic history read only while a job is
# alive (and briefly after, to explain a failure). The producer already
# deduplicates on the verdict so the row count per attempt is small; a short
# retention keeps the timeline cheap on the finelog server.
TASK_EVENT_STORAGE_POLICY = StoragePolicy(
    max_bytes=100 * 1024 * 1024,
    max_age_seconds=3600,
)


def stats_timestamp() -> datetime:
    """Current tz-naive UTC datetime for the stats namespaces' ``ts`` segment key.

    Worker and task stats schemas key their parquet segments on a ``ts``
    datetime column (stored as TIMESTAMP_MS by finelog). Built from rigging's
    ``Timestamp.now()`` so the time source stays consistent with the rest of iris.
    """
    return Timestamp.now().as_naive_utc()


class WorkerStatus(StrEnum):
    """Worker-reported lifecycle state. The controller's liveness verdict is independent."""

    IDLE = "IDLE"
    RUNNING = "RUNNING"


def _attr_string(metadata: job_pb2.WorkerMetadata, key: str) -> str:
    av = metadata.attributes.get(key)
    if av is None:
        return ""
    return av.string_value or ""


@dataclass
class IrisWorkerStat:
    """One row per worker heartbeat (host-level utilization + identity)."""

    # Dashboard reads cluster heartbeats one worker at a time (worker detail page,
    # per-worker task assignment lookups). Clustering by worker_id lets parquet
    # row-group min/max prune scans to a handful of groups; the original ts
    # ordering was correct for the workload but produced wide worker_id ranges
    # in every row group.
    key_column: ClassVar[str] = "worker_id"

    # identity
    worker_id: str
    ts: datetime
    status: str
    address: str
    # per-tick utilization (host-level, from HostMetricsCollector)
    cpu_pct: float
    mem_bytes: int
    mem_total_bytes: int
    disk_used_bytes: int
    disk_total_bytes: int
    running_task_count: int
    total_process_count: int
    net_recv_bytes: int
    net_sent_bytes: int
    # static metadata (replicated each tick — keeps the table self-contained)
    device_type: str
    device_variant: str
    cpu_count: int
    memory_bytes: int
    tpu_name: str
    gce_instance_name: str
    zone: str


@dataclass
class IrisTaskStat:
    """One row per attempt resource update."""

    # Dashboard hot path is ``WHERE task_id IN (...) ORDER BY ts DESC LIMIT 1``
    # per task. Sorting compacted segments by task_id (with seq as the
    # secondary key, monotonic with ts because seq is allocated at the
    # insertion lock) gives parquet row-group pruning on task_id while
    # preserving in-task time order within each group.
    key_column: ClassVar[str] = "task_id"

    task_id: str
    attempt_id: int
    worker_id: str
    ts: datetime
    cpu_millicores: int
    memory_mb: int
    disk_mb: int
    memory_peak_mb: int
    accelerator_util_pct: float | None = None
    accelerator_mem_bytes: int | None = None


@dataclass
class TaskStatusRow:
    """One row per ``report_task_status_text`` push from a running task.

    ``attempt_id`` tiebreaks two attempts colliding within a single millisecond
    during preemption so the newer attempt wins deterministically.
    """

    key_column: ClassVar[str] = "task_id"

    task_id: str
    attempt_id: int
    ts: datetime
    status_text_detail_md: str
    status_text_summary_md: str


@dataclass
class TaskEventRow:
    """One scheduling/admission event observed for a task attempt.

    The "event log for every job": a backend appends a row each time the
    diagnostic verdict for a not-yet-running attempt changes (Kueue admission
    denial, image-pull failure, unschedulable), so the dashboard can render the
    sequence of reasons behind a wait. Read newest-first, filtered by attempt.

    Fields mirror a Kubernetes event so a future producer can relay real events
    unchanged: ``type`` is the severity (``Warning``/``Normal``), ``reason`` the
    short code, ``source`` the emitting layer (e.g. ``k8s/kueue``), ``count`` the
    repeat multiplicity.
    """

    key_column: ClassVar[str] = "task_id"

    task_id: str
    attempt_id: int
    ts: datetime
    type: str
    reason: str
    message: str
    source: str
    count: int


def build_worker_stat(
    *,
    worker_id: str,
    status: str,
    address: str,
    snapshot: job_pb2.WorkerResourceSnapshot,
    metadata: job_pb2.WorkerMetadata,
    ts: datetime | None = None,
) -> IrisWorkerStat:
    """Build a heartbeat row from the per-tick snapshot and worker metadata.

    ``ts`` defaults to :func:`stats_timestamp` (current UTC, tz-naive).
    """
    return IrisWorkerStat(
        worker_id=worker_id,
        ts=ts if ts is not None else stats_timestamp(),
        status=status,
        address=address,
        cpu_pct=float(snapshot.host_cpu_percent),
        mem_bytes=int(snapshot.memory_used_bytes),
        mem_total_bytes=int(snapshot.memory_total_bytes),
        disk_used_bytes=int(snapshot.disk_used_bytes),
        disk_total_bytes=int(snapshot.disk_total_bytes),
        running_task_count=int(snapshot.running_task_count),
        total_process_count=int(snapshot.total_process_count),
        net_recv_bytes=int(snapshot.net_recv_bytes),
        net_sent_bytes=int(snapshot.net_sent_bytes),
        device_type=_attr_string(metadata, "device-type"),
        device_variant=_attr_string(metadata, "device-variant"),
        cpu_count=int(metadata.cpu_count),
        memory_bytes=int(metadata.memory_bytes),
        tpu_name=metadata.tpu_name or "",
        gce_instance_name=metadata.gce_instance_name or "",
        zone=metadata.gce_zone or _attr_string(metadata, "zone"),
    )


def build_task_stat(
    *,
    task_id: str,
    attempt_id: int,
    worker_id: str,
    usage: job_pb2.ResourceUsage,
    ts: datetime | None = None,
    accelerator_util_pct: float | None = None,
    accelerator_mem_bytes: int | None = None,
) -> IrisTaskStat:
    """Build a per-attempt resource row from a ResourceUsage proto.

    ``ts`` defaults to :func:`stats_timestamp` (current UTC, tz-naive).
    """
    return IrisTaskStat(
        task_id=task_id,
        attempt_id=attempt_id,
        worker_id=worker_id,
        ts=ts if ts is not None else stats_timestamp(),
        cpu_millicores=int(usage.cpu_millicores),
        memory_mb=int(usage.memory_mb),
        disk_mb=int(usage.disk_mb),
        memory_peak_mb=int(usage.memory_peak_mb),
        accelerator_util_pct=accelerator_util_pct,
        accelerator_mem_bytes=accelerator_mem_bytes,
    )
