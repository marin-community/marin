# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The finelog stats catalog: every ``iris.*`` namespace and its row schema.

Time-series measurements live in these namespaces rather than the controller
SQLite DB (see AGENTS.md "Decisions vs measurements"). This module is the single
home for the wire contract — namespace names, row dataclasses, storage policies —
that dashboards, Grafana alert rules, and the federation hub key on. Producers
live next to their mechanism (the worker daemon, the k8s backend's collectors,
the autoscaler, the controller's task-state emitter) and import their row types
from here; ``LogStack`` resolves every table from this catalog.

- ``iris.worker`` / ``iris.task`` — worker-emitted host and per-attempt resource
  rows. On Kubernetes clusters, which have no per-node worker daemon, the cluster
  backend emits one ``iris.worker`` row per node (host utilization + GPU
  hardware), so nodes surface as workers in the same dashboards.
- ``iris.task_status`` — markdown status text pushed from inside a running task
  via ``RemoteClusterClient.report_task_status_text``.
- ``iris.task_event`` — scheduling/admission events per task attempt.
- ``iris.profile`` — per-capture profile blobs (capture machinery:
  ``iris.cluster.runtime.profile``).
- ``iris.provisioning`` — one row per slice provisioning outcome (producer:
  the autoscaler).
- ``iris.task_state`` — periodic per-root-job task-state aggregates (producer:
  ``iris.cluster.controller.task_state_stats``).
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
PROFILE_NAMESPACE = "iris.profile"
PROVISIONING_NAMESPACE = "iris.provisioning"
TASK_STATE_NAMESPACE = "iris.task_state"

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
    # Aggregate GPU hardware readings across the host's accelerators, summed
    # (HBM, power) or reduced (mean utilization, hottest temperature) across the
    # devices. None on a host with no accelerator or whose device exporter did
    # not answer. Populated for k8s nodes from the cluster's dcgm-exporter (a
    # worker daemon leaves these unset — its accelerators report per-process
    # usage through the in-task telltale exporter instead).
    gpu_count: int | None = None
    hbm_used_bytes: int | None = None
    hbm_total_bytes: int | None = None
    gpu_util_pct: float | None = None
    gpu_temp_c: float | None = None
    gpu_power_w: float | None = None


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


class ProfileType(StrEnum):
    CPU = "cpu"
    MEMORY = "memory"
    THREAD = "thread"


class ProfileFormat(StrEnum):
    # CPU
    RAW = "raw"
    FLAMEGRAPH = "flamegraph"
    SPEEDSCOPE = "speedscope"
    # Memory
    HTML = "html"
    TABLE = "table"
    STATS = "stats"


class ProfileTrigger(StrEnum):
    PERIODIC = "periodic"
    ON_DEMAND = "on_demand"


@dataclass
class IrisProfile:
    """One row per profile capture. Written by worker / k8s provider / controller; read by dashboard."""

    # The dashboard lists captures per source (a task, a worker, or every task
    # under a job via a source prefix) ordered by captured_at. Clustering segments
    # by source lets parquet row-group min/max prune to the few segments holding
    # that source instead of scanning the whole namespace.
    key_column: ClassVar[str] = "source"

    source: str
    attempt_id: int | None
    vm_id: str
    captured_at: datetime
    duration_seconds: int
    type: str
    format: str
    trigger: str
    rate_hz: int | None = None
    native: bool | None = None
    leaks: bool | None = None
    locals_dump: bool | None = None
    profile_data: bytes = b""

    def __post_init__(self) -> None:
        ProfileType(self.type)
        ProfileFormat(self.format)
        ProfileTrigger(self.trigger)


class ProvisioningOutcome(StrEnum):
    """How a slice's provisioning attempt ended.

    One flat outcome rather than an outcome+cause pair: the create failure modes
    (``STOCKOUT``/``ERROR``) and the runtime death (``PREEMPTED``) are the
    distinctions consumers actually split on. Success rate is
    ``READY / (READY + STOCKOUT + ERROR)``; ``PREEMPTED`` is a post-ready death,
    excluded from it.
    """

    READY = "ready"  # bootstrap succeeded
    STOCKOUT = "stockout"  # create failed: no capacity in the zone
    ERROR = "error"  # create failed: a fault other than stockout
    PREEMPTED = "preempted"  # reached ready, then lost at runtime


@dataclass
class IrisProvisioning:
    """One slice provisioning outcome.

    ``outcome`` is stored as a string (finelog columns are primitive) but always
    holds a :class:`ProvisioningOutcome` value.
    """

    # Pool-level queries (per scale group over time) dominate; clustering parquet
    # by scale_group lets row-group min/max prune scans to a few groups.
    key_column: ClassVar[str] = "scale_group"

    ts: datetime
    resource_type: str  # "tpu" | "gpu" | "cpu"
    scale_group: str  # full authoritative pool name, e.g. tpu_v6e-preemptible_8-europe-west4-a
    zone: str
    accelerator_variant: str  # e.g. "v6e" ("" for cpu)
    outcome: str  # ProvisioningOutcome value
    error_message: str
    worker_count: int
    provision_latency_ms: int  # create→ready wall time; 0 for non-ready outcomes


# ``root_job_id`` of the per-cluster rollup row (the sum over every root job).
CLUSTER_ROLLUP_ROOT_JOB = ""


@dataclass
class IrisTaskState:
    """One aggregate of a root job's waiting/running tasks per tick.

    ``oldest_pending_age_ms`` measures the oldest PENDING task from its last
    requeue (or submission for a first attempt); ``oldest_building_age_ms``
    measures the oldest ASSIGNED-or-BUILDING task from its current attempt's
    creation — time since dispatch without reaching RUNNING, the "tasks stuck
    in BUILDING" alert quantity. Both are 0 when no task is in those states.
    A fully finished root job stops producing rows; terminal history stays
    queryable in the controller DB rather than being re-emitted every tick.
    """

    # Fleet queries slice one root job's history at a time; clustering parquet by
    # root_job_id lets row-group min/max prune the scan.
    key_column: ClassVar[str] = "root_job_id"

    root_job_id: str  # wire job id, or "" for the per-cluster rollup row
    ts: datetime
    pending: int
    assigned: int
    building: int
    running: int
    oldest_pending_age_ms: int
    oldest_building_age_ms: int


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
