# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-heartbeat worker stats schema for the ``iris.worker`` namespace.

Each iris worker emits one :class:`IrisWorkerStat` row per ping into the
finelog stats service so the controller dashboard's worker pane can read
historical fleet state from a service that outlives any single controller
restart. See ``.agents/projects/stats_service/design.md`` for the
end-to-end design and the cutover plan.

The fields are chosen to be the union of:

- what ``Controller.WorkerHealthStatus`` carries today and the dashboard
  Vue (``FleetTab.vue``) actually renders, plus
- per-tick resource utilization the controller does not record itself.

The schema is additive; new columns can be added later as nullable. The
``ts`` column is the explicit ordering key (per the
``ClassVar[str] key_column`` opt-in described in
``spec.md`` "Dataclass schema inference").
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from iris.rpc import job_pb2

WORKER_STATS_NAMESPACE = "iris.worker"
"""Stats-service namespace for per-heartbeat worker rows."""

# Worker-reported lifecycle state strings. These are *self-reported* — the
# controller's view of healthy/unhealthy is independent.
WORKER_STATUS_IDLE = "IDLE"
WORKER_STATUS_RUNNING = "RUNNING"


def _attr_string(metadata: job_pb2.WorkerMetadata, key: str) -> str:
    """Pull a string-typed worker attribute, returning empty if missing."""
    av = metadata.attributes.get(key)
    if av is None:
        return ""
    return av.string_value or ""


@dataclass
class IrisWorkerStat:
    """One row per worker heartbeat.

    The schema is registered once at worker startup via
    ``LogClient.get_table(WORKER_STATS_NAMESPACE, IrisWorkerStat)``; each
    ping appends a new instance through ``Table.write([stat])``.

    Fields mirror today's controller-side ``WorkerHealthStatus`` /
    ``WorkerMetadata`` rendering plus per-tick utilization. Nullable
    fields use ``str | None`` / ``int | None`` so a future schema bump
    that adds more columns can land additively.
    """

    key_column: ClassVar[str] = "ts"

    # --- identity / liveness ---
    worker_id: str
    ts: datetime
    status: str
    """Worker lifecycle stage: ``IDLE`` / ``RUNNING``. The worker reports
    its self-known state; the controller's view (``healthy`` /
    ``consecutive_failures``) is independent and stored separately."""
    address: str
    healthy: bool

    # --- per-tick utilization (host-level, from HostMetricsCollector) ---
    cpu_pct: float
    mem_bytes: int
    mem_total_bytes: int
    disk_used_bytes: int
    disk_total_bytes: int
    running_task_count: int
    total_process_count: int

    # --- metadata (constant per worker — replicated each tick to keep
    # the table self-contained for SQL joins) ---
    device_type: str
    device_variant: str
    cpu_count: int
    memory_bytes: int
    tpu_name: str
    gce_instance_name: str
    zone: str


def build_worker_stat(
    *,
    worker_id: str,
    ts: datetime,
    status: str,
    address: str,
    healthy: bool,
    snapshot: job_pb2.WorkerResourceSnapshot,
    metadata: job_pb2.WorkerMetadata,
) -> IrisWorkerStat:
    """Build a heartbeat row from the worker's existing per-tick data.

    Pulls utilization fields from the resource snapshot the worker already
    builds for the controller's Ping reply, and metadata from the constant
    WorkerMetadata captured at startup. ``zone`` and ``device_type`` /
    ``device_variant`` come from the WorkerMetadata.attributes map populated
    in ``env_probe.build_worker_metadata``.
    """
    return IrisWorkerStat(
        worker_id=worker_id,
        ts=ts,
        status=status,
        address=address,
        healthy=healthy,
        cpu_pct=float(snapshot.host_cpu_percent),
        mem_bytes=int(snapshot.memory_used_bytes),
        mem_total_bytes=int(snapshot.memory_total_bytes),
        disk_used_bytes=int(snapshot.disk_used_bytes),
        disk_total_bytes=int(snapshot.disk_total_bytes),
        running_task_count=int(snapshot.running_task_count),
        total_process_count=int(snapshot.total_process_count),
        device_type=_attr_string(metadata, "device-type"),
        device_variant=_attr_string(metadata, "device-variant"),
        cpu_count=int(metadata.cpu_count),
        memory_bytes=int(metadata.memory_bytes),
        tpu_name=metadata.tpu_name or "",
        gce_instance_name=metadata.gce_instance_name or "",
        zone=metadata.gce_zone or _attr_string(metadata, "zone"),
    )
