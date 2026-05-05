# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stats schemas emitted by iris workers.

Two namespaces:

- ``iris.worker`` — one row per ping with host-level utilization. Replaces
  the controller's ``worker_resource_history`` table.
- ``iris.task`` — one row per attempt resource update. Replaces the
  controller's ``task_resource_history`` table.

Both schemas use ``ts`` (TIMESTAMP_MS) as the ordering key. They are
registered eagerly in ``Worker.start()`` via
``LogClient.get_table(<namespace>, <dataclass>)`` so schema mismatches surface
on first ping rather than silently dropping rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import ClassVar

from iris.rpc import job_pb2

WORKER_STATS_NAMESPACE = "iris.worker"
TASK_STATS_NAMESPACE = "iris.task"


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

    key_column: ClassVar[str] = "ts"

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

    key_column: ClassVar[str] = "ts"

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


def build_worker_stat(
    *,
    worker_id: str,
    ts: datetime,
    status: str,
    address: str,
    snapshot: job_pb2.WorkerResourceSnapshot,
    metadata: job_pb2.WorkerMetadata,
) -> IrisWorkerStat:
    """Build a heartbeat row from the per-tick snapshot and worker metadata."""
    return IrisWorkerStat(
        worker_id=worker_id,
        ts=ts,
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
    ts: datetime,
    usage: job_pb2.ResourceUsage,
    accelerator_util_pct: float | None = None,
    accelerator_mem_bytes: int | None = None,
) -> IrisTaskStat:
    """Build a per-attempt resource row from a ResourceUsage proto."""
    return IrisTaskStat(
        task_id=task_id,
        attempt_id=attempt_id,
        worker_id=worker_id,
        ts=ts,
        cpu_millicores=int(usage.cpu_millicores),
        memory_mb=int(usage.memory_mb),
        disk_mb=int(usage.disk_mb),
        memory_peak_mb=int(usage.memory_peak_mb),
        accelerator_util_pct=accelerator_util_pct,
        accelerator_mem_bytes=accelerator_mem_bytes,
    )
