# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker read helpers.

Return shapes:

* ``get_detail`` — SA ``Row`` or ``None``
* ``list_active_healthy`` — ``dict[WorkerId, str]``
* ``filter_existing`` — ``set[str]``
* ``healthy_active_workers_with_attributes`` — ``list[SchedulableWorker]``

Notes:

* :class:`WorkerLivenessSource` and :class:`WorkerAttrsSource` are kept here
  so callers can satisfy them with the in-memory :class:`WorkerHealthTracker`
  without importing the tracker directly.
* :func:`get_detail` returns a raw SA Row; ``attributes`` must be hydrated
  by the caller if needed.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import bindparam, select

from iris.cluster.constraints import AttributeValue
from iris.cluster.controller.db import Tx
from iris.cluster.controller.schema import workers_table
from iris.cluster.types import WorkerId

# ---------------------------------------------------------------------------
# Liveness / attributes protocols
# ---------------------------------------------------------------------------


class WorkerLivenessSource(Protocol):
    """Read-only view over the in-memory worker liveness tracker."""

    def all(self) -> dict[WorkerId, "_LivenessEntry"]: ...


class _LivenessEntry(Protocol):
    healthy: bool
    active: bool
    last_heartbeat_ms: int


class WorkerAttrsSource(Protocol):
    """Read-only view over the worker_attributes cache."""

    def all(self) -> dict[WorkerId, dict[str, AttributeValue]]: ...


# ---------------------------------------------------------------------------
# Shared column list for worker detail reads
# ---------------------------------------------------------------------------

_WORKER_DETAIL_COLS = (
    workers_table.c.worker_id,
    workers_table.c.address,
    workers_table.c.total_cpu_millicores,
    workers_table.c.total_memory_bytes,
    workers_table.c.total_gpu_count,
    workers_table.c.total_tpu_count,
    workers_table.c.device_type,
    workers_table.c.device_variant,
    workers_table.c.md_hostname,
    workers_table.c.md_ip_address,
    workers_table.c.md_cpu_count,
    workers_table.c.md_memory_bytes,
    workers_table.c.md_disk_bytes,
    workers_table.c.md_tpu_name,
    workers_table.c.md_tpu_worker_hostnames,
    workers_table.c.md_tpu_worker_id,
    workers_table.c.md_tpu_chips_per_host_bounds,
    workers_table.c.md_gpu_count,
    workers_table.c.md_gpu_name,
    workers_table.c.md_gpu_memory_mb,
    workers_table.c.md_gce_instance_name,
    workers_table.c.md_gce_zone,
    workers_table.c.md_git_hash,
    workers_table.c.md_device_json,
)

# ---------------------------------------------------------------------------
# Simple lookups
# ---------------------------------------------------------------------------


def get_detail(tx: Tx, worker_id: WorkerId):
    """Return SA Row for ``worker_id`` or None.

    Row fields: worker_id (WorkerId), address (str), total_cpu_millicores (int),
    total_memory_bytes (int), total_gpu_count (int), total_tpu_count (int),
    device_type (str), device_variant (str), md_hostname (str), md_ip_address
    (str), md_cpu_count (int), md_memory_bytes (int), md_disk_bytes (int),
    md_tpu_name (str), md_tpu_worker_hostnames (str), md_tpu_worker_id (str),
    md_tpu_chips_per_host_bounds (str), md_gpu_count (int), md_gpu_name (str),
    md_gpu_memory_mb (int), md_gce_instance_name (str), md_gce_zone (str),
    md_git_hash (str), md_device_json (str).
    """
    return tx.execute(
        select(*_WORKER_DETAIL_COLS).where(workers_table.c.worker_id == bindparam("worker_id")),
        {"worker_id": worker_id},
    ).first()


# ---------------------------------------------------------------------------
# Liveness-aware list helpers
# ---------------------------------------------------------------------------


def list_active_healthy(tx: Tx, health: WorkerLivenessSource) -> dict[WorkerId, str]:
    """Return ``{worker_id: address}`` for all active+healthy workers."""
    liveness = health.all()
    live_ids = [wid for wid, ent in liveness.items() if ent.healthy and ent.active]
    if not live_ids:
        return {}
    rows = tx.execute(
        select(workers_table.c.worker_id, workers_table.c.address).where(
            workers_table.c.worker_id.in_(bindparam("worker_ids", expanding=True))
        ),
        {"worker_ids": live_ids},
    ).all()
    return {row.worker_id: str(row.address) for row in rows}


def filter_existing(tx: Tx, worker_ids: Iterable[WorkerId]) -> set[str]:
    """Return the subset of ``worker_ids`` (as strings) that have a ``workers`` row."""
    ids = list(worker_ids)
    if not ids:
        return set()
    rows = tx.execute(
        select(workers_table.c.worker_id).where(workers_table.c.worker_id.in_(bindparam("worker_ids", expanding=True))),
        {"worker_ids": ids},
    ).all()
    return {str(row.worker_id) for row in rows}


# ---------------------------------------------------------------------------
# Schedulable worker composite
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SchedulableWorker:
    """Worker shape consumed by the scheduler.

    Field names mirror the :class:`scheduler.WorkerSnapshot` protocol so
    instances flow into ``Scheduler.create_scheduling_context`` without
    an adapter.
    """

    worker_id: WorkerId
    address: str
    total_cpu_millicores: int
    total_memory_bytes: int
    total_gpu_count: int
    total_tpu_count: int
    device_type: str
    device_variant: str
    attributes: dict[str, AttributeValue]


def healthy_active_workers_with_attributes(
    tx: Tx,
    health: WorkerLivenessSource,
    attrs: WorkerAttrsSource,
) -> list[SchedulableWorker]:
    """Return healthy + active workers with their attributes hydrated."""
    liveness = health.all()
    healthy_active = [wid for wid, ent in liveness.items() if ent.healthy and ent.active]
    if not healthy_active:
        return []
    rows = tx.execute(
        select(*_WORKER_DETAIL_COLS).where(workers_table.c.worker_id.in_(bindparam("worker_ids", expanding=True))),
        {"worker_ids": healthy_active},
    ).all()
    if not rows:
        return []
    attrs_by_worker = attrs.all()
    return [
        SchedulableWorker(
            worker_id=row.worker_id,
            address=str(row.address),
            total_cpu_millicores=int(row.total_cpu_millicores),
            total_memory_bytes=int(row.total_memory_bytes),
            total_gpu_count=int(row.total_gpu_count),
            total_tpu_count=int(row.total_tpu_count),
            device_type=str(row.device_type),
            device_variant=str(row.device_variant),
            attributes=attrs_by_worker.get(row.worker_id, {}),
        )
        for row in rows
    ]
