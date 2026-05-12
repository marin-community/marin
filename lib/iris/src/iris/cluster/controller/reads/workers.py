# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker read helpers (SA Core expression language).

All queries use ``select(table.c.col, ...)`` rather than ``text("SELECT
...")``. TypeDecorators on the schema_v2 columns decode worker_id to
WorkerId on read.

Return shapes:

* ``address`` — ``str | None``
* ``get_detail`` — SA ``Row`` or ``None``
* ``active_healthy_address`` — ``str | None``
* ``list_active_healthy`` — ``dict[WorkerId, str]``
* ``list_active_by_ids`` — ``list[Row]``
* ``filter_existing`` — ``set[str]``
* ``healthy_active_workers_with_attributes`` — ``list[SchedulableWorker]``

Notes:

* The :class:`WorkerLivenessSource` and :class:`WorkerAttrsSource` protocols
  are kept here so callers can satisfy them with the in-memory
  :class:`WorkerHealthTracker` without importing the tracker directly.
* :func:`get_detail` returns a raw SA Row; the ``attributes={}`` default
  that the legacy projection provided must be hydrated by the caller if
  needed.
"""

from collections.abc import Iterable
from typing import Protocol

from sqlalchemy import bindparam, select

from iris.cluster.constraints import AttributeValue
from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema_v2 import workers_table
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

GET_ADDRESS_QUERY = select(workers_table.c.address).where(workers_table.c.worker_id == bindparam("worker_id"))

GET_DETAIL_QUERY = select(*_WORKER_DETAIL_COLS).where(workers_table.c.worker_id == bindparam("worker_id"))

LIST_DETAIL_BY_IDS_QUERY = select(*_WORKER_DETAIL_COLS).where(
    workers_table.c.worker_id.in_(bindparam("worker_ids", expanding=True))
)

LIST_ADDRESSES_QUERY = select(workers_table.c.worker_id, workers_table.c.address).where(
    workers_table.c.worker_id.in_(bindparam("worker_ids", expanding=True))
)

FILTER_EXISTING_QUERY = select(workers_table.c.worker_id).where(
    workers_table.c.worker_id.in_(bindparam("worker_ids", expanding=True))
)


def address(tx: Tx, worker_id: WorkerId) -> str | None:
    """Return ``workers.address`` for ``worker_id`` or None if absent."""
    row = tx.execute(GET_ADDRESS_QUERY, {"worker_id": worker_id}).first()
    return str(row.address) if row is not None else None


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
    return tx.execute(GET_DETAIL_QUERY, {"worker_id": worker_id}).first()


# ---------------------------------------------------------------------------
# Liveness-aware list helpers
# ---------------------------------------------------------------------------


def active_healthy_address(
    tx: Tx,
    worker_id: WorkerId,
    health: WorkerLivenessSource,
) -> str | None:
    """Return the worker's address only if it is currently healthy + active."""
    liveness = health.all().get(worker_id)
    if liveness is None or not (liveness.healthy and liveness.active):
        return None
    return address(tx, worker_id)


def list_active_healthy(tx: Tx, health: WorkerLivenessSource) -> dict[WorkerId, str]:
    """Return ``{worker_id: address}`` for all active+healthy workers."""
    liveness = health.all()
    live_ids = [wid for wid, ent in liveness.items() if ent.healthy and ent.active]
    if not live_ids:
        return {}
    rows = tx.execute(LIST_ADDRESSES_QUERY, {"worker_ids": live_ids}).all()
    return {row.worker_id: str(row.address) for row in rows}


def list_active_by_ids(
    tx: Tx,
    worker_ids: Iterable[str],
    health: WorkerLivenessSource,
) -> list:
    """Return detail Rows for active workers whose id is in ``worker_ids``.

    Returns list[Row]. Only workers that appear in the liveness tracker as
    active are included.
    """
    liveness = health.all()
    ids = sorted(
        {
            WorkerId(str(wid))
            for wid in worker_ids
            if (entry := liveness.get(WorkerId(str(wid)))) is not None and entry.active
        }
    )
    if not ids:
        return []
    return tx.execute(LIST_DETAIL_BY_IDS_QUERY, {"worker_ids": ids}).all()


def filter_existing(tx: Tx, worker_ids: Iterable[WorkerId]) -> set[str]:
    """Return the subset of ``worker_ids`` (as strings) that have a ``workers`` row."""
    ids = list(worker_ids)
    if not ids:
        return set()
    rows = tx.execute(FILTER_EXISTING_QUERY, {"worker_ids": ids}).all()
    return {str(row.worker_id) for row in rows}


# ---------------------------------------------------------------------------
# Schedulable worker composite (db.healthy_active_workers_with_attributes)
# ---------------------------------------------------------------------------


def healthy_active_workers_with_attributes(
    tx: Tx,
    health: WorkerLivenessSource,
    attrs: WorkerAttrsSource,
) -> list:
    """Return healthy + active workers with their attributes hydrated.

    Mirrors :func:`iris.cluster.controller.db.healthy_active_workers_with_attributes`.
    Returns ``list[iris.cluster.controller.db.SchedulableWorker]``; the
    SchedulableWorker type is imported lazily to avoid an import cycle.
    """
    from iris.cluster.controller.db import SchedulableWorker

    liveness = health.all()
    healthy_active = [wid for wid, ent in liveness.items() if ent.healthy and ent.active]
    if not healthy_active:
        return []
    rows = tx.execute(LIST_DETAIL_BY_IDS_QUERY, {"worker_ids": healthy_active}).all()
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
