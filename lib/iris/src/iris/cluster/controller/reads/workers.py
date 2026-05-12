# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker read helpers (SA Core port).

Named ``text(...)`` SQL constants and small helpers for the read paths on
:class:`iris.cluster.controller.stores.WorkerStore` and the module-level
``healthy_active_workers_with_attributes`` function in
``iris.cluster.controller.db``. Stage 10 of the SA Core migration introduces
this module alongside the legacy methods; parity tests in
``tests/cluster/controller/test_reads_workers.py`` assert the two paths
return equal results against the same DB state.

Notes:

* The :class:`WorkerHealthTracker` lives in process memory and is not part of
  the SA migration. Functions that combine DB rows with liveness (e.g.
  :func:`list_active_healthy`) take a :class:`WorkerLivenessSource` so
  callers retain control over which tracker view is consulted.
* :func:`get_detail` mirrors ``WORKER_DETAIL_PROJECTION`` field-for-field.
  The legacy projection populates an ``attributes={}`` default; this port
  preserves that — the caller is responsible for hydrating attributes via
  :class:`WorkerAttrsProjection` when needed.
"""

from collections.abc import Iterable
from typing import Protocol

from sqlalchemy import bindparam, text

from iris.cluster.constraints import AttributeValue
from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema import WorkerDetailRow
from iris.cluster.types import WorkerId

# ---------------------------------------------------------------------------
# Liveness protocol
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
# Simple lookups
# ---------------------------------------------------------------------------

_GET_ADDRESS_SQL = text("SELECT address FROM workers WHERE worker_id = :wid")


def address(tx: Tx, worker_id: WorkerId) -> str | None:
    """Return ``workers.address`` for ``worker_id`` or None if absent."""
    row = tx.execute(_GET_ADDRESS_SQL, {"wid": str(worker_id)}).first()
    return str(row.address) if row is not None else None


# ---------------------------------------------------------------------------
# Worker detail
# ---------------------------------------------------------------------------

_WORKER_DETAIL_COLS = (
    "w.worker_id AS worker_id, "
    "w.address AS address, "
    "w.total_cpu_millicores AS total_cpu_millicores, "
    "w.total_memory_bytes AS total_memory_bytes, "
    "w.total_gpu_count AS total_gpu_count, "
    "w.total_tpu_count AS total_tpu_count, "
    "w.device_type AS device_type, "
    "w.device_variant AS device_variant, "
    "w.md_hostname AS md_hostname, "
    "w.md_ip_address AS md_ip_address, "
    "w.md_cpu_count AS md_cpu_count, "
    "w.md_memory_bytes AS md_memory_bytes, "
    "w.md_disk_bytes AS md_disk_bytes, "
    "w.md_tpu_name AS md_tpu_name, "
    "w.md_tpu_worker_hostnames AS md_tpu_worker_hostnames, "
    "w.md_tpu_worker_id AS md_tpu_worker_id, "
    "w.md_tpu_chips_per_host_bounds AS md_tpu_chips_per_host_bounds, "
    "w.md_gpu_count AS md_gpu_count, "
    "w.md_gpu_name AS md_gpu_name, "
    "w.md_gpu_memory_mb AS md_gpu_memory_mb, "
    "w.md_gce_instance_name AS md_gce_instance_name, "
    "w.md_gce_zone AS md_gce_zone, "
    "w.md_git_hash AS md_git_hash, "
    "w.md_device_json AS md_device_json"
)


_GET_DETAIL_SQL = text(f"SELECT {_WORKER_DETAIL_COLS} FROM workers w WHERE w.worker_id = :wid")
_LIST_DETAIL_BY_IDS_SQL = text(f"SELECT {_WORKER_DETAIL_COLS} FROM workers w WHERE w.worker_id IN :wids").bindparams(
    bindparam("wids", expanding=True)
)


def _row_to_worker_detail(row) -> WorkerDetailRow:
    return WorkerDetailRow(
        worker_id=WorkerId(str(row.worker_id)),
        address=str(row.address),
        total_cpu_millicores=int(row.total_cpu_millicores),
        total_memory_bytes=int(row.total_memory_bytes),
        total_gpu_count=int(row.total_gpu_count),
        total_tpu_count=int(row.total_tpu_count),
        device_type=str(row.device_type),
        device_variant=str(row.device_variant),
        md_hostname=str(row.md_hostname),
        md_ip_address=str(row.md_ip_address),
        md_cpu_count=int(row.md_cpu_count),
        md_memory_bytes=int(row.md_memory_bytes),
        md_disk_bytes=int(row.md_disk_bytes),
        md_tpu_name=str(row.md_tpu_name),
        md_tpu_worker_hostnames=str(row.md_tpu_worker_hostnames),
        md_tpu_worker_id=str(row.md_tpu_worker_id),
        md_tpu_chips_per_host_bounds=str(row.md_tpu_chips_per_host_bounds),
        md_gpu_count=int(row.md_gpu_count),
        md_gpu_name=str(row.md_gpu_name),
        md_gpu_memory_mb=int(row.md_gpu_memory_mb),
        md_gce_instance_name=str(row.md_gce_instance_name),
        md_gce_zone=str(row.md_gce_zone),
        md_git_hash=str(row.md_git_hash),
        md_device_json=str(row.md_device_json),
    )


def get_detail(tx: Tx, worker_id: WorkerId) -> WorkerDetailRow | None:
    """Return the full :class:`WorkerDetailRow` for ``worker_id``, or None."""
    row = tx.execute(_GET_DETAIL_SQL, {"wid": str(worker_id)}).first()
    if row is None:
        return None
    return _row_to_worker_detail(row)


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


_LIST_ADDRESSES_SQL = text("SELECT worker_id, address FROM workers WHERE worker_id IN :wids").bindparams(
    bindparam("wids", expanding=True)
)


def list_active_healthy(tx: Tx, health: WorkerLivenessSource) -> dict[WorkerId, str]:
    """Return ``{worker_id: address}`` for all active+healthy workers."""
    liveness = health.all()
    live_ids = [str(wid) for wid, ent in liveness.items() if ent.healthy and ent.active]
    if not live_ids:
        return {}
    rows = tx.execute(_LIST_ADDRESSES_SQL, {"wids": live_ids}).all()
    return {WorkerId(str(row.worker_id)): str(row.address) for row in rows}


def list_active_by_ids(
    tx: Tx,
    worker_ids: Iterable[str],
    health: WorkerLivenessSource,
) -> list[WorkerDetailRow]:
    """Return :class:`WorkerDetailRow` for all active workers whose id is in ``worker_ids``."""
    liveness = health.all()
    ids = sorted(
        {str(wid) for wid in worker_ids if (entry := liveness.get(WorkerId(str(wid)))) is not None and entry.active}
    )
    if not ids:
        return []
    rows = tx.execute(_LIST_DETAIL_BY_IDS_SQL, {"wids": ids}).all()
    return [_row_to_worker_detail(row) for row in rows]


_FILTER_EXISTING_SQL = text("SELECT worker_id FROM workers WHERE worker_id IN :wids").bindparams(
    bindparam("wids", expanding=True)
)


def filter_existing(tx: Tx, worker_ids: Iterable[WorkerId]) -> set[str]:
    """Return the subset of ``worker_ids`` (as strings) that have a ``workers`` row."""
    ids = [str(wid) for wid in worker_ids]
    if not ids:
        return set()
    rows = tx.execute(_FILTER_EXISTING_SQL, {"wids": ids}).all()
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

    Mirrors :func:`iris.cluster.controller.db.healthy_active_workers_with_attributes`
    on the SA Core stack. The return type is
    ``list[iris.cluster.controller.db.SchedulableWorker]``; imported lazily
    to break the db <-> reads import cycle.
    """
    from iris.cluster.controller.db import SchedulableWorker

    liveness = health.all()
    healthy_active = [str(wid) for wid, ent in liveness.items() if ent.healthy and ent.active]
    if not healthy_active:
        return []
    rows = tx.execute(_LIST_DETAIL_BY_IDS_SQL, {"wids": healthy_active}).all()
    if not rows:
        return []
    attrs_by_worker = attrs.all()
    out: list = []
    for row in rows:
        wid = WorkerId(str(row.worker_id))
        out.append(
            SchedulableWorker(
                worker_id=wid,
                address=str(row.address),
                total_cpu_millicores=int(row.total_cpu_millicores),
                total_memory_bytes=int(row.total_memory_bytes),
                total_gpu_count=int(row.total_gpu_count),
                total_tpu_count=int(row.total_tpu_count),
                device_type=str(row.device_type),
                device_variant=str(row.device_variant),
                attributes=attrs_by_worker.get(wid, {}),
            )
        )
    return out
