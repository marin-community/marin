# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SA Core write helpers for ``workers`` and ``worker_attributes``.

Stage 11 of the SA Core migration. Ports :meth:`WorkerStore.upsert` and
:meth:`WorkerStore.remove` into module-level functions taking a
:class:`iris.cluster.controller.db_v2.Tx`.

The ``remove_worker`` function carries an explicit
``cascades_into=(worker_attributes_table, task_attempts_table)`` because
the FK ``ON DELETE CASCADE`` from ``workers.worker_id`` deletes from
``worker_attributes`` and ``SET NULL``-s ``task_attempts.worker_id``
without the function knowing. Stage 12's startup check uses this
declaration to verify the ``WorkerAttrsProjection`` reacts to the
cascade.
"""

from sqlalchemy import text

from iris.cluster.controller.db_v2 import Tx
from iris.cluster.controller.schema_v2 import (
    task_attempts_table,
    worker_attributes_table,
    workers_table,
)
from iris.cluster.controller.stores import WorkerUpsertParams
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.controller.writes import writes_to
from iris.cluster.types import WorkerId

_UPSERT_WORKER_SQL = text(
    "INSERT INTO workers("
    "worker_id, address, "
    "total_cpu_millicores, total_memory_bytes, total_gpu_count, total_tpu_count, "
    "device_type, device_variant, slice_id, scale_group, "
    "md_hostname, md_ip_address, md_cpu_count, md_memory_bytes, md_disk_bytes, "
    "md_tpu_name, md_tpu_worker_hostnames, md_tpu_worker_id, md_tpu_chips_per_host_bounds, "
    "md_gpu_count, md_gpu_name, md_gpu_memory_mb, "
    "md_gce_instance_name, md_gce_zone, md_git_hash, md_device_json"
    ") VALUES ("
    ":worker_id, :address, "
    ":total_cpu_millicores, :total_memory_bytes, :total_gpu_count, :total_tpu_count, "
    ":device_type, :device_variant, :slice_id, :scale_group, "
    ":md_hostname, :md_ip_address, :md_cpu_count, :md_memory_bytes, :md_disk_bytes, "
    ":md_tpu_name, :md_tpu_worker_hostnames, :md_tpu_worker_id, :md_tpu_chips_per_host_bounds, "
    ":md_gpu_count, :md_gpu_name, :md_gpu_memory_mb, "
    ":md_gce_instance_name, :md_gce_zone, :md_git_hash, :md_device_json) "
    "ON CONFLICT(worker_id) DO UPDATE SET "
    "address=excluded.address, "
    "total_cpu_millicores=excluded.total_cpu_millicores, "
    "total_memory_bytes=excluded.total_memory_bytes, "
    "total_gpu_count=excluded.total_gpu_count, total_tpu_count=excluded.total_tpu_count, "
    "device_type=excluded.device_type, device_variant=excluded.device_variant, "
    "slice_id=excluded.slice_id, scale_group=excluded.scale_group, "
    "md_hostname=excluded.md_hostname, md_ip_address=excluded.md_ip_address, "
    "md_cpu_count=excluded.md_cpu_count, md_memory_bytes=excluded.md_memory_bytes, "
    "md_disk_bytes=excluded.md_disk_bytes, md_tpu_name=excluded.md_tpu_name, "
    "md_tpu_worker_hostnames=excluded.md_tpu_worker_hostnames, "
    "md_tpu_worker_id=excluded.md_tpu_worker_id, "
    "md_tpu_chips_per_host_bounds=excluded.md_tpu_chips_per_host_bounds, "
    "md_gpu_count=excluded.md_gpu_count, md_gpu_name=excluded.md_gpu_name, "
    "md_gpu_memory_mb=excluded.md_gpu_memory_mb, "
    "md_gce_instance_name=excluded.md_gce_instance_name, md_gce_zone=excluded.md_gce_zone, "
    "md_git_hash=excluded.md_git_hash, md_device_json=excluded.md_device_json"
)

_NULL_OUT_ATTEMPTS_SQL = text("UPDATE task_attempts SET worker_id = NULL WHERE worker_id = :worker_id")
_NULL_OUT_TASKS_SQL = text("UPDATE tasks SET current_worker_id = NULL WHERE current_worker_id = :worker_id")
_DELETE_WORKER_SQL = text("DELETE FROM workers WHERE worker_id = :worker_id")


@writes_to(workers_table)
def upsert_worker(
    tx: Tx,
    params: WorkerUpsertParams,
    now_ms: int,
    health: WorkerHealthTracker,
) -> None:
    """Insert or refresh durable identity / capability metadata for a worker.

    Resource usage is derived per-cycle from unfinished worker-bound
    ``task_attempts``; the legacy ``committed_*`` columns were dropped
    by migration 0043. A post-commit hook registers the worker in the
    liveness tracker so in-memory state advances atomically with the
    DB row.
    """
    tx.execute(
        _UPSERT_WORKER_SQL,
        {
            "worker_id": str(params.worker_id),
            "address": params.address,
            "total_cpu_millicores": params.total_cpu_millicores,
            "total_memory_bytes": params.total_memory_bytes,
            "total_gpu_count": params.total_gpu_count,
            "total_tpu_count": params.total_tpu_count,
            "device_type": params.device_type,
            "device_variant": params.device_variant,
            "slice_id": params.slice_id,
            "scale_group": params.scale_group,
            "md_hostname": params.md_hostname,
            "md_ip_address": params.md_ip_address,
            "md_cpu_count": params.md_cpu_count,
            "md_memory_bytes": params.md_memory_bytes,
            "md_disk_bytes": params.md_disk_bytes,
            "md_tpu_name": params.md_tpu_name,
            "md_tpu_worker_hostnames": params.md_tpu_worker_hostnames,
            "md_tpu_worker_id": params.md_tpu_worker_id,
            "md_tpu_chips_per_host_bounds": params.md_tpu_chips_per_host_bounds,
            "md_gpu_count": params.md_gpu_count,
            "md_gpu_name": params.md_gpu_name,
            "md_gpu_memory_mb": params.md_gpu_memory_mb,
            "md_gce_instance_name": params.md_gce_instance_name,
            "md_gce_zone": params.md_gce_zone,
            "md_git_hash": params.md_git_hash,
            "md_device_json": params.md_device_json,
        },
    )
    tx.register(lambda: health.register(params.worker_id, now_ms=now_ms))


@writes_to(workers_table, cascades_into=(worker_attributes_table, task_attempts_table))
def remove_worker(tx: Tx, worker_id: WorkerId, health: WorkerHealthTracker) -> None:
    """Delete a worker row and clear back-references on attempts / tasks.

    ``cascades_into`` records the FK fanout: ``worker_attributes`` is
    deleted via ``ON DELETE CASCADE`` (Projection-owned; Stage 12 wires
    in :meth:`WorkerAttrsProjection.invalidate_for_worker`),
    ``task_attempts.worker_id`` is ``SET NULL`` via the FK. The
    pre-emptive ``UPDATE`` on ``task_attempts`` / ``tasks`` matches
    today's :meth:`WorkerStore.remove` byte-for-byte: it sets
    ``current_worker_*`` to NULL before the cascade so the row history
    is observable to readers in the same write transaction.
    """
    wid = str(worker_id)
    tx.execute(_NULL_OUT_ATTEMPTS_SQL, {"worker_id": wid})
    tx.execute(_NULL_OUT_TASKS_SQL, {"worker_id": wid})
    tx.execute(_DELETE_WORKER_SQL, {"worker_id": wid})
    tx.register(lambda: health.forget(worker_id))
