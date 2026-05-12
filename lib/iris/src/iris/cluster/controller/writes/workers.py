# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SA Core write helpers for ``workers`` and ``worker_attributes``.

Stage M2 of the SA Core migration: replaces raw ``text("INSERT/UPDATE/DELETE ...")``
strings with SA Core expression-language constructs. TypeDecorators handle
all bind-side conversions automatically.

The ``remove_worker`` function declares ``cascades_into=(task_attempts_table,)``
only — the FK ``ON DELETE CASCADE`` from ``workers.worker_id`` also
deletes from ``worker_attributes`` (Projection-owned), but Stage 12's
startup check forbids declaring that cascade because it would route a
mutation around :class:`WorkerAttrsProjection`. Instead,
``remove_worker`` calls :meth:`WorkerAttrsProjection.invalidate_for_worker`
inline so the cascade and the dict update commit atomically.
"""

from sqlalchemy import delete, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from iris.cluster.controller.db import Tx
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.schema import task_attempts_table, tasks_table, workers_table
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.controller.writes import writes_to
from iris.cluster.types import WorkerId


@writes_to(workers_table)
def upsert_worker(
    tx: Tx,
    *,
    worker_id: WorkerId,
    address: str,
    total_cpu_millicores: int,
    total_memory_bytes: int,
    total_gpu_count: int,
    total_tpu_count: int,
    device_type: str,
    device_variant: str,
    slice_id: str,
    scale_group: str,
    md_hostname: str,
    md_ip_address: str,
    md_cpu_count: int,
    md_memory_bytes: int,
    md_disk_bytes: int,
    md_tpu_name: str,
    md_tpu_worker_hostnames: str,
    md_tpu_worker_id: str,
    md_tpu_chips_per_host_bounds: str,
    md_gpu_count: int,
    md_gpu_name: str,
    md_gpu_memory_mb: int,
    md_gce_instance_name: str,
    md_gce_zone: str,
    md_git_hash: str,
    md_device_json: str,
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
    row_values = {
        "worker_id": worker_id,
        "address": address,
        "total_cpu_millicores": total_cpu_millicores,
        "total_memory_bytes": total_memory_bytes,
        "total_gpu_count": total_gpu_count,
        "total_tpu_count": total_tpu_count,
        "device_type": device_type,
        "device_variant": device_variant,
        "slice_id": slice_id,
        "scale_group": scale_group,
        "md_hostname": md_hostname,
        "md_ip_address": md_ip_address,
        "md_cpu_count": md_cpu_count,
        "md_memory_bytes": md_memory_bytes,
        "md_disk_bytes": md_disk_bytes,
        "md_tpu_name": md_tpu_name,
        "md_tpu_worker_hostnames": md_tpu_worker_hostnames,
        "md_tpu_worker_id": md_tpu_worker_id,
        "md_tpu_chips_per_host_bounds": md_tpu_chips_per_host_bounds,
        "md_gpu_count": md_gpu_count,
        "md_gpu_name": md_gpu_name,
        "md_gpu_memory_mb": md_gpu_memory_mb,
        "md_gce_instance_name": md_gce_instance_name,
        "md_gce_zone": md_gce_zone,
        "md_git_hash": md_git_hash,
        "md_device_json": md_device_json,
    }
    # ON CONFLICT(worker_id) DO UPDATE SET all columns except the PK.
    update_set = {k: v for k, v in row_values.items() if k != "worker_id"}
    tx.execute(
        sqlite_insert(workers_table)
        .values(**row_values)
        .on_conflict_do_update(index_elements=["worker_id"], set_=update_set)
    )
    tx.register(lambda: health.register(worker_id, now_ms=now_ms))


@writes_to(workers_table, cascades_into=(task_attempts_table,))
def remove_worker(
    tx: Tx,
    worker_id: WorkerId,
    health: WorkerHealthTracker,
    worker_attrs: WorkerAttrsProjection,
) -> None:
    """Delete a worker row and clear back-references on attempts / tasks.

    ``cascades_into`` records the FK fanout to ``task_attempts``
    (``SET NULL`` via FK). The cascade into ``worker_attributes`` is
    Projection-owned and therefore *not* declared on the decorator;
    instead this function calls
    :meth:`WorkerAttrsProjection.invalidate_for_worker` inline so the
    dict update commits under the same write lock as the SQL. The
    pre-emptive ``UPDATE`` on ``task_attempts`` / ``tasks`` matches
    today's `writes.workers.remove_worker` byte-for-byte: it sets
    ``current_worker_*`` to NULL before the cascade so the row history
    is observable to readers in the same write transaction.
    """
    tx.execute(update(task_attempts_table).where(task_attempts_table.c.worker_id == worker_id).values(worker_id=None))
    tx.execute(update(tasks_table).where(tasks_table.c.current_worker_id == worker_id).values(current_worker_id=None))
    tx.execute(delete(workers_table).where(workers_table.c.worker_id == worker_id))
    worker_attrs.invalidate_for_worker(tx, worker_id)
    tx.register(lambda: health.forget(worker_id))
