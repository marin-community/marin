# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate-scoped commands for workers: register/refresh and chunked failure."""

from dataclasses import dataclass

from rigging.timing import Timestamp
from sqlalchemy import bindparam, select

from iris.cluster.constraints import AttributeValue
from iris.cluster.controller import reads, writes
from iris.cluster.controller.audit_logging import log_event
from iris.cluster.controller.codec import proto_to_json
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.reconcile import ControllerEffects, ReconcileState
from iris.cluster.controller.reconcile.commit import commit_effects
from iris.cluster.controller.reconcile.loader import TransitionReader, load_closed_snapshot
from iris.cluster.controller.reconcile.worker import WorkerReconcilePlan, WorkerReconcileResult
from iris.cluster.controller.schema import workers_table
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import AttemptUid, JobName, WorkerId, get_gpu_count, get_tpu_count
from iris.rpc import job_pb2

FAIL_WORKERS_CHUNK_SIZE = 10


@dataclass(frozen=True)
class WorkerFailureBatchResult:
    """Narrow result for :func:`fail`: just the worker rows removed.

    A worker-daemon backend's teardown forwards these IDs to ``backend.autoscale``
    for slice-sibling teardown. Per-task kill targets and log events are already
    applied by ``commit_effects`` inside the batch, so they don't need to surface
    in the return value.
    """

    removed_workers: list[tuple[WorkerId, str | None]]


def register(
    cur: Tx,
    *,
    worker_id: WorkerId,
    address: str,
    metadata: job_pb2.WorkerMetadata,
    ts: Timestamp,
    health: WorkerHealthTracker,
    slice_id: str = "",
    scale_group: str = "",
) -> None:
    """Register a new worker or refresh an existing one. Caller owns the transaction."""
    attr_dict: dict[str, AttributeValue] = {}
    for key, proto in metadata.attributes.items():
        attr_dict[key] = AttributeValue.from_proto(proto)
    now_ms = ts.epoch_ms()
    gpu_count = get_gpu_count(metadata.device)
    tpu_count = get_tpu_count(metadata.device)
    if metadata.device.HasField("gpu"):
        device_type = "gpu"
        device_variant = metadata.device.gpu.variant
    elif metadata.device.HasField("tpu"):
        device_type = "tpu"
        device_variant = metadata.device.tpu.variant
    else:
        device_type = ""
        device_variant = ""
    writes.upsert_worker_row(
        cur,
        {
            "worker_id": worker_id,
            "address": address,
            "total_cpu_millicores": metadata.cpu_count * 1000,
            "total_memory_bytes": metadata.memory_bytes,
            "total_gpu_count": gpu_count,
            "total_tpu_count": tpu_count,
            "device_type": device_type,
            "device_variant": device_variant,
            "slice_id": slice_id,
            "scale_group": scale_group,
            "md_hostname": metadata.hostname,
            "md_ip_address": metadata.ip_address,
            "md_cpu_count": metadata.cpu_count,
            "md_memory_bytes": metadata.memory_bytes,
            "md_disk_bytes": metadata.disk_bytes,
            "md_tpu_name": metadata.tpu_name,
            "md_tpu_worker_hostnames": metadata.tpu_worker_hostnames,
            "md_tpu_worker_id": metadata.tpu_worker_id,
            "md_tpu_chips_per_host_bounds": metadata.tpu_chips_per_host_bounds,
            "md_gpu_count": metadata.gpu_count,
            "md_gpu_name": metadata.gpu_name,
            "md_gpu_memory_mb": metadata.gpu_memory_mb,
            "md_gce_instance_name": metadata.gce_instance_name,
            "md_gce_zone": metadata.gce_zone,
            "md_device_json": proto_to_json(metadata.device),
            "md_provenance_json": proto_to_json(metadata.provenance),
        },
    )
    cur.register(lambda: health.register(worker_id, now_ms=now_ms))
    cur.caches[WorkerAttrsProjection].set(cur, worker_id, attr_dict)
    cur.register(
        lambda: log_event(
            "worker_registered",
            str(worker_id),
            address=address,
        )
    )


def fail(
    db: ControllerDB,
    *,
    worker_ids: list[str],
    reason: str,
    health: WorkerHealthTracker,
) -> WorkerFailureBatchResult:
    """Fail active workers in chunked write transactions.

    Used for slice reaping: when one worker on a multi-VM slice fails, all
    sibling workers on that slice must be failed immediately rather than
    waiting for individual ping timeouts.

    Worker removals are written in chunks of FAIL_WORKERS_CHUNK_SIZE; each
    chunk commits its own transaction so the SQLite writer is released and
    other RPCs can interleave during a zone-wide failure.
    """
    if not worker_ids:
        return WorkerFailureBatchResult(removed_workers=[])
    with db.read_snapshot() as snap:
        liveness = health.all()
        active_ids = sorted(
            {
                WorkerId(str(wid))
                for wid in worker_ids
                if (e := liveness.get(WorkerId(str(wid)))) is not None and e.active
            }
        )
        rows = (
            snap.execute(
                select(*reads.WORKER_DETAIL_COLS).where(
                    workers_table.c.worker_id.in_(bindparam("worker_ids", expanding=True))
                ),
                {"worker_ids": active_ids},
            ).all()
            if active_ids
            else []
        )
    failures: list[tuple[WorkerId, str | None, str]] = [(row.worker_id, row.address, reason) for row in rows]
    if not failures:
        return WorkerFailureBatchResult(removed_workers=[])

    removed_workers: list[tuple[WorkerId, str | None]] = []

    for chunk_start in range(0, len(failures), FAIL_WORKERS_CHUNK_SIZE):
        chunk = failures[chunk_start : chunk_start + FAIL_WORKERS_CHUNK_SIZE]
        with db.transaction() as cur:
            now = Timestamp.now()
            # Re-check liveness inside the write tx: a concurrent reaper
            # may have already failed some workers since we snapshotted.
            live_chunk = [
                (worker_id, worker_address, error)
                for worker_id, worker_address, error in chunk
                if health.liveness(worker_id).active
            ]
            if not live_chunk:
                continue
            _apply_worker_failures_chunk(
                cur,
                live_chunk,
                health=health,
                now=now,
            )
            for worker_id, worker_address, _ in live_chunk:
                if worker_address is not None:
                    removed_workers.append((worker_id, worker_address))

    return WorkerFailureBatchResult(removed_workers=removed_workers)


def _apply_worker_failures_chunk(
    cur: Tx,
    failures: list[tuple[WorkerId, str | None, str]],
    *,
    health: WorkerHealthTracker,
    now: Timestamp,
) -> None:
    """Glue: load the worker slice for ``failures``, run the worker-failure
    kernel, apply effects.

    Per-chunk shape: one closed snapshot, one :class:`Overlay`, then
    ``writes.remove_worker`` after ``commit_effects``.
    """
    worker_ids = [wid for wid, _, _ in failures]
    # Seeding by worker closes every active task on the failed workers (plus
    # their jobs' peer/descendant graph), so the batch derives its per-worker
    # task rows from the snapshot.
    snapshot = load_closed_snapshot(cur, now=now, seed_worker_ids=worker_ids)
    effects = ReconcileState.open(snapshot).fail_workers(failures)

    # commit_effects before remove_worker: task mutations reference attempt rows
    # that would be CASCADE-deleted by remove_worker; order must be preserved.
    commit_effects(cur, effects)
    for worker_id, _, _ in failures:
        writes.remove_worker(cur, worker_id, health=health)


def apply_reconcile(
    source: TransitionReader,
    plan_results: list[tuple[WorkerReconcilePlan, WorkerReconcileResult]],
    *,
    now: Timestamp,
) -> ControllerEffects:
    """Author reconcile effects from the backend's read snapshot (no commit).

    Loads ONE snapshot covering every (plan, result) pair through the backend's
    own read surface, then runs the reconcile kernel once: the pure
    :meth:`ReconcileState.reconcile` shares one ``Overlay`` across all pairs so
    cascade kills triggered by earlier workers are visible to later ones. The
    caller (the backend) folds the returned ``effects.health`` and the controller
    commits the ``effects`` via ``commit_effects``.
    """
    all_task_ids: list[JobName] = []
    all_attempt_keys: list[tuple[JobName, int]] = []
    all_attempt_uids: list[AttemptUid] = []
    all_worker_ids: list[WorkerId] = []

    for plan, result in plan_results:
        all_worker_ids.append(plan.worker_id)

        if result.error is not None:
            for desired in plan.request.desired:
                if not desired.HasField("run") or not desired.run.HasField("request"):
                    continue
                req_proto = desired.run.request
                tid = JobName.from_wire(req_proto.task_id)
                all_task_ids.append(tid)
                all_attempt_keys.append((tid, req_proto.attempt_id))
        else:
            # Extract only plan-scoped UIDs for snapshot preloading (no logging here;
            # worker.filter_observations_to_plan logs dropped observations inline).
            plan_uids = {d.attempt_uid for d in plan.request.desired if d.attempt_uid}
            for obs in result.observations:
                if obs.attempt_uid and obs.attempt_uid in plan_uids:
                    all_attempt_uids.append(AttemptUid(obs.attempt_uid))

    snapshot = source.transition_snapshot(
        now=now,
        seed_worker_ids=all_worker_ids,
        observation_uids=all_attempt_uids,
        seed_task_ids=all_task_ids,
        extra_attempt_keys=all_attempt_keys,
    )
    return ReconcileState.open(snapshot).reconcile(plan_results, now)
