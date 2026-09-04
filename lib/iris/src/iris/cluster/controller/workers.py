# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller operations and projections for worker resources."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from rigging.timing import Timestamp
from sqlalchemy import Row, bindparam, case, select

from iris.cluster.controller import ops, reads, tasks
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.backend import BackendCapability, TaskBackend
from iris.cluster.controller.codec import decode_attribute_value, resource_spec_from_scalars, worker_metadata_to_proto
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import (
    job_config_table,
    task_attempts_table,
    tasks_table,
    worker_attributes_table,
    workers_table,
)
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES, AttemptDetailRow
from iris.cluster.controller.worker_health import WorkerHealthTracker, WorkerLiveness
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.auth import AuthzAction, authorize
from iris.time_proto import timestamp_to_proto

logger = logging.getLogger(__name__)

MAX_LIST_WORKERS_LIMIT = 1000
WORKER_TARGET_PREFIX = "/system/worker/"


class WorkerRuntime(Protocol):
    @property
    def backend(self) -> TaskBackend: ...

    @property
    def worker_health(self) -> WorkerHealthTracker: ...

    def all_liveness(self) -> dict[WorkerId, WorkerLiveness]: ...

    def liveness_for_worker(self, worker_id: WorkerId) -> WorkerLiveness: ...

    def request_worker_eviction(self, worker_ids: Sequence[WorkerId]) -> None: ...


@dataclass(frozen=True, slots=True)
class WorkerDependencies:
    db: ControllerDB
    runtime: WorkerRuntime
    auth: ControllerAuth


@dataclass(frozen=True, slots=True)
class WorkerRecord:
    worker_id: WorkerId
    address: str
    total_cpu_millicores: int
    total_memory_bytes: int
    total_gpu_count: int
    total_tpu_count: int
    device_type: str
    device_variant: str
    md_hostname: str
    md_ip_address: str
    md_cpu_count: int
    md_memory_bytes: int
    md_disk_bytes: int
    md_tpu_name: str
    md_tpu_worker_hostnames: str
    md_tpu_worker_id: str
    md_tpu_chips_per_host_bounds: str
    md_gpu_count: int
    md_gpu_name: str
    md_gpu_memory_mb: int
    md_gce_instance_name: str
    md_gce_zone: str
    md_device_json: str
    md_provenance_json: str
    scale_group: str

    @classmethod
    def from_row(cls, row: Row) -> "WorkerRecord":
        return cls(
            worker_id=row.worker_id,
            address=row.address,
            total_cpu_millicores=row.total_cpu_millicores,
            total_memory_bytes=row.total_memory_bytes,
            total_gpu_count=row.total_gpu_count,
            total_tpu_count=row.total_tpu_count,
            device_type=row.device_type,
            device_variant=row.device_variant,
            md_hostname=row.md_hostname,
            md_ip_address=row.md_ip_address,
            md_cpu_count=row.md_cpu_count,
            md_memory_bytes=row.md_memory_bytes,
            md_disk_bytes=row.md_disk_bytes,
            md_tpu_name=row.md_tpu_name,
            md_tpu_worker_hostnames=row.md_tpu_worker_hostnames,
            md_tpu_worker_id=row.md_tpu_worker_id,
            md_tpu_chips_per_host_bounds=row.md_tpu_chips_per_host_bounds,
            md_gpu_count=row.md_gpu_count,
            md_gpu_name=row.md_gpu_name,
            md_gpu_memory_mb=row.md_gpu_memory_mb,
            md_gce_instance_name=row.md_gce_instance_name,
            md_gce_zone=row.md_gce_zone,
            md_device_json=row.md_device_json,
            md_provenance_json=row.md_provenance_json,
            scale_group=row.scale_group,
        )


@dataclass(frozen=True, slots=True)
class WorkerRuntimeRecord:
    worker_id: WorkerId
    address: str
    scale_group: str


@dataclass(frozen=True)
class WorkerDetail:
    worker: WorkerRecord
    attributes: dict[str, str | int | float]
    running_tasks: frozenset[JobName]


def register(
    dependencies: WorkerDependencies,
    request: controller_pb2.Controller.RegisterRequest,
    context: RequestContext,
) -> controller_pb2.Controller.RegisterResponse:
    """Register one worker and defer eviction of any recycled-address owner."""
    del context
    if dependencies.auth.provider is not None:
        authorize(AuthzAction.ACT_AS_WORKER)
    if not request.worker_id:
        logger.error("Worker at %s registered without worker_id", request.address)
        return controller_pb2.Controller.RegisterResponse(worker_id="", accepted=False)

    worker_id = WorkerId(request.worker_id)
    with dependencies.db.transaction() as tx:
        ops.worker.register(
            tx,
            worker_id=worker_id,
            address=request.address,
            metadata=request.metadata,
            ts=Timestamp.now(),
            health=dependencies.runtime.worker_health,
            slice_id=request.slice_id,
            scale_group=request.scale_group,
        )
    _request_recycled_address_eviction(dependencies, worker_id, request.address)
    logger.info("Worker registered: %s at %s", worker_id, request.address)
    return controller_pb2.Controller.RegisterResponse(worker_id=str(worker_id), accepted=True)


def list_workers(
    dependencies: WorkerDependencies,
    request: controller_pb2.Controller.ListWorkersRequest,
    context: RequestContext,
) -> controller_pb2.Controller.ListWorkersResponse:
    """List workers with liveness and running-task counts."""
    del context
    backend = dependencies.runtime.backend
    if BackendCapability.WORKER_FLEET not in backend.descriptor.capabilities:
        return controller_pb2.Controller.ListWorkersResponse()

    query = controller_pb2.Controller.WorkerQuery()
    if request.HasField("query"):
        query.CopyFrom(request.query)

    workers_all = worker_roster(dependencies.db)
    all_liveness = dependencies.runtime.all_liveness()
    liveness_by_id = {
        worker.worker_id: all_liveness.get(worker.worker_id, WorkerLiveness()) for worker, _attrs in workers_all
    }
    if query.backend_id and query.backend_id != backend.descriptor.backend_id:
        workers_all = []
    filtered = _filter_and_sort_workers(workers_all, liveness_by_id, query)
    total_count = len(filtered)

    offset = max(query.offset, 0)
    limit = min(max(query.limit, 0), MAX_LIST_WORKERS_LIMIT)
    if limit > 0:
        page_rows = filtered[offset : offset + limit]
        has_more = offset + limit < total_count
    else:
        page_rows = filtered[offset:] if offset else filtered
        has_more = False

    if page_rows:
        with dependencies.db.read_snapshot() as tx:
            running = reads.running_tasks_by_worker(tx, {worker.worker_id for worker, _attrs in page_rows})
    else:
        running = {}

    workers = []
    for worker, attributes in page_rows:
        liveness = liveness_by_id[worker.worker_id]
        workers.append(
            controller_pb2.Controller.WorkerHealthStatus(
                worker_id=worker.worker_id,
                healthy=liveness.healthy,
                consecutive_failures=liveness.consecutive_failures,
                last_heartbeat=timestamp_to_proto(Timestamp.from_ms(liveness.last_heartbeat_ms)),
                running_job_ids=[task_id.to_wire() for task_id in running.get(worker.worker_id, set())],
                address=worker.address,
                metadata=worker_metadata_to_proto(worker, attributes),
                status_message=status_message(liveness),
                backend_id=backend.descriptor.backend_id,
                scale_group=str(worker.scale_group or ""),
            )
        )
    return controller_pb2.Controller.ListWorkersResponse(
        workers=workers,
        total_count=total_count,
        has_more=has_more,
    )


def get_worker_status(
    dependencies: WorkerDependencies,
    request: controller_pb2.Controller.GetWorkerStatusRequest,
    context: RequestContext,
) -> controller_pb2.Controller.GetWorkerStatusResponse:
    """Return one worker's liveness, metadata, and recent attempts."""
    del context
    backend = dependencies.runtime.backend
    if BackendCapability.WORKER_FLEET not in backend.descriptor.capabilities:
        raise ConnectError(Code.UNIMPLEMENTED, "Direct provider mode: no workers")
    if not request.id:
        raise ConnectError(Code.INVALID_ARGUMENT, "id is required")

    detail = read_worker_detail(dependencies.db, WorkerId(str(request.id)))
    if detail is None:
        raise ConnectError(Code.NOT_FOUND, f"No worker found for '{request.id}'")
    worker = detail.worker
    liveness = dependencies.runtime.liveness_for_worker(worker.worker_id)
    worker_health = controller_pb2.Controller.WorkerHealthStatus(
        worker_id=worker.worker_id,
        healthy=liveness.healthy,
        consecutive_failures=liveness.consecutive_failures,
        last_heartbeat=timestamp_to_proto(Timestamp.from_ms(liveness.last_heartbeat_ms)),
        running_job_ids=[task_id.to_wire() for task_id in detail.running_tasks],
        address=worker.address,
        metadata=worker_metadata_to_proto(worker, detail.attributes),
        status_message=status_message(liveness),
        scale_group=str(worker.scale_group or ""),
        backend_id=backend.descriptor.backend_id,
    )
    response = controller_pb2.Controller.GetWorkerStatusResponse(
        recent_attempts=_attempts_for_worker(dependencies.db, worker.worker_id, limit=50)
    )
    response.worker.CopyFrom(worker_health)
    return response


def read_worker(db: ControllerDB, worker_id: WorkerId) -> WorkerRuntimeRecord | None:
    with db.read_snapshot() as tx:
        row = tx.execute(
            select(workers_table.c.worker_id, workers_table.c.address, workers_table.c.scale_group).where(
                workers_table.c.worker_id == worker_id
            )
        ).first()
    if row is None:
        return None
    return WorkerRuntimeRecord(worker_id=row.worker_id, address=row.address, scale_group=row.scale_group)


def read_worker_detail(db: ControllerDB, worker_id: WorkerId) -> WorkerDetail | None:
    with db.read_snapshot() as tx:
        worker_row = reads.get_worker_detail(tx, worker_id)
        if worker_row is None:
            return None
        worker = WorkerRecord.from_row(worker_row)
        attribute_rows = tx.execute(
            select(
                worker_attributes_table.c.key,
                worker_attributes_table.c.value_type,
                worker_attributes_table.c.str_value,
                worker_attributes_table.c.int_value,
                worker_attributes_table.c.float_value,
            ).where(worker_attributes_table.c.worker_id == worker_id)
        ).all()
        attributes = dict(decode_attribute_value(row) for row in attribute_rows)
        running_rows = tx.execute(
            select(tasks_table.c.task_id)
            .select_from(
                tasks_table.join(
                    task_attempts_table,
                    (tasks_table.c.task_id == task_attempts_table.c.task_id)
                    & (tasks_table.c.current_attempt_id == task_attempts_table.c.attempt_id),
                )
            )
            .where(
                task_attempts_table.c.worker_id == worker_id,
                tasks_table.c.state.in_(bindparam("active_states", expanding=True)),
            ),
            {"active_states": list(ACTIVE_TASK_STATES)},
        ).all()
    return WorkerDetail(
        worker=worker,
        attributes=attributes,
        running_tasks=frozenset(row.task_id for row in running_rows),
    )


def parse_worker_target(target: str) -> str | None:
    if not target.startswith(WORKER_TARGET_PREFIX):
        return None
    worker_id = target[len(WORKER_TARGET_PREFIX) :]
    return worker_id or None


def status_message(liveness: WorkerLiveness) -> str:
    if liveness.healthy:
        return ""
    age_ms = max(0, Timestamp.now().epoch_ms() - liveness.last_heartbeat_ms)
    return f"Unhealthy (last seen {age_ms // 1000}s ago)"


def _request_recycled_address_eviction(
    dependencies: WorkerDependencies,
    worker_id: WorkerId,
    address: str,
) -> None:
    with dependencies.db.read_snapshot() as snapshot:
        stale = reads.worker_ids_at_address(snapshot, address, exclude=worker_id)
    if not stale:
        return
    logger.warning(
        "Worker %s registered at %s held by %d stale row(s) (recycled IP); evicting: %s",
        worker_id,
        address,
        len(stale),
        [str(stale_worker_id) for stale_worker_id in stale],
    )
    dependencies.runtime.request_worker_eviction(stale)


def worker_roster(db: ControllerDB) -> list[tuple[WorkerRecord, dict[str, str | int | float]]]:
    with db.read_snapshot() as tx:
        decoded = [WorkerRecord.from_row(row) for row in tx.execute(select(*reads.WORKER_DETAIL_COLS)).all()]
        if not decoded:
            return []
        worker_ids = [worker.worker_id for worker in decoded]
        attribute_rows = tx.execute(
            select(
                worker_attributes_table.c.worker_id,
                worker_attributes_table.c.key,
                worker_attributes_table.c.value_type,
                worker_attributes_table.c.str_value,
                worker_attributes_table.c.int_value,
                worker_attributes_table.c.float_value,
            ).where(worker_attributes_table.c.worker_id.in_(bindparam("worker_ids", expanding=True))),
            {"worker_ids": list(worker_ids)},
        ).all()
        attributes_by_worker: dict[str, dict[str, str | int | float]] = {}
        for row in attribute_rows:
            key, value = decode_attribute_value(row)
            attributes_by_worker.setdefault(str(row.worker_id), {})[key] = value
    return [(worker, attributes_by_worker.get(str(worker.worker_id), {})) for worker in decoded]


def _filter_and_sort_workers(
    workers: list[tuple[WorkerRecord, dict[str, str | int | float]]],
    liveness_by_id: dict[WorkerId, WorkerLiveness],
    query: controller_pb2.Controller.WorkerQuery,
) -> list[tuple[WorkerRecord, dict[str, str | int | float]]]:
    needle = query.contains.lower() if query.contains else ""
    if needle:
        workers = [
            (worker, attributes)
            for worker, attributes in workers
            if needle in str(worker.worker_id).lower() or (worker.address and needle in worker.address.lower())
        ]

    sort_field = query.sort_field or controller_pb2.Controller.WORKER_SORT_FIELD_WORKER_ID
    descending = query.sort_direction == controller_pb2.Controller.SORT_DIRECTION_DESC
    if sort_field == controller_pb2.Controller.WORKER_SORT_FIELD_LAST_HEARTBEAT:
        return sorted(
            workers,
            key=lambda worker_and_attributes: liveness_by_id[worker_and_attributes[0].worker_id].last_heartbeat_ms,
            reverse=descending,
        )
    if sort_field == controller_pb2.Controller.WORKER_SORT_FIELD_DEVICE_TYPE:
        return sorted(
            workers,
            key=lambda worker_and_attributes: (
                worker_and_attributes[0].device_type,
                str(worker_and_attributes[0].worker_id),
            ),
            reverse=descending,
        )
    return sorted(workers, key=lambda worker_and_attributes: str(worker_and_attributes[0].worker_id), reverse=descending)


def _attempts_for_worker(
    db: ControllerDB,
    worker_id: WorkerId,
    limit: int,
) -> list[controller_pb2.Controller.WorkerTaskAttempt]:
    with db.read_snapshot() as tx:
        raw_rows = tx.execute(
            reads.attempt_select()
            .where(task_attempts_table.c.worker_id == worker_id)
            .order_by(
                case(
                    (task_attempts_table.c.started_at_ms.is_not(None), task_attempts_table.c.started_at_ms),
                    else_=task_attempts_table.c.created_at_ms,
                ).desc()
            )
            .limit(limit)
        ).all()
        job_ids = {row.task_id.parent for row in raw_rows if row.task_id.parent is not None}
        resources_by_job: dict[JobName, job_pb2.ResourceSpecProto] = {}
        if job_ids:
            job_config_rows = tx.execute(
                select(
                    job_config_table.c.job_id,
                    job_config_table.c.res_cpu_millicores,
                    job_config_table.c.res_memory_bytes,
                    job_config_table.c.res_disk_bytes,
                    job_config_table.c.res_device_json,
                ).where(job_config_table.c.job_id.in_(list(job_ids)))
            ).all()
            for row in job_config_rows:
                if row.res_cpu_millicores or row.res_memory_bytes or row.res_disk_bytes or row.res_device_json:
                    resources_by_job[row.job_id] = resource_spec_from_scalars(
                        row.res_cpu_millicores,
                        row.res_memory_bytes,
                        row.res_disk_bytes,
                        row.res_device_json,
                    )

    attempts = []
    for row in raw_rows:
        attempts.append(
            controller_pb2.Controller.WorkerTaskAttempt(
                task_id=row.task_id.to_wire(),
                attempt=tasks.attempt_to_proto(AttemptDetailRow.from_row(row)),
                resources=resources_by_job.get(row.task_id.parent),
            )
        )
    return attempts
