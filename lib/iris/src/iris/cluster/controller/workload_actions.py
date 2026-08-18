# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Durable Job, Task, and Attempt mutations owned by the controller."""

import hashlib
import json
import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from google.protobuf import any_pb2
from google.protobuf.message import Message
from rigging.server_auth import ANONYMOUS_ADMIN, get_verified_identity, require_identity
from rigging.timing import Timestamp
from sqlalchemy import select

from iris.cluster.controller import operation_store, reads, writes
from iris.cluster.controller.auth import ADMIN_ROLE
from iris.cluster.controller.backend import ReconcileResult, RuntimeReleaseTarget
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.operation_store import RuntimeTarget
from iris.cluster.controller.ops import job as job_ops
from iris.cluster.controller.ops.task import finalize
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.controller.schema import federated_jobs_table, jobs_table, task_attempts_table, tasks_table
from iris.cluster.controller.task_state import ACTIVE_TASK_STATES
from iris.cluster.federation.store import FederationDirection, HandoffState
from iris.cluster.types import DEFAULT_BACKEND_ID, LOCAL_CLUSTER, JobName, WorkerId
from iris.rpc import resource_job_pb2, resource_pb2, resource_task_pb2
from iris.rpc.auth import DASHBOARD_ROLE, FEDERATION_PEER_ROLE, authorize_resource_owner
from iris.rpc.resource_registry import ResourceRegistryBuilder

JOB_TYPE = "iris/job"
TASK_TYPE = "iris/task"
ATTEMPT_TYPE = "iris/attempt"

_UID_NAMESPACE = uuid.UUID("2c72b7f4-a156-5d27-8b58-7de28d5ec4cc")
_RETRYABLE_REMOTE_CODES = frozenset(
    {
        Code.ABORTED,
        Code.DEADLINE_EXCEEDED,
        Code.RESOURCE_EXHAUSTED,
        Code.UNAVAILABLE,
    }
)
_TERMINAL_REMOTE_CODES = frozenset(
    {
        Code.INVALID_ARGUMENT,
        Code.NOT_FOUND,
        Code.PERMISSION_DENIED,
        Code.FAILED_PRECONDITION,
        Code.ALREADY_EXISTS,
        Code.UNIMPLEMENTED,
    }
)

logger = logging.getLogger(__name__)


class FederationResourceActions(Protocol):
    def supports_update(self, peer_id: str, resource_type: str, update_type_url: str) -> bool: ...

    def update_resource(self, peer_id: str, request: resource_pb2.UpdateResourceRequest) -> resource_pb2.Operation: ...

    def get_resource(
        self, peer_id: str, request: resource_pb2.GetResourceRequest
    ) -> resource_pb2.GetResourceResponse: ...


@dataclass(frozen=True, slots=True)
class _JobCoordinates:
    job_id: JobName
    submitted_at: Timestamp
    cluster: str
    direction: int | None
    peer_id: str | None
    handoff_state: int | None
    handoff_nonce: str

    def authority(self, local_cluster_id: str) -> str:
        if self.direction == int(FederationDirection.RECEIVED):
            assert self.peer_id is not None
            return self.peer_id
        return local_cluster_id

    def uid(self, local_cluster_id: str) -> str:
        incarnation: object = self.submitted_at.epoch_ms()
        if self.job_id.is_root and self.handoff_nonce:
            incarnation = self.handoff_nonce
        return _uid(JOB_TYPE, self.authority(local_cluster_id), self.job_id.to_wire(), incarnation)


@dataclass(frozen=True, slots=True)
class _TaskCoordinates:
    task_id: JobName
    state: int
    current_attempt_id: int
    cluster: str
    backend_id: str
    job: _JobCoordinates

    def ref(self, local_cluster_id: str) -> resource_pb2.ResourceRef:
        job_uid = self.job.uid(local_cluster_id)
        _, task_index = self.task_id.require_task()
        return _ref(
            self.job.authority(local_cluster_id),
            TASK_TYPE,
            self.task_id.to_wire(),
            _uid(TASK_TYPE, job_uid, task_index),
        )


@dataclass(frozen=True, slots=True)
class _AttemptCoordinates:
    task: _TaskCoordinates
    attempt_id: int
    attempt_uid: str
    worker_id: WorkerId | None
    finished_at: Timestamp | None

    def ref(self, local_cluster_id: str) -> resource_pb2.ResourceRef:
        return _ref(
            self.task.job.authority(local_cluster_id),
            ATTEMPT_TYPE,
            f"{self.task.task_id.to_wire()}:{self.attempt_id}",
            self.attempt_uid,
        )


class WorkloadActions:
    """Accept mutations atomically and resume verification after restart."""

    def __init__(
        self,
        db: ControllerDB,
        *,
        cluster_id: str,
        auth_required: bool,
        federation: FederationResourceActions,
        wake: Callable[[], None],
        mutation_lock: threading.RLock,
        mutation_committed: Callable[[], None],
    ) -> None:
        self._db = db
        self._cluster_id = cluster_id
        self._auth_required = auth_required
        self._federation = federation
        self._wake = wake
        self._mutation_lock = mutation_lock
        self._mutation_committed = mutation_committed

    def logical_ref(self, resource_type: str, resource_id: str) -> resource_pb2.ResourceRef:
        """Return a logical-current reference for a workload resource."""
        with self._db.read_snapshot() as tx:
            if resource_type == JOB_TYPE:
                job = self._job(tx, JobName.from_wire(resource_id))
            elif resource_type == TASK_TYPE:
                job = self._task(tx, JobName.from_wire(resource_id)).job
            elif resource_type == ATTEMPT_TYPE:
                task_id, _ = _parse_attempt_id(resource_id)
                job = self._task(tx, task_id).job
            else:
                raise ValueError(f"unsupported workload resource type {resource_type!r}")
        return resource_pb2.ResourceRef(
            authority_cluster_id=job.authority(self._cluster_id),
            type=resource_type,
            id=resource_id,
        )

    def update_job(
        self,
        request: resource_pb2.UpdateResourceRequest,
        update: resource_job_pb2.JobUpdate,
        _context: RequestContext,
    ) -> resource_pb2.Operation:
        if request.ref.type != JOB_TYPE or update.WhichOneof("intent") != "cancel":
            raise ConnectError(Code.INVALID_ARGUMENT, "Job update requires cancel")
        packed = _pack(update)
        reason = request.mutation.reason or "Cancelled by operator"
        with self._mutation_lock:
            with self._db.transaction() as tx:
                principal = self._principal()
                duplicate = self._replay(tx, principal, request, update)
                if duplicate is not None:
                    return duplicate
                self._authorize_new(tx, request.ref)

                job = self._job(tx, JobName.from_wire(request.ref.id))
                resolved = _ref(job.authority(self._cluster_id), JOB_TYPE, request.ref.id, job.uid(self._cluster_id))
                _require_selector(request.ref, resolved)
                targets = self._job_runtime_targets(tx, job)
                sent = job.direction == int(FederationDirection.SENT)
                queued = sent and job.handoff_state == int(HandoffState.QUEUED_HANDOFF)
                remote = sent and not queued
                if remote and not self._federation.supports_update(job.cluster, request.ref.type, packed.type_url):
                    raise ConnectError(
                        Code.FAILED_PRECONDITION,
                        f"Peer {job.cluster!r} does not support {request.ref.type} update {packed.type_url}",
                    )
                if sent:
                    writes.bump_cancel_intent(tx, job.job_id)
                    if job.handoff_state in (
                        int(HandoffState.QUEUED_HANDOFF),
                        int(HandoffState.PENDING_HANDOFF),
                    ):
                        writes.mark_federated_job_killed(
                            tx,
                            job.job_id,
                            now_ms=Timestamp.now().epoch_ms(),
                            error="Cancelled before handoff",
                        )
                phase = (
                    operation_store.OperationPhase.ACCEPTED
                    if remote
                    else (
                        operation_store.OperationPhase.VERIFYING if targets else operation_store.OperationPhase.VERIFIED
                    )
                )
                operation_id = uuid.uuid4().hex
                operation_store.insert_operation(
                    tx,
                    operation_id=operation_id,
                    principal_id=principal,
                    request_id=request.mutation.request_id,
                    request_hash=_request_hash(request, update),
                    requested_ref=request.ref,
                    resolved_ref=resolved,
                    update=packed,
                    reason=reason,
                    phase=phase,
                    peer_id=job.cluster if remote else "",
                    targets=targets,
                    now=Timestamp.now(),
                )
                if not sent:
                    job_ops.cancel(tx, job_id=job.job_id, reason=reason)
                    if job.direction == int(FederationDirection.RECEIVED):
                        writes.record_federation_change(tx, job.job_id)
                row = operation_store.by_id(tx, operation_id)
                assert row is not None
                operation = operation_store.to_proto(tx, row)
            self._mutation_committed()
        self._wake()
        return operation

    def update_task(
        self,
        request: resource_pb2.UpdateResourceRequest,
        update: resource_task_pb2.TaskUpdate,
        _context: RequestContext,
    ) -> resource_pb2.Operation:
        if request.ref.type != TASK_TYPE:
            raise ConnectError(Code.INVALID_ARGUMENT, "Task update requires a Task ref")
        return self._accept_attempt_update(request, update, self._resolve_task_selector)

    def update_attempt(
        self,
        request: resource_pb2.UpdateResourceRequest,
        update: resource_task_pb2.AttemptUpdate,
        _context: RequestContext,
    ) -> resource_pb2.Operation:
        if request.ref.type != ATTEMPT_TYPE:
            raise ConnectError(Code.INVALID_ARGUMENT, "Attempt update requires an Attempt ref")
        return self._accept_attempt_update(request, update, self._resolve_attempt_selector)

    def get_operation(
        self,
        request: resource_pb2.GetResourceRequest,
        _context: RequestContext,
    ) -> resource_pb2.GetResourceResponse:
        if request.ref.type != operation_store.OPERATION_TYPE:
            raise ConnectError(Code.INVALID_ARGUMENT, "Get operation requires an Operation ref")
        with self._db.read_snapshot() as tx:
            row = operation_store.by_id(tx, request.ref.id)
            if row is None:
                raise ConnectError(Code.NOT_FOUND, f"Operation {request.ref.id!r} not found")
            self._authorize_operation(row.principal_id)
            operation = operation_store.to_proto(tx, row)
        if request.ref.authority_cluster_id != operation.ref.authority_cluster_id:
            raise ConnectError(Code.NOT_FOUND, f"Operation {request.ref.id!r} not found")
        if request.ref.HasField("uid") and request.ref.uid != operation.ref.uid:
            raise ConnectError(Code.FAILED_PRECONDITION, f"Operation {request.ref.id!r} was replaced")
        return resource_pb2.GetResourceResponse(resource=resource_pb2.Resource(ref=operation.ref, body=_pack(operation)))

    def verification_candidates(self) -> dict[str, tuple[RuntimeTarget, ...]]:
        """Snapshot operations eligible for verification by the next reconcile."""
        with self._db.read_snapshot() as tx:
            return operation_store.verification_targets(tx)

    def release_targets(
        self,
        candidates: dict[str, tuple[RuntimeTarget, ...]],
        backend_id: str,
    ) -> tuple[RuntimeReleaseTarget, ...]:
        """Return exact Attempt coordinates one backend must verify as absent."""
        targets: dict[str, RuntimeReleaseTarget] = {}
        for operation_targets in candidates.values():
            for target in operation_targets:
                if target.backend_id != backend_id:
                    continue
                task_id, attempt_id = _parse_attempt_id(target.ref.id)
                targets[target.ref.uid] = RuntimeReleaseTarget(
                    task_id=task_id.to_wire(),
                    attempt_id=attempt_id,
                    attempt_uid=target.ref.uid,
                    worker_id=target.worker_id,
                )
        return tuple(targets[uid] for uid in sorted(targets))

    def verify_released(
        self,
        *,
        verification_candidates: dict[str, tuple[RuntimeTarget, ...]],
        reconcile_results: dict[str, ReconcileResult],
    ) -> None:
        """Advance operations after every exact runtime is observed absent."""
        with self._db.read_snapshot() as tx:
            rows = [row for row in operation_store.pending(tx) if not row.peer_id]
        for row in rows:
            if (
                row.phase == operation_store.OperationPhase.VERIFYING
                and row.operation_id in verification_candidates
                and all(
                    target.backend_id in reconcile_results
                    and target.ref.uid in reconcile_results[target.backend_id].released_attempt_uids
                    for target in verification_candidates[row.operation_id]
                )
            ):
                with self._db.transaction() as tx:
                    current = operation_store.by_id(tx, row.operation_id)
                    if current is not None and current.phase == operation_store.OperationPhase.VERIFYING:
                        operation_store.mark_verified(tx, row.operation_id)

    def progress_remote(self) -> None:
        """Deliver and poll federated operations outside the controller tick lock."""
        with self._db.read_snapshot() as tx:
            rows = [row for row in operation_store.pending(tx) if row.peer_id]
        for row in rows:
            self._progress_remote(row)

    def _accept_attempt_update(
        self,
        request: resource_pb2.UpdateResourceRequest,
        update: resource_task_pb2.TaskUpdate | resource_task_pb2.AttemptUpdate,
        resolve: Callable[
            [Tx, resource_pb2.ResourceRef],
            tuple[_TaskCoordinates, _AttemptCoordinates, resource_pb2.ResourceRef],
        ],
    ) -> resource_pb2.Operation:
        kind, default_reason = _terminal_intent(update.WhichOneof("intent"))
        reason = request.mutation.reason or default_reason
        packed = _pack(update)
        with self._mutation_lock:
            with self._db.transaction() as tx:
                principal = self._principal()
                duplicate = self._replay(tx, principal, request, update)
                if duplicate is not None:
                    return duplicate
                self._authorize_new(tx, request.ref)

                task, attempt, resolved = resolve(tx, request.ref)
                if task.state not in ACTIVE_TASK_STATES or attempt.finished_at is not None:
                    raise ConnectError(Code.FAILED_PRECONDITION, "Task has no active Attempt")
                target = RuntimeTarget(
                    ref=attempt.ref(self._cluster_id),
                    backend_id=task.backend_id or DEFAULT_BACKEND_ID,
                    worker_id=attempt.worker_id,
                )
                remote = task.cluster != LOCAL_CLUSTER
                if remote and not self._federation.supports_update(task.cluster, request.ref.type, packed.type_url):
                    raise ConnectError(
                        Code.FAILED_PRECONDITION,
                        f"Peer {task.cluster!r} does not support {request.ref.type} update {packed.type_url}",
                    )
                operation_id = uuid.uuid4().hex
                operation_store.insert_operation(
                    tx,
                    operation_id=operation_id,
                    principal_id=principal,
                    request_id=request.mutation.request_id,
                    request_hash=_request_hash(request, update),
                    requested_ref=request.ref,
                    resolved_ref=resolved,
                    update=packed,
                    reason=reason,
                    phase=(
                        operation_store.OperationPhase.ACCEPTED if remote else operation_store.OperationPhase.VERIFYING
                    ),
                    peer_id=task.cluster if remote else "",
                    targets=(target,),
                    now=Timestamp.now(),
                )
                if not remote:
                    finalize(
                        tx,
                        [TerminalDecision(kind=kind, task_id=task.task_id, reason=reason)],
                        now=Timestamp.now(),
                    )
                row = operation_store.by_id(tx, operation_id)
                assert row is not None
                operation = operation_store.to_proto(tx, row)
            self._mutation_committed()
        self._wake()
        return operation

    def _resolve_task_selector(
        self,
        tx: Tx,
        requested: resource_pb2.ResourceRef,
    ) -> tuple[_TaskCoordinates, _AttemptCoordinates, resource_pb2.ResourceRef]:
        task = self._task(tx, JobName.from_wire(requested.id))
        resolved = task.ref(self._cluster_id)
        _require_selector(requested, resolved)
        return task, self._current_attempt(tx, task), resolved

    def _resolve_attempt_selector(
        self,
        tx: Tx,
        requested: resource_pb2.ResourceRef,
    ) -> tuple[_TaskCoordinates, _AttemptCoordinates, resource_pb2.ResourceRef]:
        task_id, attempt_id = _parse_attempt_id(requested.id)
        task = self._task(tx, task_id)
        attempt = self._attempt(tx, task, attempt_id)
        resolved = attempt.ref(self._cluster_id)
        _require_selector(requested, resolved)
        if task.current_attempt_id != attempt_id:
            raise ConnectError(Code.FAILED_PRECONDITION, "Attempt is no longer current")
        return task, attempt, resolved

    def _replay(
        self,
        tx: Tx,
        principal: str,
        request: resource_pb2.UpdateResourceRequest,
        update: Message,
    ) -> resource_pb2.Operation | None:
        row = operation_store.by_request(tx, principal, request.mutation.request_id)
        if row is None:
            return None
        if row.request_hash != _request_hash(request, update):
            raise ConnectError(Code.ALREADY_EXISTS, "request_id was already used for a different mutation")
        return operation_store.to_proto(tx, row)

    def _principal(self) -> str:
        identity = get_verified_identity()
        if not self._auth_required:
            return identity.user_id if identity is not None else ANONYMOUS_ADMIN.user_id
        identity = identity or require_identity()
        if identity.role == DASHBOARD_ROLE:
            raise ConnectError(Code.PERMISSION_DENIED, "read-only identity cannot mutate resources")
        return identity.user_id

    def _authorize_new(self, tx: Tx, ref: resource_pb2.ResourceRef) -> None:
        if not self._auth_required:
            return
        resource_name = ref.id.rpartition(":")[0] if ref.type == ATTEMPT_TYPE else ref.id
        job_id = JobName.from_wire(resource_name).root_job
        identity = require_identity()
        if identity.role == FEDERATION_PEER_ROLE:
            handoff = reads.received_handoff(tx, job_id)
            if handoff is None or handoff.requester_id != identity.user_id:
                raise ConnectError(Code.PERMISSION_DENIED, f"Peer {identity.user_id!r} did not federate {job_id}")
            return
        authorize_resource_owner(job_id.user)

    def _authorize_operation(self, principal_id: str) -> None:
        if not self._auth_required:
            return
        identity = require_identity()
        if identity.role != ADMIN_ROLE and identity.user_id != principal_id:
            raise ConnectError(Code.PERMISSION_DENIED, "Operation belongs to another principal")

    def _job(self, tx: Tx, job_id: JobName) -> _JobCoordinates:
        root_federation = federated_jobs_table.alias("root_federation")
        row = tx.execute(
            select(
                jobs_table.c.job_id,
                jobs_table.c.submitted_at_ms,
                jobs_table.c.cluster,
                root_federation.c.direction,
                root_federation.c.peer_id,
                root_federation.c.handoff_state,
                root_federation.c.handoff_nonce,
            )
            .select_from(jobs_table.outerjoin(root_federation, root_federation.c.job_id == jobs_table.c.root_job_id))
            .where(jobs_table.c.job_id == job_id)
        ).first()
        if row is None:
            raise ConnectError(Code.NOT_FOUND, f"Job {job_id} not found")
        return _JobCoordinates(
            job_id=row.job_id,
            submitted_at=row.submitted_at_ms,
            cluster=str(row.cluster),
            direction=int(row.direction) if row.direction is not None else None,
            peer_id=str(row.peer_id) if row.peer_id is not None else None,
            handoff_state=int(row.handoff_state) if row.handoff_state is not None else None,
            handoff_nonce=str(row.handoff_nonce or ""),
        )

    def _task(self, tx: Tx, task_id: JobName) -> _TaskCoordinates:
        job_id, _ = task_id.require_task()
        row = tx.execute(
            select(
                tasks_table.c.task_id,
                tasks_table.c.state,
                tasks_table.c.current_attempt_id,
                tasks_table.c.cluster,
                tasks_table.c.backend_id,
            ).where(tasks_table.c.task_id == task_id)
        ).first()
        if row is None:
            raise ConnectError(Code.NOT_FOUND, f"Task {task_id} not found")
        return _TaskCoordinates(
            task_id=row.task_id,
            state=int(row.state),
            current_attempt_id=int(row.current_attempt_id),
            cluster=str(row.cluster),
            backend_id=str(row.backend_id),
            job=self._job(tx, job_id),
        )

    def _current_attempt(self, tx: Tx, task: _TaskCoordinates) -> _AttemptCoordinates:
        if task.current_attempt_id < 0:
            raise ConnectError(Code.FAILED_PRECONDITION, "Task has no current Attempt")
        return self._attempt(tx, task, task.current_attempt_id)

    def _attempt(self, tx: Tx, task: _TaskCoordinates, attempt_id: int) -> _AttemptCoordinates:
        row = tx.execute(
            select(
                task_attempts_table.c.attempt_id,
                task_attempts_table.c.attempt_uid,
                task_attempts_table.c.worker_id,
                task_attempts_table.c.finished_at_ms,
            ).where(
                task_attempts_table.c.task_id == task.task_id,
                task_attempts_table.c.attempt_id == attempt_id,
            )
        ).first()
        if row is None:
            raise ConnectError(Code.NOT_FOUND, f"Attempt {task.task_id}:{attempt_id} not found")
        return _AttemptCoordinates(
            task=task,
            attempt_id=int(row.attempt_id),
            attempt_uid=str(row.attempt_uid),
            worker_id=row.worker_id,
            finished_at=row.finished_at_ms,
        )

    def _job_runtime_targets(self, tx: Tx, job: _JobCoordinates) -> tuple[RuntimeTarget, ...]:
        subtree = select(jobs_table.c.job_id).where(jobs_table.c.job_id == job.job_id).cte("subtree", recursive=True)
        descendants = jobs_table.alias("descendants")
        subtree = subtree.union_all(
            select(descendants.c.job_id).join(subtree, descendants.c.parent_job_id == subtree.c.job_id)
        )
        rows = tx.execute(
            select(
                tasks_table.c.task_id,
                task_attempts_table.c.attempt_id,
                task_attempts_table.c.attempt_uid,
                tasks_table.c.backend_id,
                task_attempts_table.c.worker_id,
            )
            .select_from(
                tasks_table.join(
                    task_attempts_table,
                    (task_attempts_table.c.task_id == tasks_table.c.task_id)
                    & (task_attempts_table.c.attempt_id == tasks_table.c.current_attempt_id),
                )
            )
            .where(
                tasks_table.c.job_id.in_(select(subtree.c.job_id)),
                tasks_table.c.state.in_(ACTIVE_TASK_STATES),
                task_attempts_table.c.finished_at_ms.is_(None),
            )
        ).all()
        authority = job.authority(self._cluster_id)
        return tuple(
            RuntimeTarget(
                ref=_ref(
                    authority,
                    ATTEMPT_TYPE,
                    f"{row.task_id.to_wire()}:{row.attempt_id}",
                    str(row.attempt_uid),
                ),
                backend_id=str(row.backend_id) or DEFAULT_BACKEND_ID,
                worker_id=row.worker_id,
            )
            for row in rows
        )

    def _progress_remote(self, row: operation_store.OperationRow) -> None:
        try:
            if row.phase == operation_store.OperationPhase.ACCEPTED:
                remote = self._federation.update_resource(row.peer_id, operation_store.forwarded_request(row))
                self._record_remote_phase(row, remote)
                return
            response = self._federation.get_resource(
                row.peer_id,
                resource_pb2.GetResourceRequest(
                    ref=_ref(
                        row.resolved_authority,
                        operation_store.OPERATION_TYPE,
                        row.remote_operation_id,
                        row.remote_operation_id,
                    )
                ),
            )
            remote = resource_pb2.Operation()
            if not response.resource.body.Unpack(remote):
                raise ConnectError(Code.INTERNAL, "peer returned an invalid Operation resource")
            self._record_remote_phase(row, remote)
        except ConnectError as error:
            if error.code == Code.NOT_FOUND and row.resolved_type == JOB_TYPE:
                with self._mutation_lock:
                    with self._db.transaction() as tx:
                        operation_store.mark_verified(tx, row.operation_id)
                        writes.mark_federated_job_killed(
                            tx,
                            JobName.from_wire(row.resolved_id),
                            now_ms=Timestamp.now().epoch_ms(),
                            error="Cancelled (peer reported the job gone)",
                        )
                    self._mutation_committed()
            elif error.code in _TERMINAL_REMOTE_CODES:
                with self._db.transaction() as tx:
                    operation_store.mark_failed(tx, row.operation_id, code=error.code, message=error.message)
            elif error.code in _RETRYABLE_REMOTE_CODES:
                logger.debug("Resource operation %s will retry after peer error: %s", row.operation_id, error)
            else:
                with self._db.transaction() as tx:
                    operation_store.mark_failed(tx, row.operation_id, code=error.code, message=error.message)
        except (ConnectionError, OSError) as error:
            logger.debug("Resource operation %s will retry after peer error: %s", row.operation_id, error)

    def _record_remote_phase(self, row: operation_store.OperationRow, remote: resource_pb2.Operation) -> None:
        with self._db.transaction() as tx:
            current = operation_store.by_id(tx, row.operation_id)
            if current is None or current.phase not in (
                operation_store.OperationPhase.ACCEPTED,
                operation_store.OperationPhase.VERIFYING,
            ):
                return
            expected_target = operation_store.resolved_ref(current)
            if (
                remote.ref.type != operation_store.OPERATION_TYPE
                or not remote.ref.id
                or remote.ref.uid != remote.ref.id
                or remote.ref.authority_cluster_id != expected_target.authority_cluster_id
                or remote.verb != "update"
                or (current.remote_operation_id and remote.ref.id != current.remote_operation_id)
                or remote.resolved_ref != expected_target
            ):
                operation_store.mark_failed(
                    tx,
                    row.operation_id,
                    code=Code.INTERNAL,
                    message="peer returned an operation for a different resource",
                )
                return
            if remote.phase == resource_pb2.OPERATION_PHASE_VERIFIED:
                operation_store.mark_verified(tx, row.operation_id)
            elif remote.phase == resource_pb2.OPERATION_PHASE_FAILED:
                try:
                    code = Code(remote.error.code) if remote.HasField("error") else Code.INTERNAL
                except ValueError:
                    code = Code.INTERNAL
                message = remote.error.message if remote.HasField("error") else "peer operation failed"
                operation_store.mark_failed(tx, row.operation_id, code=code, message=message)
            elif remote.phase in (
                resource_pb2.OPERATION_PHASE_ACCEPTED,
                resource_pb2.OPERATION_PHASE_VERIFYING,
            ):
                operation_store.mark_verifying(tx, row.operation_id, remote_operation_id=remote.ref.id)
            else:
                operation_store.mark_failed(
                    tx,
                    row.operation_id,
                    code=Code.INTERNAL,
                    message="peer returned an invalid operation phase",
                )


def register_workload_actions(registry: ResourceRegistryBuilder, actions: WorkloadActions) -> None:
    """Register workload nouns and their accepted update payloads."""
    registry.bind("/job/update", actions.update_job, payload=resource_job_pb2.JobUpdate)
    registry.bind("/task/update", actions.update_task, payload=resource_task_pb2.TaskUpdate)
    registry.bind("/attempt/update", actions.update_attempt, payload=resource_task_pb2.AttemptUpdate)
    registry.bind("/operation/get", actions.get_operation)


def _uid(resource_type: str, *parts: object) -> str:
    value = "\0".join(("iris-resource-v1", resource_type, *(str(part) for part in parts)))
    return str(uuid.uuid5(_UID_NAMESPACE, value))


def _ref(authority: str, resource_type: str, resource_id: str, uid: str) -> resource_pb2.ResourceRef:
    return resource_pb2.ResourceRef(authority_cluster_id=authority, type=resource_type, id=resource_id, uid=uid)


def _pack(message: Message) -> any_pb2.Any:
    packed = any_pb2.Any()
    packed.Pack(message)
    return packed


def _request_hash(request: resource_pb2.UpdateResourceRequest, update: Message) -> str:
    selector = {
        "authority": request.ref.authority_cluster_id,
        "type": request.ref.type,
        "id": request.ref.id,
        "uid": request.ref.uid if request.ref.HasField("uid") else None,
        "update_type": update.DESCRIPTOR.full_name,
        "update": update.SerializeToString(deterministic=True).hex(),
        "reason": request.mutation.reason,
    }
    return hashlib.sha256(json.dumps(selector, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _require_selector(requested: resource_pb2.ResourceRef, resolved: resource_pb2.ResourceRef) -> None:
    if requested.authority_cluster_id != resolved.authority_cluster_id:
        raise ConnectError(Code.NOT_FOUND, f"Resource {requested.id!r} is not authoritative on this cluster")
    if requested.HasField("uid") and requested.uid != resolved.uid:
        raise ConnectError(Code.FAILED_PRECONDITION, f"Resource {requested.id!r} was replaced")


def _parse_attempt_id(value: str) -> tuple[JobName, int]:
    task_id, separator, ordinal = value.rpartition(":")
    if not separator or not ordinal.isdecimal() or str(int(ordinal)) != ordinal:
        raise ConnectError(Code.INVALID_ARGUMENT, "Attempt id must be '<task>:<non-negative ordinal>'")
    task = JobName.from_wire(task_id)
    task.require_task()
    return task, int(ordinal)


def _terminal_intent(intent: str | None) -> tuple[TerminalKind, str]:
    if intent == "preempt":
        return TerminalKind.PREEMPT, "Preempted by operator"
    if intent == "fail":
        return TerminalKind.FAIL, "Failed by operator"
    if intent == "terminate":
        return TerminalKind.TERMINATE, "Terminated by operator"
    raise ConnectError(Code.INVALID_ARGUMENT, "Task or Attempt update requires preempt, fail, or terminate")
