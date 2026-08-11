# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical resource operations over controller state and backend observations."""

from rigging.timing import Timestamp

from iris.cluster.controller.action import (
    _action_payload_hash,
    _completed_action,
    _CompletedAction,
    _duplicate_action,
    _RemoteActionContext,
    _RemoteTerminalAction,
    _require_idempotency_key,
)
from iris.cluster.controller.dependencies import ResourceDependencies
from iris.cluster.controller.persistence import action as action_persistence
from iris.cluster.controller.persistence import reads
from iris.cluster.controller.persistence.database import Tx
from iris.cluster.controller.persistence.operations.task import finalize
from iris.cluster.controller.reconcile.task import TerminalDecision, TerminalKind
from iris.cluster.controller.resource_identity import (
    _execution_cluster,
    _job_uid,
    _task_uid,
)
from iris.cluster.controller.task import TaskResources
from iris.cluster.federation.protocol import FederationDirection
from iris.cluster.types import LOCAL_ADMIN_SUBMITTER
from iris.resources.action import ActionKind, ActionReceipt, ActionResult
from iris.resources.attempt import AttemptDetail, AttemptRuntimeObject
from iris.resources.errors import (
    ActionIdempotencyConflict,
    ActionPolicyRejected,
    BackendIdentityUnknown,
    ResourceNotFound,
    ResourceReplaced,
)
from iris.resources.identity import (
    AttemptIdentity,
    AttemptLocator,
    JobIdentity,
    ResourceKey,
    ResourceKind,
    TaskIdentity,
)
from iris.resources.names import JobName


class AttemptResources:
    """Attempt resource operations."""

    def __init__(self, dependencies: ResourceDependencies, tasks: "TaskResources") -> None:
        self._dependencies = dependencies
        self._tasks = tasks

    def describe_attempt(self, locator: AttemptLocator) -> AttemptDetail:
        detail = self._tasks.describe_task(locator.task)
        if locator.attempt_number is None:
            identity = detail.summary.current_attempt
            if identity is None:
                raise ResourceNotFound(f"{locator.task.resource_id}:current")
            number = identity.attempt_number
        else:
            number = locator.attempt_number
        attempt = next((candidate for candidate in detail.attempts if candidate.identity.attempt_number == number), None)
        if attempt is None:
            raise ResourceNotFound(f"{locator.task.resource_id}:{number}")
        task_id = JobName.from_wire(locator.task.resource_id)
        with self._dependencies.db.read_snapshot() as tx:
            attempt_row = reads.bulk_get_attempts(tx, [(task_id, number)]).get((task_id, number))
            task_row = reads.get_task_detail(tx, task_id)
        if attempt_row is None or task_row is None:
            raise ResourceNotFound(f"{locator.task.resource_id}:{number}")
        container_id = str(task_row.container_id or "") if number == task_row.current_attempt_id else ""
        return AttemptDetail(
            summary=attempt,
            runtime=self._attempt_runtime(
                attempt_row,
                attempt.execution_cluster_id,
                attempt.backend_id,
                container_id,
            ),
            source_statuses=detail.source_statuses,
        )

    def require_attempt(self, identity: AttemptIdentity) -> AttemptDetail:
        detail = self.describe_attempt(AttemptLocator(identity.task, identity.attempt_number))
        if detail.summary.identity.attempt_uid != identity.attempt_uid:
            raise ResourceReplaced(f"{identity.task.resource_id}:{identity.attempt_number}")
        return detail

    def retry_task(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        idempotency_key: str,
        reason: str = "Requested through the resource API",
        principal_id: str = LOCAL_ADMIN_SUBMITTER,
    ) -> ActionReceipt:
        return self._terminal_action(
            identity,
            expected_attempt_uid=expected_attempt_uid,
            idempotency_key=idempotency_key,
            reason=reason,
            principal_id=principal_id,
            kind=ActionKind.RETRY_TASK,
            terminal_kind=TerminalKind.PREEMPT,
            result=ActionResult.TARGET_ABSENT,
        )

    def terminate_attempt(
        self,
        identity: AttemptIdentity,
        *,
        idempotency_key: str,
        reason: str = "Requested through the resource API",
        principal_id: str = LOCAL_ADMIN_SUBMITTER,
    ) -> ActionReceipt:
        task = self._tasks.describe_task(identity.task).summary.identity
        if identity.attempt_number < 0:
            raise ActionPolicyRejected("attempt_number must be non-negative")
        return self._terminal_action(
            task,
            expected_attempt_uid=identity.attempt_uid,
            expected_attempt_number=identity.attempt_number,
            idempotency_key=idempotency_key,
            reason=reason,
            principal_id=principal_id,
            kind=ActionKind.TERMINATE_ATTEMPT,
            terminal_kind=TerminalKind.TERMINATE,
            result=ActionResult.SATISFIED,
        )

    def fail_attempt(
        self,
        identity: AttemptIdentity,
        *,
        idempotency_key: str,
        reason: str = "Failed through the resource API",
        principal_id: str = LOCAL_ADMIN_SUBMITTER,
    ) -> ActionReceipt:
        task = self._tasks.describe_task(identity.task).summary.identity
        if identity.attempt_number < 0:
            raise ActionPolicyRejected("attempt_number must be non-negative")
        return self._terminal_action(
            task,
            expected_attempt_uid=identity.attempt_uid,
            expected_attempt_number=identity.attempt_number,
            idempotency_key=idempotency_key,
            reason=reason,
            principal_id=principal_id,
            kind=ActionKind.FAIL_ATTEMPT,
            terminal_kind=TerminalKind.TIMEOUT,
            result=ActionResult.SATISFIED,
        )

    def get_action_receipt(self, action_id: str) -> ActionReceipt:
        with self._dependencies.db.read_snapshot() as tx:
            receipt = action_persistence.action_by_id(tx, action_id)
        if receipt is None:
            raise ResourceNotFound(action_id)
        return receipt

    def replay_action(
        self,
        *,
        principal_id: str,
        idempotency_key: str,
        kind: ActionKind,
        reason: str,
    ) -> ActionReceipt | None:
        """Return a matching accepted action before resolving a CURRENT target."""
        with self._dependencies.db.read_snapshot() as tx:
            existing = action_persistence.action_by_idempotency_key(
                tx,
                principal_id=principal_id,
                idempotency_key=_require_idempotency_key(idempotency_key),
            )
        if existing is None:
            return None
        receipt, stored_hash = existing
        expected_hash = _action_payload_hash(
            kind,
            receipt.expected_target_uid,
            receipt.expected_attempt_uid,
            reason,
        )
        if receipt.kind is not kind or stored_hash != expected_hash:
            raise ActionIdempotencyConflict("idempotency key was already used for a different action")
        return receipt

    def _terminal_action(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        idempotency_key: str,
        reason: str,
        principal_id: str,
        kind: ActionKind,
        terminal_kind: TerminalKind,
        result: ActionResult,
        expected_attempt_number: int | None = None,
    ) -> ActionReceipt:
        payload_hash = _action_payload_hash(kind, identity.task_uid, expected_attempt_uid, reason)
        preparation = self._prepare_terminal_action(
            identity,
            expected_attempt_uid=expected_attempt_uid,
            expected_attempt_number=expected_attempt_number,
            idempotency_key=idempotency_key,
            reason=reason,
            principal_id=principal_id,
            kind=kind,
            terminal_kind=terminal_kind,
            result=result,
            payload_hash=payload_hash,
        )
        if isinstance(preparation, _CompletedAction):
            return preparation.receipt

        receipt = self._proxy_terminal_action(
            preparation.context,
            preparation.attempt,
            identity,
            kind=kind,
            idempotency_key=idempotency_key,
            expected_attempt_uid=expected_attempt_uid,
            reason=reason,
        )
        return self._persist_remote_action(
            receipt,
            preparation.context,
            principal_id=principal_id,
            kind=kind,
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
        )

    def _prepare_terminal_action(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        expected_attempt_number: int | None,
        idempotency_key: str,
        reason: str,
        principal_id: str,
        kind: ActionKind,
        terminal_kind: TerminalKind,
        result: ActionResult,
        payload_hash: str,
    ) -> _CompletedAction | _RemoteTerminalAction:
        with self._dependencies.db.transaction() as tx:
            duplicate = _duplicate_action(
                tx,
                principal_id=principal_id,
                kind=kind,
                idempotency_key=idempotency_key,
                payload_hash=payload_hash,
            )
            if duplicate is not None:
                return _CompletedAction(duplicate)
            task_id = JobName.from_wire(identity.key.resource_id)
            row = reads.get_task_detail(tx, task_id)
            if row is None:
                raise ResourceNotFound(identity.key.resource_id)
            job = self._job_rows(tx, {row.job_id})[row.job_id]
            authority = self._authority_cluster(job)
            current_identity = _task_uid(self._job_identity(job).job_uid, row.task_id)
            if identity.key.cluster_id != authority or identity.task_uid != current_identity:
                raise ResourceReplaced(identity.key.resource_id)
            attempt = reads.bulk_get_attempts(tx, [(row.task_id, int(row.current_attempt_id))]).get(
                (row.task_id, int(row.current_attempt_id))
            )
            if attempt is None:
                raise ActionPolicyRejected("Task has no current Attempt")
            if expected_attempt_number is not None and int(attempt.attempt_id) != expected_attempt_number:
                raise ResourceReplaced(f"{identity.key.resource_id}:{expected_attempt_number}")
            if str(attempt.attempt_uid) != expected_attempt_uid:
                raise ResourceReplaced(f"{identity.key.resource_id}:{attempt.attempt_id}")
            handle = reads.federated_handle(tx, row.task_id.root_job)
            if handle is not None:
                remote_attempt = AttemptIdentity(identity.key, int(attempt.attempt_id), str(attempt.attempt_uid))
                remote = _RemoteActionContext(
                    handle.peer_id,
                    authority,
                    str(row.backend_id or ""),
                    _execution_cluster(self._dependencies.cluster_id, str(row.cluster)),
                )
                return _RemoteTerminalAction(remote, remote_attempt)

            finalize(
                tx,
                [
                    TerminalDecision(
                        kind=terminal_kind,
                        task_id=row.task_id,
                        reason=reason,
                    )
                ],
                now=Timestamp.now(),
            )
            target = identity.key
            if kind in {ActionKind.TERMINATE_ATTEMPT, ActionKind.FAIL_ATTEMPT}:
                target = ResourceKey(
                    authority,
                    ResourceKind.ATTEMPT,
                    f"{row.task_id.to_wire()}:{attempt.attempt_id}",
                )
            receipt = _completed_action(
                kind=kind,
                target=target,
                expected_target_uid=identity.task_uid,
                expected_attempt_uid=expected_attempt_uid,
                expected_attempt_number=int(attempt.attempt_id),
                result=result,
            )
            action_persistence.insert_action(
                tx,
                receipt,
                authority_cluster_id=authority,
                authority_action_id=receipt.action_id,
                backend_id=self._backend_id(str(row.backend_id)),
                execution_cluster_id=_execution_cluster(self._dependencies.cluster_id, str(row.cluster)),
                principal_id=principal_id,
                idempotency_key=_require_idempotency_key(idempotency_key),
                payload_hash=payload_hash,
            )
        return _CompletedAction(receipt)

    def _proxy_terminal_action(
        self,
        remote: _RemoteActionContext,
        remote_attempt: AttemptIdentity,
        identity: TaskIdentity,
        *,
        kind: ActionKind,
        idempotency_key: str,
        expected_attempt_uid: str,
        reason: str,
    ) -> ActionReceipt:
        if kind is ActionKind.RETRY_TASK:
            return self._dependencies.runtime.federation.proxy_to_peer(
                remote.peer_id,
                lambda peer: peer.retry_task(
                    identity,
                    expected_attempt_uid=expected_attempt_uid,
                    idempotency_key=idempotency_key,
                    reason=reason,
                ),
            )
        if kind is ActionKind.FAIL_ATTEMPT:
            return self._dependencies.runtime.federation.proxy_to_peer(
                remote.peer_id,
                lambda peer: peer.fail_attempt(
                    remote_attempt,
                    idempotency_key=idempotency_key,
                    reason=reason,
                ),
            )
        return self._dependencies.runtime.federation.proxy_to_peer(
            remote.peer_id,
            lambda peer: peer.terminate_attempt(
                remote_attempt,
                idempotency_key=idempotency_key,
                reason=reason,
            ),
        )

    def _attempt_runtime(
        self,
        attempt,
        execution_cluster_id: str,
        backend_id: str,
        container_id: str,
    ) -> AttemptRuntimeObject | None:
        config = (
            self._dependencies.backend_configs.get(backend_id)
            if execution_cluster_id == self._dependencies.cluster_id
            else None
        )
        provider_kind = ""
        namespace = ""
        provider_node_id = ""
        if config is not None and config.kind == "k8s":
            provider_kind = "kubernetes"
            if config.kubernetes_provider is not None:
                namespace = config.kubernetes_provider.namespace
            provider_node_id = str(attempt.node_name or "")
        elif config is not None and config.kind == "worker_daemon":
            provider_kind = "rpc"
            provider_node_id = str(attempt.worker_id or "")
        name = str(attempt.pod_name or "")
        provider_uid = str(attempt.pod_uid or "")
        if not any((namespace, name, provider_uid, provider_node_id, container_id)):
            return None
        observed_at = attempt.finished_at_ms or attempt.started_at_ms or attempt.created_at_ms
        return AttemptRuntimeObject(
            provider_kind=provider_kind,
            namespace=namespace,
            name=name,
            provider_uid=provider_uid,
            provider_node_id=provider_node_id,
            provider_node_uid="",
            container_id=container_id,
            observed_at=observed_at,
        )

    def _job_identity(self, row: reads.JobCoordinates) -> JobIdentity:
        authority = self._authority_cluster(row)
        return JobIdentity(
            ResourceKey(authority, ResourceKind.JOB, row.job_id.to_wire()),
            _job_uid(
                authority,
                row.job_id,
                row.submitted_at_ms,
                handoff_nonce=str(row.handoff_nonce or ""),
            ),
        )

    def _authority_cluster(self, row: reads.JobCoordinates) -> str:
        if row.direction == int(FederationDirection.RECEIVED):
            return str(row.peer_id)
        return self._dependencies.cluster_id

    def _backend_id(self, stored: str) -> str:
        if stored:
            if stored not in self._dependencies.backends:
                raise BackendIdentityUnknown(stored)
            return stored
        if len(self._dependencies.backends) == 1:
            return next(iter(self._dependencies.backends))
        raise BackendIdentityUnknown("Task has no retained backend coordinate")

    def _job_rows(self, tx: Tx, job_ids: set[JobName]) -> dict[JobName, reads.JobCoordinates]:
        return reads.job_coordinates(tx, job_ids)

    def _persist_remote_action(
        self,
        receipt: ActionReceipt,
        remote: _RemoteActionContext,
        *,
        principal_id: str,
        kind: ActionKind,
        idempotency_key: str,
        payload_hash: str,
    ) -> ActionReceipt:
        with self._dependencies.db.transaction() as tx:
            duplicate = _duplicate_action(
                tx,
                principal_id=principal_id,
                kind=kind,
                idempotency_key=idempotency_key,
                payload_hash=payload_hash,
            )
            if duplicate is not None:
                return duplicate
            action_persistence.insert_action(
                tx,
                receipt,
                authority_cluster_id=remote.authority_cluster_id,
                authority_action_id=receipt.action_id,
                backend_id=remote.backend_id,
                execution_cluster_id=remote.execution_cluster_id,
                principal_id=principal_id,
                idempotency_key=_require_idempotency_key(idempotency_key),
                payload_hash=payload_hash,
            )
        return receipt
