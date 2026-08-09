# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical resource operations over controller state and backend observations."""

from rigging.timing import Duration

from iris.backends.protocol import BackendCapability, TaskBackend, TaskTarget
from iris.cluster.controller.attempt import AttemptResources
from iris.cluster.controller.dependencies import ResourceDependencies
from iris.cluster.controller.persistence import reads
from iris.cluster.types import (
    JobName,
)
from iris.resources.endpoint import (
    ExecRequest,
    ExecResult,
    ProfileConfiguration,
    ProfileRequest,
    ProfileResult,
)
from iris.resources.errors import (
    ActionPolicyRejected,
    BackendIdentityUnknown,
    ResourceNotFound,
    ResourceReplaced,
    ResourceSourceUnavailable,
)
from iris.resources.identity import (
    AttemptIdentity,
    AttemptLocator,
)


class ProfileResources:
    """Profile resource operations."""

    def __init__(self, dependencies: ResourceDependencies, attempts: "AttemptResources") -> None:
        self._dependencies = dependencies
        self._attempts = attempts

    def exec_attempt(
        self,
        identity: AttemptIdentity,
        command: tuple[str, ...],
        timeout: Duration | None,
    ) -> ExecResult:
        attempt = self._attempts.describe_attempt(AttemptLocator(identity.task, identity.attempt_number))
        if attempt.summary.identity != identity:
            raise ResourceReplaced(identity.task.resource_id)
        self._require_current_attempt(identity)
        task_id = JobName.from_wire(identity.task.resource_id)
        request = ExecRequest(identity, command, timeout)
        handle = self._federated_handle(task_id)
        if handle is not None:
            return self._dependencies.runtime.federation.proxy_to_peer(
                handle.peer_id,
                lambda peer: peer.exec_in_container(request),
            )
        target, backend = self._task_target(identity)
        self._require_current_attempt(identity)
        return backend.exec_in_container(target, request)

    def profile_attempt(
        self,
        identity: AttemptIdentity,
        profile: ProfileConfiguration | None,
        duration: Duration | None,
    ) -> ProfileResult:
        attempt = self._attempts.describe_attempt(AttemptLocator(identity.task, identity.attempt_number))
        if attempt.summary.identity != identity:
            raise ResourceReplaced(identity.task.resource_id)
        self._require_current_attempt(identity)
        task_id = JobName.from_wire(identity.task.resource_id)
        request = ProfileRequest(identity, profile, duration)
        handle = self._federated_handle(task_id)
        if handle is not None:
            return self._dependencies.runtime.federation.proxy_to_peer(
                handle.peer_id,
                lambda peer: peer.profile_task(request),
            )
        target, backend = self._task_target(identity)
        self._require_current_attempt(identity)
        return backend.profile_task(target, request)

    def _federated_handle(self, task_id: JobName):
        with self._dependencies.db.read_snapshot() as tx:
            return reads.federated_handle(tx, task_id.root_job)

    def _task_target(self, identity: AttemptIdentity) -> tuple[TaskTarget, TaskBackend]:
        task_id = JobName.from_wire(identity.task.resource_id)
        with self._dependencies.db.read_snapshot() as tx:
            task = reads.get_task_detail(tx, task_id)
            attempt = reads.bulk_get_attempts(tx, [(task_id, identity.attempt_number)]).get(
                (task_id, identity.attempt_number)
            )
        if task is None or attempt is None:
            raise ResourceNotFound(identity.task.resource_id)
        if task.current_attempt_id != identity.attempt_number or str(attempt.attempt_uid) != identity.attempt_uid:
            raise ResourceReplaced(f"{identity.task.resource_id}:{identity.attempt_number}")
        backend = self._dependencies.backends[self._backend_id(str(task.backend_id))]
        if BackendCapability.CLUSTER_VIEW in backend.capabilities:
            return (
                TaskTarget(
                    task_id=task_id.to_wire(),
                    attempt_id=identity.attempt_number,
                    worker_id=None,
                    address=None,
                    attempt_uid=identity.attempt_uid,
                ),
                backend,
            )
        worker_id = attempt.worker_id or task.current_worker_id
        if worker_id is None:
            raise ActionPolicyRejected(f"Task {task_id} is not assigned to a node")
        if not self._dependencies.runtime.liveness_for_worker(worker_id).healthy:
            raise ResourceSourceUnavailable(f"Node {worker_id} is unavailable")
        return (
            TaskTarget(
                task_id=task_id.to_wire(),
                attempt_id=identity.attempt_number,
                worker_id=worker_id,
                address=task.current_worker_address,
                attempt_uid=identity.attempt_uid,
            ),
            backend,
        )

    def _require_current_attempt(self, identity: AttemptIdentity) -> None:
        task_id = JobName.from_wire(identity.task.resource_id)
        with self._dependencies.db.read_snapshot() as tx:
            row = reads.current_attempt_identity(tx, task_id)
        if row is None:
            raise ResourceNotFound(identity.task.resource_id)
        if row.attempt_id != identity.attempt_number or str(row.attempt_uid or "") != identity.attempt_uid:
            raise ResourceReplaced(f"{identity.task.resource_id}:{identity.attempt_number}")

    def _backend_id(self, stored: str) -> str:
        if stored:
            if stored not in self._dependencies.backends:
                raise BackendIdentityUnknown(stored)
            return stored
        if len(self._dependencies.backends) == 1:
            return next(iter(self._dependencies.backends))
        raise BackendIdentityUnknown("Task has no retained backend coordinate")
