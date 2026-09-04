# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller operations that perform exact, on-demand Attempt I/O."""

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from finelog.client.log_client import Table
from rigging.connect import capability_path, federated_capability_path
from rigging.server_auth import get_verified_identity
from rigging.timing import Timer, Timestamp

from iris.cluster.controller import reads, tasks, workers
from iris.cluster.controller.auth import (
    DEFAULT_ENDPOINT_TOKEN_TTL_SECONDS,
    MAX_ENDPOINT_TOKEN_TTL_SECONDS,
    ControllerAuth,
    authorize_owner_if_configured,
)
from iris.cluster.controller.backend import BackendCapability, ProviderError, TaskBackend, TaskTarget
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.projections.endpoints import EndpointRow
from iris.cluster.controller.worker_health import WorkerLiveness
from iris.cluster.federation.manager import FederationManager
from iris.cluster.federation.peer import FederationPeer
from iris.cluster.process_status import get_process_status as local_process_status
from iris.cluster.runtime.profile import build_profile_row, profile_local_process
from iris.cluster.types import JobName, TaskAttempt, WorkerId
from iris.rpc import controller_pb2, job_pb2, worker_pb2
from iris.rpc.auth import FEDERATION_PEER_ROLE
from iris.time_proto import duration_from_proto, timestamp_to_proto

Response = TypeVar("Response")


@dataclass(frozen=True, slots=True)
class CapabilityUrlConfig:
    cluster_name: str = ""
    local_origin: str = ""
    parent_origin: str = ""

    def build(self, name: str, token: str) -> str:
        if self.parent_origin and self.cluster_name:
            return f"{self.parent_origin.rstrip('/')}{federated_capability_path(self.cluster_name, name, token)}"
        if self.local_origin:
            return f"{self.local_origin.rstrip('/')}{capability_path(name, token)}"
        return ""


class AttemptRuntime(Protocol):
    @property
    def backend(self) -> TaskBackend: ...

    @property
    def federation(self) -> FederationManager: ...

    def liveness_for_worker(self, worker_id: WorkerId) -> WorkerLiveness: ...


class EndpointLookup(Protocol):
    def resolve_task_endpoint(self, name: str) -> EndpointRow | None: ...


@dataclass(frozen=True, slots=True)
class AttemptDependencies:
    db: ControllerDB
    runtime: AttemptRuntime
    auth: ControllerAuth
    endpoints: EndpointLookup
    profile_table: Table
    capability_urls: CapabilityUrlConfig
    timer: Timer


def profile_task(
    dependencies: AttemptDependencies,
    request: job_pb2.ProfileTaskRequest,
    context: RequestContext,
) -> job_pb2.ProfileTaskResponse:
    """Capture a profile from the controller, a worker, or a task Attempt."""
    del context
    if not request.HasField("profile_type"):
        raise ConnectError(Code.INVALID_ARGUMENT, "profile_type is required")

    if request.target in ("/system/controller", "/system/process"):
        try:
            duration = request.duration_seconds or 10
            data = profile_local_process(duration, request.profile_type)
            dependencies.profile_table.write(
                [
                    build_profile_row(
                        source="/system/controller",
                        attempt_id=None,
                        vm_id="controller-self",
                        duration_seconds=duration,
                        profile_type=request.profile_type,
                        profile_data=data,
                    )
                ]
            )
            return job_pb2.ProfileTaskResponse(profile_data=data)
        except Exception as error:
            return job_pb2.ProfileTaskResponse(error=str(error))

    worker_id = workers.parse_worker_target(request.target)
    if worker_id is not None:
        worker = workers.read_worker(dependencies.db, WorkerId(worker_id))
        if worker is None:
            raise ConnectError(Code.NOT_FOUND, f"Worker {worker_id} not found")
        if not dependencies.runtime.liveness_for_worker(worker.worker_id).healthy:
            raise ConnectError(Code.UNAVAILABLE, f"Worker {worker_id} is unavailable")
        forwarded = job_pb2.ProfileTaskRequest(
            target="/system/process",
            duration_seconds=request.duration_seconds,
            profile_type=request.profile_type,
        )
        response = dependencies.runtime.backend.profile_task(
            TaskTarget(task_id="", attempt_id=0, worker_id=worker.worker_id, address=worker.address),
            forwarded,
            (request.duration_seconds or 10) * 1000 + 30000,
        )
        return job_pb2.ProfileTaskResponse(profile_data=response.profile_data, error=response.error)

    try:
        target = TaskAttempt.from_wire(request.target)
        target.task_id.require_task()
    except ValueError as error:
        raise ConnectError(Code.INVALID_ARGUMENT, str(error)) from error
    _authorize_federated_debug_target(dependencies, target.task_id.root_job)
    task = tasks.read_task_with_attempts(dependencies.db, target.task_id)
    if task is None:
        raise ConnectError(Code.NOT_FOUND, f"Task {request.target} not found")

    proxied = _proxy_if_federated(dependencies, target.task_id, lambda peer: peer.profile_task(request))
    if proxied is not None:
        return proxied
    attempt_id = target.attempt_id if target.attempt_id is not None else task.current_attempt_id
    task_target = _resolve_task_target(dependencies, task, attempt_id, wire_name=request.target)
    response = dependencies.runtime.backend.profile_task(
        task_target,
        request,
        (request.duration_seconds or 10) * 1000 + 30000,
    )
    return job_pb2.ProfileTaskResponse(profile_data=response.profile_data, error=response.error)


def get_process_status(
    dependencies: AttemptDependencies,
    request: job_pb2.GetProcessStatusRequest,
    context: RequestContext,
) -> job_pb2.GetProcessStatusResponse:
    """Return controller, worker, or task-container process information."""
    del context
    target = request.target
    if not target or target == "/system/process":
        return local_process_status(dependencies.timer)

    worker_id = workers.parse_worker_target(target)
    if worker_id is None:
        return _task_process_status(dependencies, target, request)
    worker = workers.read_worker(dependencies.db, WorkerId(worker_id))
    if worker is None:
        raise ConnectError(Code.NOT_FOUND, f"Worker {worker_id} not found")
    if not dependencies.runtime.liveness_for_worker(worker.worker_id).healthy:
        raise ConnectError(Code.UNAVAILABLE, f"Worker {worker_id} is unavailable")
    try:
        return dependencies.runtime.backend.get_process_status(
            TaskTarget(task_id="", attempt_id=0, worker_id=WorkerId(worker_id), address=worker.address),
            request,
        )
    except ProviderError as error:
        raise ConnectError(Code.UNAVAILABLE, str(error)) from error


def mint_endpoint_token(
    dependencies: AttemptDependencies,
    request: controller_pb2.Controller.MintEndpointTokenRequest,
    context: RequestContext,
) -> controller_pb2.Controller.MintEndpointTokenResponse:
    """Mint a scoped, expiring bearer token for one task endpoint."""
    del context
    jwt_manager = dependencies.auth.jwt_manager
    if jwt_manager is None:
        raise ConnectError(Code.INTERNAL, "JWT manager not configured")
    row = dependencies.endpoints.resolve_task_endpoint(request.endpoint_name)
    if row is None:
        raise ConnectError(Code.NOT_FOUND, f"No endpoint '{request.endpoint_name}'")
    authorize_owner_if_configured(dependencies.auth, row.task_id.user)

    if request.HasField("ttl"):
        ttl = max(1, min(int(duration_from_proto(request.ttl).to_seconds()), MAX_ENDPOINT_TOKEN_TTL_SECONDS))
    else:
        ttl = DEFAULT_ENDPOINT_TOKEN_TTL_SECONDS
    now = Timestamp.now()
    expires_at = Timestamp.from_ms(now.epoch_ms() + ttl * 1000)
    token = jwt_manager.create_endpoint_token(row.name, f"iris_ket_{secrets.token_urlsafe(8)}", ttl_seconds=ttl)
    return controller_pb2.Controller.MintEndpointTokenResponse(
        token=token,
        expires_at=timestamp_to_proto(expires_at),
        capability_url=dependencies.capability_urls.build(row.name, token),
    )


def exec_in_container(
    dependencies: AttemptDependencies,
    request: controller_pb2.Controller.ExecInContainerRequest,
    context: RequestContext,
) -> controller_pb2.Controller.ExecInContainerResponse:
    """Execute a command in the current Attempt for one task."""
    del context
    try:
        task_id = JobName.from_wire(request.task_id)
        task_id.require_task()
    except ValueError as error:
        raise ConnectError(Code.INVALID_ARGUMENT, str(error)) from error

    _authorize_federated_debug_target(dependencies, task_id.root_job)
    task = tasks.read_task_with_attempts(dependencies.db, task_id)
    if task is None:
        raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found")
    proxied = _proxy_if_federated(dependencies, task_id, lambda peer: peer.exec_in_container(request))
    if proxied is not None:
        return proxied

    worker_request = worker_pb2.Worker.ExecInContainerRequest(
        task_id=request.task_id,
        command=request.command,
        timeout_seconds=request.timeout_seconds,
    )
    target = _resolve_task_target(dependencies, task, task.current_attempt_id, wire_name=request.task_id)
    timeout = request.timeout_seconds or (60 if target.worker_id is None else 0)
    response = dependencies.runtime.backend.exec_in_container(target, worker_request, timeout)
    return controller_pb2.Controller.ExecInContainerResponse(
        exit_code=response.exit_code,
        stdout=response.stdout,
        stderr=response.stderr,
        error=response.error,
    )


def _task_process_status(
    dependencies: AttemptDependencies,
    target: str,
    request: job_pb2.GetProcessStatusRequest,
) -> job_pb2.GetProcessStatusResponse:
    try:
        task_id = JobName.from_wire(target)
        task_id.require_task()
    except ValueError as error:
        raise ConnectError(Code.INVALID_ARGUMENT, f"Invalid target: {target}") from error
    _authorize_federated_debug_target(dependencies, task_id.root_job)
    task = tasks.read_task_with_attempts(dependencies.db, task_id)
    if task is None:
        raise ConnectError(Code.NOT_FOUND, f"Task {target} not found")
    proxied = _proxy_if_federated(dependencies, task_id, lambda peer: peer.get_process_status(request))
    if proxied is not None:
        return proxied
    task_target = _resolve_task_target(dependencies, task, task.current_attempt_id, wire_name=target)
    try:
        return dependencies.runtime.backend.get_process_status(task_target, request)
    except ProviderError as error:
        raise ConnectError(Code.UNAVAILABLE, str(error)) from error


def _authorize_federated_debug_target(dependencies: AttemptDependencies, root_job: JobName) -> None:
    if not dependencies.auth.provider:
        return
    identity = get_verified_identity()
    if identity is None or identity.role != FEDERATION_PEER_ROLE:
        return
    with dependencies.db.read_snapshot() as snapshot:
        handoff = reads.received_handoff(snapshot, root_job)
    if handoff is not None and handoff.requester_id == identity.user_id:
        return
    raise ConnectError(Code.PERMISSION_DENIED, f"Peer {identity.user_id!r} did not federate job {root_job}")


def _proxy_if_federated(
    dependencies: AttemptDependencies,
    task_id: JobName,
    call: Callable[[FederationPeer], Response],
) -> Response | None:
    with dependencies.db.read_snapshot() as snapshot:
        handle = reads.federated_handle(snapshot, task_id.root_job)
    if handle is None:
        return None
    return dependencies.runtime.federation.proxy_to_peer(handle.peer_id, call)


def _resolve_task_target(
    dependencies: AttemptDependencies,
    task: tasks.TaskWithAttempts,
    attempt_id: int,
    *,
    wire_name: str,
) -> TaskTarget:
    attempt_uid = next(
        (attempt.attempt_uid for attempt in task.attempts if attempt.attempt_id == attempt_id),
        "",
    )
    worker_id = tasks.task_worker_id(task)
    if worker_id is None:
        if BackendCapability.DIRECT_DISPATCH not in dependencies.runtime.backend.descriptor.capabilities:
            raise ConnectError(Code.FAILED_PRECONDITION, f"Task {wire_name} not yet assigned to a worker")
        return TaskTarget(
            task_id=task.task_id.to_wire(),
            attempt_id=attempt_id,
            worker_id=None,
            address=None,
            attempt_uid=attempt_uid,
        )
    worker = workers.read_worker(dependencies.db, worker_id)
    if worker is None or not dependencies.runtime.liveness_for_worker(worker_id).healthy:
        raise ConnectError(Code.UNAVAILABLE, f"Worker {worker_id} is unavailable")
    return TaskTarget(
        task_id=task.task_id.to_wire(),
        attempt_id=attempt_id,
        worker_id=worker_id,
        address=worker.address,
        attempt_uid=attempt_uid,
    )
