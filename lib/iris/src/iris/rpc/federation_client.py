# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Connect transport and legacy-wire codecs for native federation operations."""

from collections.abc import Callable, Iterable
from typing import TypeVar

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.interceptor import InterceptorSync
from rigging.auth import TokenProvider
from rigging.cluster_manifest import load_manifest
from rigging.credentials import ClientCredentials, credentials_for
from rigging.timing import Timestamp

from iris.cluster.config import PeerConfig
from iris.cluster.federation.protocol import (
    FederationBackendObservation,
    FederationJobDelta,
    FederationPeerObservation,
    FederationResourceAvailability,
    FederationSyncBatch,
    HandoffDelivery,
    PeerCallError,
    PeerConnectFactory,
    PeerConnection,
    PeerErrorCode,
    SyncedAttempt,
    SyncedEndpoint,
    SyncedJob,
    SyncedTask,
)
from iris.cluster.types import JobName
from iris.resources.action import ActionReceipt
from iris.resources.endpoint import EndpointAccess, ExecRequest, ExecResult, ProfileRequest, ProfileResult
from iris.resources.execution import ResourceSpec
from iris.resources.identity import AttemptIdentity, JobIdentity, TaskIdentity
from iris.resources.system import ProcessInfo
from iris.rpc import controller_pb2, job_pb2, resource_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.legacy.job_codec import (
    constraint_to_proto,
    device_from_proto,
    environment_to_proto,
    resource_spec_to_proto,
    runtime_entrypoint_to_proto,
)
from iris.rpc.profile_codec import profile_configuration_to_proto
from iris.rpc.resource_client_codec import (
    action_receipt_from_proto,
    attempt_identity_to_proto,
    exec_result_from_proto,
    job_identity_to_proto,
    profile_result_from_proto,
    task_identity_to_proto,
)
from iris.rpc.resource_connect import ResourceServiceClientSync
from iris.rpc.worker_client import EXEC_IN_CONTAINER_MAX_TIMEOUT
from iris.rpc.worker_codec import process_info_from_proto
from iris.time_proto import duration_from_proto, duration_to_proto, timestamp_from_proto, timestamp_to_proto

_LAUNCH_JOB_TIMEOUT_FLOOR_MS = 180_000
_PROFILE_PROXY_TIMEOUT_MARGIN_MS = 60_000
_EXEC_PROXY_TIMEOUT_MARGIN_MS = 60_000
_DEFAULT_PROFILE_DURATION = 10
_DEFAULT_EXEC_TIMEOUT = 60
_PROCESS_STATUS_PROXY_TIMEOUT_MS = 30_000

_T = TypeVar("_T")

_ERROR_CODES = {
    Code.ALREADY_EXISTS: PeerErrorCode.ALREADY_EXISTS,
    Code.PERMISSION_DENIED: PeerErrorCode.PERMISSION_DENIED,
    Code.INVALID_ARGUMENT: PeerErrorCode.INVALID_ARGUMENT,
    Code.NOT_FOUND: PeerErrorCode.NOT_FOUND,
    Code.UNAUTHENTICATED: PeerErrorCode.UNAUTHENTICATED,
    Code.UNAVAILABLE: PeerErrorCode.UNAVAILABLE,
    Code.DEADLINE_EXCEEDED: PeerErrorCode.UNAVAILABLE,
    Code.RESOURCE_EXHAUSTED: PeerErrorCode.UNAVAILABLE,
}
_CONNECT_CODES = {
    PeerErrorCode.ALREADY_EXISTS: Code.ALREADY_EXISTS,
    PeerErrorCode.PERMISSION_DENIED: Code.PERMISSION_DENIED,
    PeerErrorCode.INVALID_ARGUMENT: Code.INVALID_ARGUMENT,
    PeerErrorCode.NOT_FOUND: Code.NOT_FOUND,
    PeerErrorCode.UNAUTHENTICATED: Code.UNAUTHENTICATED,
    PeerErrorCode.UNAVAILABLE: Code.UNAVAILABLE,
    PeerErrorCode.UNKNOWN: Code.UNKNOWN,
}


def peer_transport_call(call: Callable[[], _T]) -> _T:
    """Translate one Connect peer call into the native federation contract."""
    try:
        return call()
    except ConnectError as error:
        raise PeerCallError(_ERROR_CODES.get(error.code, PeerErrorCode.UNKNOWN), error.message) from error
    except (ConnectionError, OSError) as error:
        raise PeerCallError(PeerErrorCode.UNAVAILABLE, str(error)) from error


def peer_rpc_call(call: Callable[[], _T]) -> _T:
    """Expose one native peer operation through a Connect service boundary."""
    try:
        return call()
    except PeerCallError as error:
        raise peer_connect_error(error) from error


def peer_connect_error(error: PeerCallError) -> ConnectError:
    """Translate a native peer failure to its Connect status."""
    return ConnectError(_CONNECT_CODES[error.code], error.message)


class ConnectPeerConnection:
    """Thread-safe Connect implementation of the native peer port."""

    def __init__(self, controller_address: str, interceptors: Iterable[InterceptorSync]):
        self._client = ControllerServiceClientSync(address=controller_address, interceptors=interceptors)
        self._resources = ResourceServiceClientSync(address=controller_address, interceptors=interceptors)

    def list_backends(self) -> tuple[FederationBackendObservation, ...]:
        response = peer_transport_call(
            lambda: self._client.list_backends(controller_pb2.Controller.ListBackendsRequest())
        )
        return tuple(_backend_from_proto(backend) for backend in response.backends)

    def launch_job(self, delivery: HandoffDelivery) -> None:
        peer_transport_call(
            lambda: self._client.launch_job(
                handoff_to_legacy_request(delivery),
                timeout_ms=_LAUNCH_JOB_TIMEOUT_FLOOR_MS,
            )
        )

    def terminate_job(self, job_id: JobName) -> None:
        peer_transport_call(
            lambda: self._client.terminate_job(controller_pb2.Controller.TerminateJobRequest(job_id=job_id.to_wire()))
        )

    def federation_sync(self, requester_id: str, cursor: str) -> FederationSyncBatch:
        response = peer_transport_call(
            lambda: self._client.federation_sync(
                controller_pb2.Controller.FederationSyncRequest(requester_id=requester_id, cursor=cursor)
            )
        )
        return federation_batch_from_legacy(response)

    def cancel_job(self, identity: JobIdentity, *, idempotency_key: str) -> ActionReceipt:
        response = peer_transport_call(
            lambda: self._resources.cancel_job(
                resource_pb2.CancelJobRequest(
                    job=job_identity_to_proto(identity),
                    idempotency_key=idempotency_key,
                )
            )
        )
        return action_receipt_from_proto(response.receipt)

    def retry_task(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        idempotency_key: str,
    ) -> ActionReceipt:
        response = peer_transport_call(
            lambda: self._resources.retry_task(
                resource_pb2.RetryTaskRequest(
                    task=task_identity_to_proto(identity),
                    expected_attempt_uid=expected_attempt_uid,
                    idempotency_key=idempotency_key,
                )
            )
        )
        return action_receipt_from_proto(response.receipt)

    def terminate_attempt(self, identity: AttemptIdentity, *, idempotency_key: str) -> ActionReceipt:
        response = peer_transport_call(
            lambda: self._resources.terminate_attempt(
                resource_pb2.TerminateAttemptRequest(
                    attempt=attempt_identity_to_proto(identity),
                    idempotency_key=idempotency_key,
                )
            )
        )
        return action_receipt_from_proto(response.receipt)

    def profile_task(self, request: ProfileRequest) -> ProfileResult:
        if request.attempt is None:
            raise ValueError("federated task profiling requires an Attempt identity")
        duration_seconds = int(request.duration.to_seconds()) if request.duration is not None else 0
        wire_request = resource_pb2.ProfileAttemptRequest(
            attempt=attempt_identity_to_proto(request.attempt),
            profile=profile_configuration_to_proto(request.profile),
        )
        if request.duration is not None:
            wire_request.duration.CopyFrom(duration_to_proto(request.duration))
        timeout_ms = (duration_seconds or _DEFAULT_PROFILE_DURATION) * 1000 + _PROFILE_PROXY_TIMEOUT_MARGIN_MS
        response = peer_transport_call(lambda: self._resources.profile_attempt(wire_request, timeout_ms=timeout_ms))
        return profile_result_from_proto(response)

    def exec_in_container(self, request: ExecRequest) -> ExecResult:
        timeout_seconds = int(request.timeout.to_seconds()) if request.timeout is not None else 0
        wire_request = resource_pb2.ExecAttemptRequest(
            attempt=attempt_identity_to_proto(request.attempt),
            command=request.command,
        )
        if request.timeout is not None:
            wire_request.timeout.CopyFrom(duration_to_proto(request.timeout))
        budget_ms = (
            EXEC_IN_CONTAINER_MAX_TIMEOUT.to_ms()
            if timeout_seconds < 0
            else (timeout_seconds or _DEFAULT_EXEC_TIMEOUT) * 1000
        )
        response = peer_transport_call(
            lambda: self._resources.exec_attempt(
                wire_request,
                timeout_ms=budget_ms + _EXEC_PROXY_TIMEOUT_MARGIN_MS,
            )
        )
        return exec_result_from_proto(response)

    def get_process_status(self, target: str) -> ProcessInfo:
        response = peer_transport_call(
            lambda: self._client.get_process_status(
                job_pb2.GetProcessStatusRequest(target=target),
                timeout_ms=_PROCESS_STATUS_PROXY_TIMEOUT_MS,
            )
        )
        return process_info_from_proto(response.process_info)

    def shutdown(self) -> None:
        self._client.close()
        self._resources.close()


def peer_connection_factory(
    federation_token_provider: TokenProvider | None = None,
) -> PeerConnectFactory:
    """Build the authenticated native connection factory for controller composition."""

    return lambda peer: connect_to_peer(peer, federation_token_provider)


def connect_to_peer(
    peer: PeerConfig,
    federation_token_provider: TokenProvider | None = None,
) -> PeerConnection:
    """Open one authenticated Connect connection to a peer controller."""

    credentials = _peer_credentials(peer, federation_token_provider)
    return ConnectPeerConnection(peer.controller_address, credentials.interceptors())


def _peer_credentials(peer: PeerConfig, federation_token_provider: TokenProvider | None) -> ClientCredentials:
    base = credentials_for(peer.cluster, load_manifest(peer.cluster).auth) if peer.cluster else ClientCredentials()
    return ClientCredentials(
        token_provider=federation_token_provider or base.token_provider,
        iap_provider=base.iap_provider,
    )


def handoff_to_legacy_request(delivery: HandoffDelivery) -> controller_pb2.Controller.LaunchJobRequest:
    """Encode a canonical handoff for the old ControllerService boundary."""

    spec = delivery.spec
    request = controller_pb2.Controller.LaunchJobRequest(
        name=spec.name,
        entrypoint=runtime_entrypoint_to_proto(spec.entrypoint),
        resources=resource_spec_to_proto(spec.resources),
        environment=environment_to_proto(spec.environment),
        bundle_id=spec.bundle_id,
        bundle_blob=delivery.bundle_blob,
        ports=spec.ports,
        max_task_failures=spec.max_task_failures,
        max_retries_failure=spec.max_retries_failure,
        max_retries_preemption=spec.max_retries_preemption,
        constraints=[constraint_to_proto(constraint) for constraint in spec.constraints],
        replicas=spec.replicas,
        fail_if_exists=spec.fail_if_exists,
        preemption_policy=spec.preemption_policy,
        existing_job_policy=spec.existing_job_policy,
        priority_band=spec.priority_band,
        task_image=spec.task_image,
        submit_argv=spec.submit_argv,
        client_revision_date=spec.client_revision_date,
        container_profile=spec.container_profile,
        federation=controller_pb2.Controller.FederationHandoff(
            requester_id=delivery.requester_id,
            owner_principal=delivery.owner_principal,
            submitting_user=delivery.submitting_user,
            handoff_nonce=delivery.handoff_nonce,
        ),
    )
    if spec.scheduling_timeout is not None:
        request.scheduling_timeout.CopyFrom(duration_to_proto(spec.scheduling_timeout))
    if spec.coscheduling is not None:
        request.coscheduling.group_by = spec.coscheduling.group_by
    if spec.timeout is not None:
        request.timeout.CopyFrom(duration_to_proto(spec.timeout))
    return request


def _backend_from_proto(
    backend: controller_pb2.Controller.BackendSummary,
) -> FederationBackendObservation:
    availability = None
    if backend.HasField("availability"):
        availability = FederationResourceAvailability(
            version=backend.availability.version,
            observation_epoch_ms=backend.availability.observation_epoch_ms,
            amounts=dict(backend.availability.amounts),
            total_amounts=dict(backend.availability.total_amounts),
            held_by_band={held.band: dict(held.amounts) for held in backend.availability.held_by_band},
        )
    return FederationBackendObservation(
        backend_id=backend.backend_id,
        name=backend.name,
        kind=backend.kind,
        capabilities=tuple(backend.capabilities),
        advertised_attributes={key: tuple(values.values) for key, values in backend.advertised_attributes.items()},
        scale_groups=tuple(backend.scale_groups),
        worker_count=backend.worker_count,
        pending_task_count=backend.pending_task_count,
        running_task_count=backend.running_task_count,
        has_autoscaler=backend.has_autoscaler,
        capacity_health=dict(backend.capacity_health),
        availability=availability,
    )


def peer_observation_to_legacy(
    peer: FederationPeerObservation,
) -> controller_pb2.Controller.PeerSummary:
    """Encode one native peer observation for the old ControllerService boundary."""

    return controller_pb2.Controller.PeerSummary(
        peer_id=peer.peer_id,
        controller_address=peer.controller_address,
        reachable=peer.reachable,
        last_contact_ms=peer.last_contact_ms,
        active_federated_jobs=peer.active_federated_jobs,
        backends=[_backend_to_proto(backend) for backend in peer.backends],
    )


def _backend_to_proto(
    backend: FederationBackendObservation,
) -> controller_pb2.Controller.BackendSummary:
    wire = controller_pb2.Controller.BackendSummary(
        backend_id=backend.backend_id,
        name=backend.name,
        kind=backend.kind,
        capabilities=backend.capabilities,
        scale_groups=backend.scale_groups,
        worker_count=backend.worker_count,
        pending_task_count=backend.pending_task_count,
        running_task_count=backend.running_task_count,
        has_autoscaler=backend.has_autoscaler,
        capacity_health=dict(backend.capacity_health),
    )
    for key, values in backend.advertised_attributes.items():
        wire.advertised_attributes[key].values.extend(values)
    if backend.availability is not None:
        availability = backend.availability
        wire.availability.version = availability.version
        wire.availability.observation_epoch_ms = availability.observation_epoch_ms
        wire.availability.amounts.update(availability.amounts)
        wire.availability.total_amounts.update(availability.total_amounts)
        for band, amounts in sorted(availability.held_by_band.items()):
            wire.availability.held_by_band.add(band=band, amounts=amounts)
    return wire


def federation_batch_from_legacy(
    response: controller_pb2.Controller.FederationSyncResponse,
) -> FederationSyncBatch:
    """Decode an old ControllerService sync response into native records."""

    def maybe_timestamp(message, field: str) -> Timestamp | None:
        return timestamp_from_proto(getattr(message, field)) if message.HasField(field) else None

    deltas = []
    for delta in response.deltas:
        summary = None
        if delta.HasField("summary"):
            wire = delta.summary
            summary = SyncedJob(
                job_id=JobName.from_wire(delta.job_id),
                state=wire.state,
                error_message=wire.error,
                exit_code=wire.exit_code or None,
                submitted_at=maybe_timestamp(wire, "submitted_at"),
                started_at=maybe_timestamp(wire, "started_at"),
                finished_at=maybe_timestamp(wire, "finished_at"),
                task_count=wire.task_count,
                backend_id=wire.backend_id,
                resources=ResourceSpec(
                    cpu=wire.resources.cpu_millicores / 1_000,
                    memory=wire.resources.memory_bytes,
                    disk=wire.resources.disk_bytes,
                    device=device_from_proto(wire.resources.device) if wire.resources.HasField("device") else None,
                ),
            )
        tasks = []
        for wire in delta.changed_tasks:
            attempts = tuple(
                SyncedAttempt(
                    attempt_id=attempt.attempt_id,
                    state=attempt.state,
                    exit_code=attempt.exit_code or None,
                    error_message=attempt.error,
                    attempt_uid=attempt.attempt_uid,
                    started_at=maybe_timestamp(attempt, "started_at"),
                    finished_at=maybe_timestamp(attempt, "finished_at"),
                )
                for attempt in wire.attempts
            )
            tasks.append(
                SyncedTask(
                    task_id=JobName.from_wire(wire.task_id),
                    state=wire.state,
                    error_message=wire.error,
                    exit_code=wire.exit_code or None,
                    submitted_at=maybe_timestamp(wire, "submitted_at"),
                    started_at=maybe_timestamp(wire, "started_at"),
                    finished_at=maybe_timestamp(wire, "finished_at"),
                    current_attempt_id=wire.current_attempt_id,
                    worker_address=wire.worker_address,
                    worker_label=wire.worker_id or wire.worker_address,
                    status_message=wire.status_message,
                    backend_id=wire.backend_id,
                    attempts=attempts,
                )
            )
        deltas.append(
            FederationJobDelta(
                job_id=JobName.from_wire(delta.job_id),
                summary=summary,
                changed_tasks=tuple(tasks),
                tombstone=delta.tombstone,
            )
        )
    endpoints = tuple(
        SyncedEndpoint(
            endpoint_id=wire.endpoint_id,
            name=wire.name,
            address=wire.address,
            task_id=JobName.from_wire(wire.task_id),
            access=(
                EndpointAccess.LINK
                if wire.access == controller_pb2.Controller.ENDPOINT_ACCESS_LINK
                else EndpointAccess.PRIVATE
            ),
            metadata=dict(wire.metadata),
            lease_remaining=(duration_from_proto(wire.lease_remaining) if wire.HasField("lease_remaining") else None),
        )
        for wire in response.endpoints
    )
    return FederationSyncBatch(tuple(deltas), response.next_cursor, response.cursor_stale, endpoints)


def federation_batch_to_legacy(
    batch: FederationSyncBatch,
) -> controller_pb2.Controller.FederationSyncResponse:
    """Encode native sync records for the old ControllerService boundary."""

    deltas = []
    for delta in batch.deltas:
        wire_delta = controller_pb2.Controller.FederationJobDelta(
            job_id=delta.job_id.to_wire(),
            tombstone=delta.tombstone,
        )
        if delta.summary is not None:
            summary = delta.summary
            wire_summary = job_pb2.JobStatus(
                job_id=summary.job_id.to_wire(),
                state=summary.state,
                error=summary.error_message,
                exit_code=summary.exit_code or 0,
                backend_id=summary.backend_id,
                task_count=summary.task_count,
                resources=resource_spec_to_proto(summary.resources),
            )
            _set_timestamps(wire_summary, summary.submitted_at, summary.started_at, summary.finished_at)
            wire_delta.summary.CopyFrom(wire_summary)
        for task in delta.changed_tasks:
            wire_task = job_pb2.TaskStatus(
                task_id=task.task_id.to_wire(),
                state=task.state,
                error=task.error_message,
                exit_code=task.exit_code or 0,
                current_attempt_id=task.current_attempt_id,
                worker_address=task.worker_address,
                worker_id=task.worker_label,
                status_message=task.status_message,
                backend_id=task.backend_id,
            )
            _set_timestamps(wire_task, task.submitted_at, task.started_at, task.finished_at)
            for attempt in task.attempts:
                wire_attempt = job_pb2.TaskAttempt(
                    attempt_id=attempt.attempt_id,
                    state=attempt.state,
                    exit_code=attempt.exit_code or 0,
                    error=attempt.error_message,
                    attempt_uid=attempt.attempt_uid,
                )
                if attempt.started_at is not None:
                    wire_attempt.started_at.CopyFrom(timestamp_to_proto(attempt.started_at))
                if attempt.finished_at is not None:
                    wire_attempt.finished_at.CopyFrom(timestamp_to_proto(attempt.finished_at))
                wire_task.attempts.append(wire_attempt)
            wire_delta.changed_tasks.append(wire_task)
        deltas.append(wire_delta)

    endpoints = []
    for endpoint in batch.endpoints:
        wire_endpoint = controller_pb2.Controller.FederationEndpoint(
            endpoint_id=endpoint.endpoint_id,
            name=endpoint.name,
            address=endpoint.address,
            task_id=endpoint.task_id.to_wire(),
            access=(
                controller_pb2.Controller.ENDPOINT_ACCESS_LINK
                if endpoint.access == EndpointAccess.LINK
                else controller_pb2.Controller.ENDPOINT_ACCESS_PRIVATE
            ),
            metadata=dict(endpoint.metadata),
        )
        if endpoint.lease_remaining is not None:
            wire_endpoint.lease_remaining.CopyFrom(duration_to_proto(endpoint.lease_remaining))
        endpoints.append(wire_endpoint)
    return controller_pb2.Controller.FederationSyncResponse(
        deltas=deltas,
        next_cursor=batch.next_cursor,
        cursor_stale=batch.cursor_stale,
        endpoints=endpoints,
    )


def _set_timestamps(
    message,
    submitted_at: Timestamp | None,
    started_at: Timestamp | None,
    finished_at: Timestamp | None,
) -> None:
    if submitted_at is not None:
        message.submitted_at.CopyFrom(timestamp_to_proto(submitted_at))
    if started_at is not None:
        message.started_at.CopyFrom(timestamp_to_proto(started_at))
    if finished_at is not None:
        message.finished_at.CopyFrom(timestamp_to_proto(finished_at))
