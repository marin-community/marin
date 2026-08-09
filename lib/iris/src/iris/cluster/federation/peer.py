# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authenticated peer operations and last-observed federation availability."""

import logging
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from connectrpc.errors import ConnectError
from connectrpc.interceptor import InterceptorSync
from rigging.auth import TokenProvider
from rigging.cluster_manifest import load_manifest
from rigging.credentials import ClientCredentials, credentials_for
from rigging.timing import Timestamp

from iris.cluster.backends.rpc.backend import EXEC_IN_CONTAINER_MAX_TIMEOUT
from iris.cluster.client.resource_conversion import (
    action_receipt_from_proto,
    attempt_identity_to_proto,
    exec_result_from_proto,
    job_identity_to_proto,
    profile_result_from_proto,
    task_identity_to_proto,
)
from iris.cluster.config import PeerConfig
from iris.cluster.federation.legacy_rpc import federation_batch_from_legacy
from iris.cluster.federation.store import FederationSyncBatch
from iris.cluster.resources.action import ActionReceipt
from iris.cluster.resources.endpoint import ExecRequest, ExecResult, ProfileRequest, ProfileResult
from iris.cluster.resources.identity import AttemptIdentity, JobIdentity, TaskIdentity
from iris.cluster.resources.job import JobSpec
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2, resource_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.profile_codec import profile_configuration_to_proto
from iris.rpc.resource_connect import ResourceServiceClientSync
from iris.time_proto import duration_to_proto

# A handoff carries a full request and a peer's cold boot can outrun the default
# RPC deadline, so deliver LaunchJob with this floor to avoid spurious failures.
_LAUNCH_JOB_TIMEOUT_FLOOR_MS = 180_000

# A proxied exec/profile does the parent's own local-dispatch work on the peer:
# the peer resolves task->worker and runs the operation for its full duration.
# Give the parent->peer hop the peer's own budget plus margin for the extra
# controller hop, so the parent waits out the peer rather than timing out first.
_PROFILE_PROXY_TIMEOUT_MARGIN_MS = 60_000
_EXEC_PROXY_TIMEOUT_MARGIN_MS = 60_000
_DEFAULT_PROFILE_DURATION = 10
_DEFAULT_EXEC_TIMEOUT = 60
# Process status has no duration; the peer's own worker/pod hop is bounded (a worker
# forward caps at ~10s, a kubectl-exec /proc read is quick), so give the parent->peer
# hop that budget plus margin to outlast the peer rather than time out first.
_PROCESS_STATUS_PROXY_TIMEOUT_MS = 30_000

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HandoffDelivery:
    spec: JobSpec
    bundle_blob: bytes
    requester_id: str
    owner_principal: str
    submitting_user: str
    handoff_nonce: str


class PeerConnection(Protocol):
    """Remote operations available to the federation coordinator."""

    def list_backends(self) -> list[controller_pb2.Controller.BackendSummary]: ...

    def launch_job(self, delivery: HandoffDelivery) -> None: ...

    def terminate_job(self, job_id: JobName) -> None: ...

    def federation_sync(self, requester_id: str, cursor: str) -> FederationSyncBatch: ...

    def cancel_job(self, identity: JobIdentity, *, idempotency_key: str) -> ActionReceipt: ...

    def retry_task(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        idempotency_key: str,
    ) -> ActionReceipt: ...

    def terminate_attempt(self, identity: AttemptIdentity, *, idempotency_key: str) -> ActionReceipt: ...

    def profile_task(self, request: ProfileRequest) -> ProfileResult: ...

    def exec_in_container(self, request: ExecRequest) -> ExecResult: ...

    def get_process_status(self, request: job_pb2.GetProcessStatusRequest) -> job_pb2.GetProcessStatusResponse: ...

    def shutdown(self) -> None: ...


PeerConnectFactory = Callable[[PeerConfig], PeerConnection]


@dataclass(frozen=True)
class PeerHeartbeat:
    """The latest capability-heartbeat observation for one peer."""

    reachable: bool = False
    backends: tuple[controller_pb2.Controller.BackendSummary, ...] = ()
    last_contact_ms: int = 0


def _peer_credentials(peer: PeerConfig, federation_token_provider: TokenProvider | None) -> ClientCredentials:
    """The client credentials this controller presents to ``peer``.

    The federation bearer — this cluster's short-lived ``aud="federation"`` token,
    minted per request by ``federation_token_provider`` — rides on ``Authorization``
    so an enforcing peer admits the handoff as a trusted requester. The peer's
    manifest still supplies the IAP edge token on ``Proxy-Authorization``. With no
    provider and an empty ``cluster``, no credentials are sent — loopback/no-auth,
    for a local or same-VPC peer that trusts the connection.
    """
    base = credentials_for(peer.cluster, load_manifest(peer.cluster).auth) if peer.cluster else ClientCredentials()
    return ClientCredentials(
        token_provider=federation_token_provider or base.token_provider,
        iap_provider=base.iap_provider,
    )


def legacy_handoff_request(delivery: HandoffDelivery) -> controller_pb2.Controller.LaunchJobRequest:
    """Encode a canonical handoff for a legacy controller peer."""
    spec = delivery.spec
    request = controller_pb2.Controller.LaunchJobRequest(
        name=spec.name,
        entrypoint=spec.entrypoint,
        resources=spec.resources.to_exact_proto(),
        environment=spec.environment,
        bundle_id=spec.bundle_id,
        bundle_blob=delivery.bundle_blob,
        ports=spec.ports,
        max_task_failures=spec.max_task_failures,
        max_retries_failure=spec.max_retries_failure,
        max_retries_preemption=spec.max_retries_preemption,
        constraints=[constraint.to_proto() for constraint in spec.constraints],
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
        request.coscheduling.CopyFrom(spec.coscheduling.to_proto())
    if spec.timeout is not None:
        request.timeout.CopyFrom(duration_to_proto(spec.timeout))
    return request


class _PeerRpcConnection:
    """A :class:`PeerConnection` over the generated controller stub.

    One instance is shared across the heartbeat thread, the sync thread, and the
    RPC-handler threads that deliver a handoff or a routed cancel. That is safe: the
    stub's transport is a connection-pooled ``reqwest`` client built for concurrent
    use, and each call builds its request independently — there is no per-call
    mutable client state to guard.
    """

    def __init__(self, controller_address: str, interceptors: Iterable[InterceptorSync]):
        self._client = ControllerServiceClientSync(address=controller_address, interceptors=interceptors)
        self._resources = ResourceServiceClientSync(address=controller_address, interceptors=interceptors)

    def list_backends(self) -> list[controller_pb2.Controller.BackendSummary]:
        response = self._client.list_backends(controller_pb2.Controller.ListBackendsRequest())
        return list(response.backends)

    def launch_job(self, delivery: HandoffDelivery) -> None:
        self._client.launch_job(legacy_handoff_request(delivery), timeout_ms=_LAUNCH_JOB_TIMEOUT_FLOOR_MS)

    def terminate_job(self, job_id: JobName) -> None:
        self._client.terminate_job(controller_pb2.Controller.TerminateJobRequest(job_id=job_id.to_wire()))

    def federation_sync(self, requester_id: str, cursor: str) -> FederationSyncBatch:
        response = self._client.federation_sync(
            controller_pb2.Controller.FederationSyncRequest(requester_id=requester_id, cursor=cursor)
        )
        return federation_batch_from_legacy(response)

    def cancel_job(self, identity: JobIdentity, *, idempotency_key: str) -> ActionReceipt:
        response = self._resources.cancel_job(
            resource_pb2.CancelJobRequest(
                job=job_identity_to_proto(identity),
                idempotency_key=idempotency_key,
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
        response = self._resources.retry_task(
            resource_pb2.RetryTaskRequest(
                task=task_identity_to_proto(identity),
                expected_attempt_uid=expected_attempt_uid,
                idempotency_key=idempotency_key,
            )
        )
        return action_receipt_from_proto(response.receipt)

    def terminate_attempt(self, identity: AttemptIdentity, *, idempotency_key: str) -> ActionReceipt:
        response = self._resources.terminate_attempt(
            resource_pb2.TerminateAttemptRequest(
                attempt=attempt_identity_to_proto(identity),
                idempotency_key=idempotency_key,
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
        return profile_result_from_proto(self._resources.profile_attempt(wire_request, timeout_ms=timeout_ms))

    def exec_in_container(self, request: ExecRequest) -> ExecResult:
        timeout_seconds = int(request.timeout.to_seconds()) if request.timeout is not None else 0
        wire_request = resource_pb2.ExecAttemptRequest(
            attempt=attempt_identity_to_proto(request.attempt),
            command=request.command,
        )
        if request.timeout is not None:
            wire_request.timeout.CopyFrom(duration_to_proto(request.timeout))
        # Mirror the worker backend's exec timeout contract (backend.py): a
        # negative timeout is "no caller limit", which the peer caps at
        # EXEC_IN_CONTAINER_MAX_TIMEOUT, so the parent->peer hop must outlast that
        # cap rather than collapse to the margin.
        if timeout_seconds < 0:
            budget_ms = EXEC_IN_CONTAINER_MAX_TIMEOUT.to_ms()
        else:
            budget_ms = (timeout_seconds or _DEFAULT_EXEC_TIMEOUT) * 1000
        return exec_result_from_proto(
            self._resources.exec_attempt(wire_request, timeout_ms=budget_ms + _EXEC_PROXY_TIMEOUT_MARGIN_MS)
        )

    def get_process_status(self, request: job_pb2.GetProcessStatusRequest) -> job_pb2.GetProcessStatusResponse:
        return self._client.get_process_status(request, timeout_ms=_PROCESS_STATUS_PROXY_TIMEOUT_MS)

    def shutdown(self) -> None:
        self._client.close()
        self._resources.close()


def connect_to_peer(peer: PeerConfig, federation_token_provider: TokenProvider | None = None) -> PeerConnection:
    """Open one authenticated connection to a peer controller.

    The connection presents this cluster's federation token (via
    ``federation_token_provider``) on every RPC to the peer.
    """
    return _PeerRpcConnection(peer.controller_address, _peer_credentials(peer, federation_token_provider).interceptors())


class FederationPeer:
    """Coordinate remote operations and thread-safe capability observations for one peer."""

    def __init__(self, peer_id: str, config: PeerConfig, connection: PeerConnection):
        self.peer_id = peer_id
        self.controller_address = config.controller_address
        self._connection = connection
        self._lock = threading.Lock()
        self._heartbeat = PeerHeartbeat()

    def probe(self) -> None:
        """Refresh the peer's advertised backends via one heartbeat RPC.

        On success, records the peer's backends, marks it reachable, and stamps the
        contact time. On failure, marks it unreachable and keeps the last-known
        backends — staleness is signalled by ``reachable``.
        """
        try:
            backends = self._connection.list_backends()
        except (ConnectError, ConnectionError, OSError) as exc:
            logger.warning("Federation heartbeat to peer %s failed: %s", self.peer_id, exc)
            with self._lock:
                self._heartbeat = replace(self._heartbeat, reachable=False)
            return
        with self._lock:
            self._heartbeat = PeerHeartbeat(
                reachable=True,
                backends=tuple(backends),
                last_contact_ms=Timestamp.now().epoch_ms(),
            )

    def heartbeat(self) -> PeerHeartbeat:
        """The peer's latest heartbeat observation."""
        with self._lock:
            return self._heartbeat

    def launch_job(self, delivery: HandoffDelivery) -> None:
        """Submit a handed-off Job to its remote execution authority."""
        self._connection.launch_job(delivery)

    def terminate_job(self, job_id: JobName) -> None:
        """Cancel a handed-off Job on its execution peer.

        The ``job_id`` is the cluster-invariant local id: the peer runs and
        reports the same id the parent submitted, so there is nothing to rebase.
        """
        self._connection.terminate_job(job_id)

    def federation_sync(self, requester_id: str, cursor: str) -> FederationSyncBatch:
        """Run one delta-sync round against the peer."""
        return self._connection.federation_sync(requester_id, cursor)

    def cancel_job(self, identity: JobIdentity, *, idempotency_key: str) -> ActionReceipt:
        """Cancel an exact Job incarnation on the execution peer."""
        return self._connection.cancel_job(identity, idempotency_key=idempotency_key)

    def retry_task(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        idempotency_key: str,
    ) -> ActionReceipt:
        """Retry an exact Task incarnation on the execution peer."""
        return self._connection.retry_task(
            identity,
            expected_attempt_uid=expected_attempt_uid,
            idempotency_key=idempotency_key,
        )

    def terminate_attempt(self, identity: AttemptIdentity, *, idempotency_key: str) -> ActionReceipt:
        """Terminate an exact Attempt incarnation on the execution peer."""
        return self._connection.terminate_attempt(identity, idempotency_key=idempotency_key)

    def profile_task(self, request: ProfileRequest) -> ProfileResult:
        """Profile an Attempt running on the peer."""
        return self._connection.profile_task(request)

    def exec_in_container(self, request: ExecRequest) -> ExecResult:
        """Execute a command in an Attempt running on the peer."""
        return self._connection.exec_in_container(request)

    def get_process_status(self, request: job_pb2.GetProcessStatusRequest) -> job_pb2.GetProcessStatusResponse:
        """Read process status for an Attempt running on the peer."""
        return self._connection.get_process_status(request)

    def close(self) -> None:
        """Release the peer connection."""
        self._connection.shutdown()


def build_peers(
    peers: Mapping[str, PeerConfig],
    *,
    federation_token_provider: TokenProvider | None = None,
    connect: PeerConnectFactory | None = None,
) -> list[FederationPeer]:
    """Build one :class:`FederationPeer` per configured peer, ordered by peer id.

    Each connection presents this cluster's federation token (via
    ``federation_token_provider``) so an enforcing peer admits the handoff as a
    trusted requester. ``connect`` overrides the connection factory (tests inject a
    stub); the default opens a real authenticated connection to the peer's stub.
    """
    factory = connect or (lambda config: connect_to_peer(config, federation_token_provider))
    return [FederationPeer(peer_id, config, factory(config)) for peer_id, config in sorted(peers.items())]
