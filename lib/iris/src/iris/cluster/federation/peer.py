# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One connection per federation peer, plus its capability-heartbeat state.

:class:`FederationPeer` holds one connection per peer (keyed by peer id) and caches
the backends that peer last advertised. The connection speaks the generated controller
stub directly (not the end-user ``RemoteClusterClient``): federation drives a peer only
with the raw RPCs — ``LaunchJob`` (handoff), ``FederationSync``, and ``ListBackends``
(heartbeat). It presents the credentials
resolved from the peer's cluster manifest via ``credentials_for``.
"""

import logging
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.interceptor import InterceptorSync
from rigging.auth import TokenProvider
from rigging.cluster_manifest import load_manifest
from rigging.credentials import ClientCredentials, credentials_for
from rigging.timing import Timestamp

from iris.cluster.backends.rpc.backend import EXEC_IN_CONTAINER_MAX_TIMEOUT
from iris.cluster.config import PeerConfig
from iris.rpc import controller_pb2, job_pb2, resource_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.resource_connect import ResourceServiceClientSync

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


class PeerConnection(Protocol):
    """The peer-controller surface federation drives.

    ``list_backends`` is the capability heartbeat; ``launch_job`` delivers a handoff;
    ``federation_sync`` runs one delta-sync round;
    ``profile_task``/``exec_in_container``/``get_process_status`` proxy an on-demand RPC
    against a handed-off task, which the peer resolves to its own worker or pod.
    """

    def list_backends(self) -> list[controller_pb2.Controller.BackendSummary]: ...

    def launch_job(
        self, request: controller_pb2.Controller.LaunchJobRequest
    ) -> controller_pb2.Controller.LaunchJobResponse: ...

    def federation_sync(
        self, request: controller_pb2.Controller.FederationSyncRequest
    ) -> controller_pb2.Controller.FederationSyncResponse: ...

    def profile_task(self, request: job_pb2.ProfileTaskRequest) -> job_pb2.ProfileTaskResponse: ...

    def exec_in_container(
        self, request: controller_pb2.Controller.ExecInContainerRequest
    ) -> controller_pb2.Controller.ExecInContainerResponse: ...

    def get_process_status(self, request: job_pb2.GetProcessStatusRequest) -> job_pb2.GetProcessStatusResponse: ...

    def shutdown(self) -> None: ...


class ResourcePeerConnection(Protocol):
    """Generic resource methods used only after capability negotiation."""

    def get_service_info(self) -> resource_pb2.GetServiceInfoResponse: ...

    def update_resource(self, request: resource_pb2.UpdateResourceRequest) -> resource_pb2.Operation: ...

    def get_resource(self, request: resource_pb2.GetResourceRequest) -> resource_pb2.GetResourceResponse: ...


class _UnsupportedResourcePeerConnection:
    def get_service_info(self) -> resource_pb2.GetServiceInfoResponse:
        raise ConnectError(Code.UNIMPLEMENTED, "peer does not expose ResourceService")

    def update_resource(self, request: resource_pb2.UpdateResourceRequest) -> resource_pb2.Operation:
        raise ConnectError(Code.UNIMPLEMENTED, "peer does not expose ResourceService")

    def get_resource(self, request: resource_pb2.GetResourceRequest) -> resource_pb2.GetResourceResponse:
        raise ConnectError(Code.UNIMPLEMENTED, "peer does not expose ResourceService")


_UNSUPPORTED_RESOURCE_PEER = _UnsupportedResourcePeerConnection()


PeerConnectFactory = Callable[[PeerConfig], PeerConnection]


@dataclass(frozen=True)
class PeerHeartbeat:
    """The latest capability-heartbeat observation for one peer."""

    reachable: bool = False
    backends: tuple[controller_pb2.Controller.BackendSummary, ...] = ()
    resource_info: resource_pb2.GetServiceInfoResponse | None = None
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


class _PeerRpcConnection:
    """A :class:`PeerConnection` over the generated controller stub.

    One instance is shared across the heartbeat thread, the sync thread, and the
    RPC-handler threads that deliver a handoff or resource mutation. That is safe: the
    stub's transport is a connection-pooled ``reqwest`` client built for concurrent
    use, and each call builds its request independently — there is no per-call
    mutable client state to guard.
    """

    def __init__(self, controller_address: str, interceptors: Iterable[InterceptorSync]):
        self._client = ControllerServiceClientSync(address=controller_address, interceptors=interceptors)
        self._resource_client = ResourceServiceClientSync(address=controller_address, interceptors=interceptors)

    def list_backends(self) -> list[controller_pb2.Controller.BackendSummary]:
        response = self._client.list_backends(controller_pb2.Controller.ListBackendsRequest())
        return list(response.backends)

    def launch_job(
        self, request: controller_pb2.Controller.LaunchJobRequest
    ) -> controller_pb2.Controller.LaunchJobResponse:
        return self._client.launch_job(request, timeout_ms=_LAUNCH_JOB_TIMEOUT_FLOOR_MS)

    def federation_sync(
        self, request: controller_pb2.Controller.FederationSyncRequest
    ) -> controller_pb2.Controller.FederationSyncResponse:
        return self._client.federation_sync(request)

    def profile_task(self, request: job_pb2.ProfileTaskRequest) -> job_pb2.ProfileTaskResponse:
        timeout_ms = (request.duration_seconds or _DEFAULT_PROFILE_DURATION) * 1000 + _PROFILE_PROXY_TIMEOUT_MARGIN_MS
        return self._client.profile_task(request, timeout_ms=timeout_ms)

    def exec_in_container(
        self, request: controller_pb2.Controller.ExecInContainerRequest
    ) -> controller_pb2.Controller.ExecInContainerResponse:
        # Mirror the worker backend's exec timeout contract (backend.py): a
        # negative timeout is "no caller limit", which the peer caps at
        # EXEC_IN_CONTAINER_MAX_TIMEOUT, so the parent->peer hop must outlast that
        # cap rather than collapse to the margin.
        if request.timeout_seconds < 0:
            budget_ms = EXEC_IN_CONTAINER_MAX_TIMEOUT.to_ms()
        else:
            budget_ms = (request.timeout_seconds or _DEFAULT_EXEC_TIMEOUT) * 1000
        return self._client.exec_in_container(request, timeout_ms=budget_ms + _EXEC_PROXY_TIMEOUT_MARGIN_MS)

    def get_process_status(self, request: job_pb2.GetProcessStatusRequest) -> job_pb2.GetProcessStatusResponse:
        return self._client.get_process_status(request, timeout_ms=_PROCESS_STATUS_PROXY_TIMEOUT_MS)

    def get_service_info(self) -> resource_pb2.GetServiceInfoResponse:
        return self._resource_client.get_service_info(resource_pb2.GetServiceInfoRequest())

    def update_resource(self, request: resource_pb2.UpdateResourceRequest) -> resource_pb2.Operation:
        return self._resource_client.update_resource(request)

    def get_resource(self, request: resource_pb2.GetResourceRequest) -> resource_pb2.GetResourceResponse:
        return self._resource_client.get_resource(request)

    def shutdown(self) -> None:
        self._client.close()
        self._resource_client.close()


def connect_to_peer(peer: PeerConfig, federation_token_provider: TokenProvider | None = None) -> PeerConnection:
    """Open one authenticated connection to a peer controller.

    The connection presents this cluster's federation token (via
    ``federation_token_provider``) on every RPC to the peer.
    """
    return _PeerRpcConnection(peer.controller_address, _peer_credentials(peer, federation_token_provider).interceptors())


class FederationPeer:
    """One federation peer: a connection plus its latest heartbeat state.

    Thread-safe: the heartbeat loop writes via :meth:`probe`; RPC handlers read
    via :meth:`heartbeat`.
    """

    def __init__(
        self,
        peer_id: str,
        config: PeerConfig,
        connection: PeerConnection,
        resource_connection: ResourcePeerConnection = _UNSUPPORTED_RESOURCE_PEER,
    ):
        self.peer_id = peer_id
        self.controller_address = config.controller_address
        self._connection = connection
        self._resource_connection = resource_connection
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
        try:
            resource_info = self._resource_connection.get_service_info()
        except (ConnectError, ConnectionError, OSError):
            resource_info = None
        with self._lock:
            self._heartbeat = PeerHeartbeat(
                reachable=True,
                backends=tuple(backends),
                resource_info=resource_info,
                last_contact_ms=Timestamp.now().epoch_ms(),
            )

    def heartbeat(self) -> PeerHeartbeat:
        """The peer's latest heartbeat observation."""
        with self._lock:
            return self._heartbeat

    def supports_update(self, resource_type: str, update_type_url: str) -> bool:
        with self._lock:
            info = self._heartbeat.resource_info
        return info is not None and any(
            capability.type == resource_type and update_type_url in capability.update_type_urls
            for capability in info.resources
        )

    def launch_job(
        self, request: controller_pb2.Controller.LaunchJobRequest
    ) -> controller_pb2.Controller.LaunchJobResponse:
        """Deliver a handed-off job to the peer (reuses its ``LaunchJob``)."""
        return self._connection.launch_job(request)

    def federation_sync(
        self, request: controller_pb2.Controller.FederationSyncRequest
    ) -> controller_pb2.Controller.FederationSyncResponse:
        """Run one delta-sync round against the peer."""
        return self._connection.federation_sync(request)

    def profile_task(self, request: job_pb2.ProfileTaskRequest) -> job_pb2.ProfileTaskResponse:
        """Proxy a profile RPC for a handed-off task to the peer (reuses its ``ProfileTask``)."""
        return self._connection.profile_task(request)

    def exec_in_container(
        self, request: controller_pb2.Controller.ExecInContainerRequest
    ) -> controller_pb2.Controller.ExecInContainerResponse:
        """Proxy an exec RPC for a handed-off task to the peer (reuses its ``ExecInContainer``)."""
        return self._connection.exec_in_container(request)

    def get_process_status(self, request: job_pb2.GetProcessStatusRequest) -> job_pb2.GetProcessStatusResponse:
        """Proxy a process-status RPC for a handed-off task to the peer (reuses its ``GetProcessStatus``)."""
        return self._connection.get_process_status(request)

    def update_resource(self, request: resource_pb2.UpdateResourceRequest) -> resource_pb2.Operation:
        return self._resource_connection.update_resource(request)

    def get_resource(self, request: resource_pb2.GetResourceRequest) -> resource_pb2.GetResourceResponse:
        return self._resource_connection.get_resource(request)

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
    result = []
    for peer_id, config in sorted(peers.items()):
        connection = factory(config)
        resource_connection = connection if connect is None else _UNSUPPORTED_RESOURCE_PEER
        result.append(FederationPeer(peer_id, config, connection, resource_connection))
    return result
