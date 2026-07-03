# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One connection per federation peer, plus its capability-heartbeat state.

:class:`FederationPeer` holds one connection per peer (keyed by peer id) and caches
the backends that peer last advertised — its static topology and current state. The
connection speaks the generated controller stub directly (not the end-user
``RemoteClusterClient``): federation only ever drives a peer with the raw RPCs —
``LaunchJob`` (handoff), ``TerminateJob`` (routed cancel), ``FederationSync``, and
``ListBackends`` (heartbeat). It is authenticated with the credentials this
controller presents to the peer, resolved from the peer's cluster manifest via the
shared ``credentials_for`` path — no second credential system.
"""

import logging
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from connectrpc.errors import ConnectError
from connectrpc.interceptor import InterceptorSync
from rigging.cluster_manifest import load_manifest
from rigging.credentials import ClientCredentials, credentials_for
from rigging.timing import Timestamp

from iris.cluster.backends.rpc.backend import EXEC_IN_CONTAINER_MAX_TIMEOUT
from iris.cluster.config import PeerConfig
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.controller_connect import ControllerServiceClientSync

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

logger = logging.getLogger(__name__)


class PeerConnection(Protocol):
    """The peer-controller surface federation drives: capability heartbeat plus
    the handoff, delta-sync, routed-cancel, and proxied on-demand RPCs.

    Handoff reuses the ordinary ``LaunchJob`` (the request carries the remote job
    name and federation attribution); a routed cancel reuses ``TerminateJob``
    (targeting the remote job id); ``federation_sync`` is federation's one
    purpose-built endpoint; ``profile_task``/``exec_in_container`` proxy an
    on-demand RPC against a handed-off task through the peer controller, which
    does its own task->worker resolution.
    """

    def list_backends(self) -> list[controller_pb2.Controller.BackendSummary]: ...

    def launch_job(
        self, request: controller_pb2.Controller.LaunchJobRequest
    ) -> controller_pb2.Controller.LaunchJobResponse: ...

    def terminate_job(self, job_id: JobName) -> None: ...

    def federation_sync(
        self, request: controller_pb2.Controller.FederationSyncRequest
    ) -> controller_pb2.Controller.FederationSyncResponse: ...

    def profile_task(self, request: job_pb2.ProfileTaskRequest) -> job_pb2.ProfileTaskResponse: ...

    def exec_in_container(
        self, request: controller_pb2.Controller.ExecInContainerRequest
    ) -> controller_pb2.Controller.ExecInContainerResponse: ...

    def shutdown(self) -> None: ...


PeerConnectFactory = Callable[[PeerConfig], PeerConnection]


@dataclass(frozen=True)
class PeerHeartbeat:
    """The latest capability-heartbeat observation for one peer."""

    reachable: bool = False
    backends: tuple[controller_pb2.Controller.BackendSummary, ...] = ()
    last_contact_ms: int = 0


def _peer_credentials(peer: PeerConfig) -> ClientCredentials:
    """The client credentials this controller presents to ``peer``.

    Resolved from the peer's cluster manifest via the shared ``credentials_for``
    path. An empty ``cluster`` yields no credentials — loopback/no-auth, for a
    local or same-VPC peer that trusts the connection.
    """
    if not peer.cluster:
        return ClientCredentials()
    manifest = load_manifest(peer.cluster)
    return credentials_for(peer.cluster, manifest.auth, static_token=peer.static_token or None)


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

    def list_backends(self) -> list[controller_pb2.Controller.BackendSummary]:
        response = self._client.list_backends(controller_pb2.Controller.ListBackendsRequest())
        return list(response.backends)

    def launch_job(
        self, request: controller_pb2.Controller.LaunchJobRequest
    ) -> controller_pb2.Controller.LaunchJobResponse:
        return self._client.launch_job(request, timeout_ms=_LAUNCH_JOB_TIMEOUT_FLOOR_MS)

    def terminate_job(self, job_id: JobName) -> None:
        self._client.terminate_job(controller_pb2.Controller.TerminateJobRequest(job_id=job_id.to_wire()))

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

    def shutdown(self) -> None:
        self._client.close()


def connect_to_peer(peer: PeerConfig) -> PeerConnection:
    """Open one authenticated connection to a peer controller."""
    return _PeerRpcConnection(peer.controller_address, _peer_credentials(peer).interceptors())


class FederationPeer:
    """One federation peer: a connection plus its latest heartbeat state.

    Thread-safe: the heartbeat loop writes via :meth:`probe`; RPC handlers read
    via :meth:`heartbeat`.
    """

    def __init__(self, peer_id: str, config: PeerConfig, connection: PeerConnection):
        self.peer_id = peer_id
        self.controller_address = config.controller_address
        self.dashboard_url = config.dashboard_url
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

    def launch_job(
        self, request: controller_pb2.Controller.LaunchJobRequest
    ) -> controller_pb2.Controller.LaunchJobResponse:
        """Deliver a handed-off job to the peer (reuses its ``LaunchJob``)."""
        return self._connection.launch_job(request)

    def terminate_job(self, remote_job_id: JobName) -> None:
        """Route a cancel to the peer (reuses its ``TerminateJob``)."""
        self._connection.terminate_job(remote_job_id)

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

    def close(self) -> None:
        """Release the peer connection."""
        self._connection.shutdown()


def build_peers(
    peers: Mapping[str, PeerConfig],
    *,
    connect: PeerConnectFactory = connect_to_peer,
) -> list[FederationPeer]:
    """Build one :class:`FederationPeer` per configured peer, ordered by peer id.

    ``connect`` builds each peer connection; the default opens a real
    authenticated connection to the peer's controller stub.
    """
    return [FederationPeer(peer_id, config, connect(config)) for peer_id, config in sorted(peers.items())]
