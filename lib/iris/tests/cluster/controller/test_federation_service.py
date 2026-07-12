# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Federation is observable through the RPC service, but not targetable.

``ListPeers`` returns the configured peers with the backends each forwarded on
its last heartbeat, and — the load-bearing invariant — a job submitted while a
reachable peer is configured still materializes local tasks. Nothing is handed
off.
"""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.bundle import BundleStore
from iris.cluster.config import PeerConfig
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.schema import tasks_table
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.federation.manager import FederationManager
from iris.cluster.federation.peer import FederationPeer
from iris.cluster.types import LOCAL_CLUSTER, JobName
from iris.managed_thread import get_thread_container
from iris.rpc import controller_pb2
from rigging.server_auth import VerifiedIdentity, identity_scope
from sqlalchemy import select

from .conftest import make_job_request

_IDENTITY = VerifiedIdentity(user_id="alice", role="user")

_PEER_BACKEND = controller_pb2.Controller.BackendSummary(
    backend_id="tpu-fleet",
    kind="worker-daemon",
    worker_count=3,
    advertised_attributes={"device-type": controller_pb2.StringList(values=["tpu"])},
)


class _StubPeerConnection:
    def __init__(self, backends: tuple[controller_pb2.Controller.BackendSummary, ...]):
        self._backends = backends

    def list_backends(self) -> list[controller_pb2.Controller.BackendSummary]:
        return list(self._backends)

    def shutdown(self) -> None:
        pass


def _attach_peer(
    mock_controller,
    backends: tuple[controller_pb2.Controller.BackendSummary, ...] = (_PEER_BACKEND,),
) -> FederationPeer:
    """Give the controller one reachable peer (already heartbeated once)."""
    peer = FederationPeer(
        "cw-east",
        PeerConfig(controller_address="http://cw:10000"),
        _StubPeerConnection(backends),
    )
    peer.probe()
    mock_controller.federation = FederationManager([peer], threads=get_thread_container())
    return peer


def test_list_peers_forwards_the_peer_backends_from_its_heartbeat(controller_service, mock_controller):
    _attach_peer(mock_controller)
    with identity_scope(_IDENTITY):
        response = controller_service.list_peers(controller_pb2.Controller.ListPeersRequest(), None)
    (peer,) = response.peers
    assert peer.peer_id == "cw-east"
    assert peer.reachable is True
    (backend,) = peer.backends
    assert backend.backend_id == "tpu-fleet"
    assert backend.kind == "worker-daemon"
    assert backend.worker_count == 3
    assert list(backend.advertised_attributes["device-type"].values) == ["tpu"]


def test_client_set_federation_field_is_rejected_from_a_non_admin(state, log_client, mock_controller, tmp_path):
    """The public ``federation`` handoff field marks a trusted peer-to-peer handoff.

    With auth on, a non-admin caller that sets it is forging a handoff to run a job
    as another user, so ``LaunchJob`` denies it before any owner re-pinning.
    """
    mock_controller.provider.health = state._health
    service = ControllerServiceImpl(
        controller=mock_controller,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoint_service=EndpointServiceImpl(db=state._db),
        auth=ControllerAuth(provider="test-provider"),
    )
    request = make_job_request("forged", replicas=1)
    request.federation.requester_id = "evil-cluster"
    request.federation.owner_principal = "victim"

    with identity_scope(_IDENTITY):  # role="user", not admin
        with pytest.raises(ConnectError) as exc:
            service.launch_job(request, None)
    assert exc.value.code == Code.PERMISSION_DENIED


def test_launch_job_stays_local_when_a_peer_is_configured(controller_service, mock_controller, state):
    _attach_peer(mock_controller)
    response = controller_service.launch_job(make_job_request("federation-dark", replicas=2), None)
    job_id = JobName.from_wire(response.job_id)

    with state._db.read_snapshot() as tx:
        clusters = tx.execute(select(tasks_table.c.cluster).where(tasks_table.c.job_id == job_id)).all()
    # The job materialized its tasks locally (cluster == 'local'); the reachable
    # peer attracted nothing.
    assert len(clusters) == 2
    assert all(row.cluster == LOCAL_CLUSTER for row in clusters)
