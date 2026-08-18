# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-process federation boundary used by controller behavior tests."""

from contextlib import ExitStack

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.server_auth import VerifiedIdentity, identity_scope

from iris.cluster.config import PeerConfig
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.federation_store import ControllerFederationStore
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.federation.availability import AVAILABILITY_METRIC_VERSION
from iris.cluster.federation.manager import FederationManager
from iris.cluster.federation.peer import FederationPeer
from iris.cluster.types import WellKnownAttribute
from iris.managed_thread import get_thread_container
from iris.rpc import controller_pb2, job_pb2, resource_pb2
from iris.testing.controller import (
    MockController,
    make_controller_service,
    make_controller_state,
    make_direct_job_request,
)
from iris.testing.controller_state import ControllerTestState

PEER_IDENTITY = VerifiedIdentity(user_id="parent-cluster", role="admin")
WIRE_CONTEXT = object()


class InProcessPeerConnection:
    """Peer connection that delegates directly to an in-process service."""

    def __init__(self, service: ControllerServiceImpl):
        self._service = service
        self.launch_calls = 0

    def list_backends(self) -> list[controller_pb2.Controller.BackendSummary]:
        return []

    def shutdown(self) -> None:
        pass

    def launch_job(
        self, request: controller_pb2.Controller.LaunchJobRequest
    ) -> controller_pb2.Controller.LaunchJobResponse:
        self.launch_calls += 1
        with identity_scope(PEER_IDENTITY):
            return self._service.launch_job(request, WIRE_CONTEXT)

    def federation_sync(
        self, request: controller_pb2.Controller.FederationSyncRequest
    ) -> controller_pb2.Controller.FederationSyncResponse:
        with identity_scope(PEER_IDENTITY):
            return self._service.federation_sync(request, None)

    def get_service_info(self) -> resource_pb2.GetServiceInfoResponse:
        with identity_scope(PEER_IDENTITY):
            return self._resource_service().get_service_info(resource_pb2.GetServiceInfoRequest(), WIRE_CONTEXT)

    def update_resource(self, request: resource_pb2.UpdateResourceRequest) -> resource_pb2.Operation:
        with identity_scope(PEER_IDENTITY):
            return self._resource_service().update_resource(request, WIRE_CONTEXT)

    def get_resource(self, request: resource_pb2.GetResourceRequest) -> resource_pb2.GetResourceResponse:
        with identity_scope(PEER_IDENTITY):
            return self._resource_service().get_resource(request, WIRE_CONTEXT)

    def _resource_service(self):
        assert self._service._resource_service is not None
        return self._service._resource_service


class UnreachablePeerConnection(InProcessPeerConnection):
    """Peer connection whose launch fails."""

    def launch_job(self, request):
        self.launch_calls += 1
        raise ConnectionError("peer unreachable")


class FullGpuPeerConnection(InProcessPeerConnection):
    """Reachable peer advertising an H100 backend with no free chips."""

    def list_backends(self) -> list[controller_pb2.Controller.BackendSummary]:
        summary = controller_pb2.Controller.BackendSummary(
            backend_id="default",
            advertised_attributes={
                WellKnownAttribute.DEVICE_TYPE: controller_pb2.StringList(values=["gpu"]),
                WellKnownAttribute.DEVICE_VARIANT: controller_pb2.StringList(values=["h100"]),
            },
        )
        summary.availability.version = AVAILABILITY_METRIC_VERSION
        summary.availability.observation_epoch_ms = 1
        summary.availability.amounts["h100"] = 0
        return [summary]


class BatchOccupiedGpuPeerConnection(FullGpuPeerConnection):
    """Full GPU peer whose capacity is held by preemptible batch work."""

    def list_backends(self) -> list[controller_pb2.Controller.BackendSummary]:
        summaries = super().list_backends()
        summaries[0].availability.held_by_band.add(band=job_pb2.PRIORITY_BAND_BATCH, amounts={"h100": 8})
        return summaries


class RefusingPeerConnection(InProcessPeerConnection):
    """Peer connection whose launch answers with a configurable error code."""

    def __init__(self, service: ControllerServiceImpl, code: Code, message: str = "peer says no"):
        super().__init__(service)
        self.code = code
        self.message = message

    def launch_job(self, request):
        self.launch_calls += 1
        raise ConnectError(self.code, self.message)


def make_service(
    stack: ExitStack, subdir: str, tmp_path, log_client, auth: ControllerAuth | None = None
) -> tuple[ControllerServiceImpl, ControllerTestState]:
    """Build a controller service and its test state inside ``stack``."""
    state = stack.enter_context(make_controller_state())
    mock = MockController()
    mock.provider.health = state._health
    service = make_controller_service(state, log_client, mock, tmp_path / subdir, auth=auth, cluster_id=subdir)
    return service, state


def attach_federation(
    parent_service: ControllerServiceImpl,
    connection: InProcessPeerConnection,
    *,
    resource_api: bool = True,
) -> FederationManager:
    """Attach a one-peer federation manager to ``parent_service``."""
    if resource_api:
        peer = FederationPeer("cw", PeerConfig(controller_address="http://peer:10000"), connection, connection)
    else:
        peer = FederationPeer("cw", PeerConfig(controller_address="http://peer:10000"), connection)
    peer.probe()
    store = ControllerFederationStore(parent_service._db)
    manager = FederationManager(
        [peer],
        threads=get_thread_container(),
        store=store,
        bundles=parent_service._bundle_store,
        cluster_id="parent",
    )
    controller = parent_service._controller
    assert isinstance(controller, MockController)
    controller.federation = manager
    return manager


def progress_resource_operations(service: ControllerServiceImpl) -> None:
    """Run the operation phase that a production controller executes after a tick."""
    assert service._workload_actions is not None
    service._workload_actions.progress(verification_candidates={}, reconciled_backends=frozenset())


def cluster_pinned_request(name: str, peer: str = "cw", replicas: int = 1) -> controller_pb2.Controller.LaunchJobRequest:
    """Build a direct job request constrained to one peer cluster."""
    request = make_direct_job_request(name, replicas=replicas)
    request.constraints.append(Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value=peer).to_proto())
    return request
