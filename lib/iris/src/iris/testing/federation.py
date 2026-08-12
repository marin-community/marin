# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""In-process federation boundary used by controller behavior tests."""

from contextlib import ExitStack
from dataclasses import replace

from rigging.server_auth import VerifiedIdentity, identity_scope

from iris.cluster.bundle import BundleStore
from iris.cluster.config import PeerConfig
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.endpoint_registry import EndpointRegistry
from iris.cluster.controller.persistence.federation import ControllerFederationStore
from iris.cluster.federation.availability import AVAILABILITY_METRIC_VERSION
from iris.cluster.federation.manager import FederationManager
from iris.cluster.federation.peer import FederationPeer
from iris.cluster.federation.protocol import (
    FederationBackendObservation,
    FederationResourceAvailability,
    FederationSyncBatch,
    HandoffDelivery,
    PeerCallError,
    PeerErrorCode,
)
from iris.cluster.types import WellKnownAttribute
from iris.managed_thread import get_thread_container
from iris.resources.names import JobName
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.endpoint_service import EndpointServiceImpl
from iris.rpc.federation_client import federation_batch_from_legacy, handoff_to_legacy_request, peer_transport_call
from iris.rpc.legacy.controller_service import LegacyControllerService
from iris.rpc.legacy.job_codec import constraint_to_proto
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
    """Peer connection that delegates to an authenticated in-process service."""

    def __init__(self, service: LegacyControllerService):
        self._service = service
        self.launch_calls = 0

    def list_backends(self) -> tuple[FederationBackendObservation, ...]:
        return ()

    def shutdown(self) -> None:
        pass

    def launch_job(self, delivery: HandoffDelivery) -> None:
        self.launch_calls += 1
        with identity_scope(PEER_IDENTITY):
            peer_transport_call(lambda: self._service.launch_job(handoff_to_legacy_request(delivery), WIRE_CONTEXT))

    def federation_sync(self, requester_id: str, cursor: str) -> FederationSyncBatch:
        with identity_scope(PEER_IDENTITY):
            response = peer_transport_call(
                lambda: self._service.federation_sync(
                    controller_pb2.Controller.FederationSyncRequest(requester_id=requester_id, cursor=cursor), None
                )
            )
        return federation_batch_from_legacy(response)

    def terminate_job(self, job_id: JobName) -> None:
        with identity_scope(PEER_IDENTITY):
            peer_transport_call(
                lambda: self._service.terminate_job(
                    controller_pb2.Controller.TerminateJobRequest(job_id=job_id.to_wire()), None
                )
            )


class UnreachablePeerConnection(InProcessPeerConnection):
    """Connection whose handoff fails and whose cancellation target is absent."""

    def launch_job(self, delivery: HandoffDelivery) -> None:
        self.launch_calls += 1
        raise PeerCallError(PeerErrorCode.UNAVAILABLE, "peer unreachable")

    def terminate_job(self, job_id: JobName) -> None:
        raise PeerCallError(PeerErrorCode.NOT_FOUND, "no such job")


class FullGpuPeerConnection(InProcessPeerConnection):
    """Reachable H100 peer with no currently free devices."""

    def list_backends(self) -> tuple[FederationBackendObservation, ...]:
        return (
            FederationBackendObservation(
                backend_id="default",
                advertised_attributes={
                    WellKnownAttribute.DEVICE_TYPE: ("gpu",),
                    WellKnownAttribute.DEVICE_VARIANT: ("h100",),
                },
                availability=FederationResourceAvailability(
                    version=AVAILABILITY_METRIC_VERSION,
                    observation_epoch_ms=1,
                    amounts={"h100": 0},
                    total_amounts={},
                    held_by_band={},
                ),
            ),
        )


class BatchOccupiedGpuPeerConnection(FullGpuPeerConnection):
    """Full GPU peer whose capacity is held by preemptible batch work."""

    def list_backends(self) -> tuple[FederationBackendObservation, ...]:
        (backend,) = super().list_backends()
        assert backend.availability is not None
        availability = replace(
            backend.availability,
            held_by_band={job_pb2.PRIORITY_BAND_BATCH: {"h100": 8}},
        )
        return (replace(backend, availability=availability),)


class RefusingPeerConnection(InProcessPeerConnection):
    """Connection whose handoff answers with a configurable peer error."""

    def __init__(self, service: LegacyControllerService, code: PeerErrorCode, message: str = "peer says no"):
        super().__init__(service)
        self.code = code
        self.message = message

    def launch_job(self, delivery: HandoffDelivery) -> None:
        self.launch_calls += 1
        raise PeerCallError(self.code, self.message)


def make_service(
    stack: ExitStack,
    subdir: str,
    tmp_path,
    log_client,
    auth: ControllerAuth | None = None,
) -> tuple[LegacyControllerService, ControllerTestState]:
    """Build a legacy wire adapter and its native controller state."""
    state = stack.enter_context(make_controller_state())
    runtime = MockController()
    runtime.provider.health = state._health
    service = make_controller_service(
        controller=runtime,
        bundle_store=BundleStore(storage_dir=str(tmp_path / subdir / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoint_service=EndpointServiceImpl(EndpointRegistry(db=state._db)),
        auth=auth,
    )
    return service, state


def attach_federation(
    parent_service: LegacyControllerService,
    parent_state: ControllerTestState,
    connection: InProcessPeerConnection,
) -> FederationManager:
    """Attach one in-process peer to the parent controller runtime."""
    peer = FederationPeer("cw", PeerConfig(controller_address="http://peer:10000"), connection)
    peer.probe()
    manager = FederationManager(
        [peer],
        threads=get_thread_container(),
        store=ControllerFederationStore(parent_state.database),
        bundles=parent_service._bundle_store,
        cluster_id="parent",
    )
    runtime = parent_service._runtime
    assert isinstance(runtime, MockController)
    runtime.federation = manager
    return manager


def cluster_pinned_request(
    name: str,
    peer: str = "cw",
    replicas: int = 1,
) -> controller_pb2.Controller.LaunchJobRequest:
    """Build a direct Job request constrained to one peer cluster."""
    request = make_direct_job_request(name, replicas=replicas)
    request.constraints.append(
        constraint_to_proto(Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value=peer))
    )
    return request
