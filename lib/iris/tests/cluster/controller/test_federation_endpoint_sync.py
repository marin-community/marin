# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cross-cluster endpoint federation, wire round-trip.

Drives the two-controller harness from ``test_federation_handoff``: a parent hands
a ``cluster``-pinned job to a peer, the peer materializes it and registers an
endpoint on the received job's task, and the parent mirrors that endpoint over the
``federation_sync`` -> ``apply_sync_batch`` sync path until it resolves locally.

Covers the peer-side snapshot (``FederationSyncResponse.endpoints``), the
parent-side set-replace into the endpoint registry (``resolve_proxy_target``
surfaces the owning peer), removal when the peer drops the endpoint, and the
requester scoping of the snapshot.
"""

from contextlib import ExitStack

from iris.cluster.controller.federation_store import ControllerFederationStore
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.federation.manager import FederationManager
from iris.cluster.types import EndpointAccess, JobName
from iris.rpc import controller_pb2
from iris.rpc.auth import FEDERATION_PEER_ROLE
from iris.time_proto import duration_from_proto
from rigging.server_auth import VerifiedIdentity, identity_scope

from ._test_support import ControllerTestState
from .conftest import promote_queued_federation, query_task
from .test_federation_handoff import (
    _attach_federation,
    _cluster_pinned_request,
    _InProcessPeerConnection,
    _make_service,
)

# The parent cluster's id (its ``cluster_id`` / the requester it stamps on handoffs).
PARENT_ID = "parent"
# The peer's id inside the parent's federation manager.
PEER_ID = "cw"

ENDPOINT_NAME = "/serve/foo"
ENDPOINT_ADDRESS = "10.0.0.9:8000"
# Proxy-encoded form of ENDPOINT_NAME (``.`` for ``/``), how a /proxy request names it.
ENDPOINT_PROXY_NAME = "serve.foo"


def _register_peer_endpoint(
    peer_service: ControllerServiceImpl,
    peer_state: ControllerTestState,
    job_id: JobName,
    *,
    name: str = ENDPOINT_NAME,
    address: str = ENDPOINT_ADDRESS,
    access: int = EndpointAccess.ENDPOINT_ACCESS_LINK,
) -> str:
    """Register an endpoint on the peer's received-job task; return its endpoint id.

    A received job's single task is ``<job>/0``; the register RPC binds the endpoint
    to that task's current attempt.
    """
    peer_task = job_id.task(0)
    attempt_id = query_task(peer_state, peer_task).current_attempt_id
    response = peer_service.endpoint_service.register_endpoint(
        controller_pb2.Controller.RegisterEndpointRequest(
            name=name,
            address=address,
            task_id=peer_task.to_wire(),
            attempt_id=attempt_id,
            access=access,
        ),
        None,
    )
    return response.endpoint_id


def _sync_as(peer_service: ControllerServiceImpl, requester_id: str) -> controller_pb2.Controller.FederationSyncResponse:
    """Run ``federation_sync`` as the verified federation peer ``requester_id``."""
    with identity_scope(VerifiedIdentity(requester_id, FEDERATION_PEER_ROLE)):
        return peer_service.federation_sync(
            controller_pb2.Controller.FederationSyncRequest(requester_id=requester_id), None
        )


def _apply(parent_service: ControllerServiceImpl, resp: controller_pb2.Controller.FederationSyncResponse) -> None:
    """Feed a sync response into the parent's federation store (as the manager would)."""
    ControllerFederationStore(parent_service._db).apply_sync_batch(
        PEER_ID,
        list(resp.deltas),
        next_cursor=resp.next_cursor,
        cursor_stale=resp.cursor_stale,
        endpoints=list(resp.endpoints),
    )


def _endpoint_named(
    resp: controller_pb2.Controller.FederationSyncResponse, name: str
) -> controller_pb2.Controller.FederationEndpoint | None:
    return next((e for e in resp.endpoints if e.name == name), None)


def _launch_federated_job(
    parent_service: ControllerServiceImpl,
    manager: FederationManager,
    parent_state: ControllerTestState,
    name: str = "fed-job",
) -> JobName:
    """Launch a cluster-pinned job and drive the tick's federation pass so the peer
    materializes the job into a task — the post-handoff state the endpoint sync
    round-trip builds on."""
    response = parent_service.launch_job(_cluster_pinned_request(name), None)
    promote_queued_federation(manager, parent_state)
    return JobName.from_wire(response.job_id)


def test_federation_sync_reports_a_peer_registered_endpoint(tmp_path, log_client):
    """After the peer receives a handoff and an endpoint is registered on it, the
    peer's federation_sync snapshot carries that endpoint with its address, access,
    owning task, and a positive lease remaining."""
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        manager = _attach_federation(parent_service, _InProcessPeerConnection(peer_service))

        job_id = _launch_federated_job(parent_service, manager, parent_state)
        _register_peer_endpoint(peer_service, peer_state, job_id)

        resp = _sync_as(peer_service, PARENT_ID)
        endpoint = _endpoint_named(resp, ENDPOINT_NAME)
        assert endpoint is not None
        assert endpoint.address == ENDPOINT_ADDRESS
        assert endpoint.access == EndpointAccess.ENDPOINT_ACCESS_LINK
        assert endpoint.task_id == job_id.task(0).to_wire()
        assert endpoint.HasField("lease_remaining")
        assert duration_from_proto(endpoint.lease_remaining).to_ms() > 0


def test_apply_sync_batch_makes_the_endpoint_resolvable_on_the_parent(tmp_path, log_client):
    """Feeding the sync response into the parent's federation store mirrors the endpoint:
    the parent resolves the proxy name to a target tagged with the owning peer."""
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        manager = _attach_federation(parent_service, _InProcessPeerConnection(peer_service))

        job_id = _launch_federated_job(parent_service, manager, parent_state)
        _register_peer_endpoint(peer_service, peer_state, job_id)

        _apply(parent_service, _sync_as(peer_service, PARENT_ID))

        resolved = parent_service.endpoint_service.resolve_proxy_target(ENDPOINT_PROXY_NAME)
        assert resolved is not None
        assert resolved.peer_id == PEER_ID
        assert resolved.address == ENDPOINT_ADDRESS


def test_unregistered_endpoint_is_dropped_from_the_parent_on_next_sync(tmp_path, log_client):
    """When the endpoint is unregistered on the peer, the next sync no longer lists it
    and a subsequent apply_sync_batch removes the parent's mirror (resolve -> None)."""
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        manager = _attach_federation(parent_service, _InProcessPeerConnection(peer_service))

        job_id = _launch_federated_job(parent_service, manager, parent_state)
        endpoint_id = _register_peer_endpoint(peer_service, peer_state, job_id)

        _apply(parent_service, _sync_as(peer_service, PARENT_ID))
        assert parent_service.endpoint_service.resolve_proxy_target(ENDPOINT_PROXY_NAME) is not None

        peer_service.endpoint_service.unregister_endpoint(
            controller_pb2.Controller.UnregisterEndpointRequest(endpoint_id=endpoint_id), None
        )

        resp = _sync_as(peer_service, PARENT_ID)
        assert _endpoint_named(resp, ENDPOINT_NAME) is None

        _apply(parent_service, resp)
        assert parent_service.endpoint_service.resolve_proxy_target(ENDPOINT_PROXY_NAME) is None


def test_federation_sync_scopes_endpoints_to_the_requester(tmp_path, log_client):
    """The endpoint snapshot is scoped to the requester: a sync for a different
    requester id does not surface this peer's endpoints."""
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        manager = _attach_federation(parent_service, _InProcessPeerConnection(peer_service))

        job_id = _launch_federated_job(parent_service, manager, parent_state)
        _register_peer_endpoint(peer_service, peer_state, job_id)

        # The owning requester sees it; a different requester's snapshot does not.
        assert _endpoint_named(_sync_as(peer_service, PARENT_ID), ENDPOINT_NAME) is not None
        assert _endpoint_named(_sync_as(peer_service, "other-parent"), ENDPOINT_NAME) is None
