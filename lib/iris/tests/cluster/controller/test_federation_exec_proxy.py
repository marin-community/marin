# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exec / profile against a federated job proxy through the peer controller.

Wires two in-process controllers — a parent and a peer — through a delegating
``PeerConnection`` and drives the on-demand RPC path: a job is handed off, its
task runs on the peer and mirrors back onto the parent, and a profile / exec
issued against the parent's mirrored task is forwarded verbatim to the peer under
the same, cluster-invariant job id (not run locally). Also covers the race
outcomes: the peer's live ``NOT_FOUND`` for a task it has since dropped is
surfaced verbatim, and a tombstoned handle resolves to ``NOT_FOUND`` with no peer
round-trip.
"""

from contextlib import ExitStack

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.bundle import BundleStore
from iris.cluster.config import PeerConfig
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.controller import reads, writes
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.federation_store import ControllerFederationStore
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.federation.manager import FederationManager
from iris.cluster.federation.peer import FederationPeer
from iris.cluster.types import JobName
from iris.managed_thread import get_thread_container
from iris.rpc import controller_pb2, job_pb2, worker_pb2
from rigging.server_auth import VerifiedIdentity, identity_scope

from ._test_support import ControllerTestState
from .conftest import (
    MockController,
    dispatch_task,
    make_controller_state,
    make_direct_job_request,
    promote_queued_federation,
    query_tasks_for_job,
    register_worker,
)

# The parent authenticates to the peer as itself; the peer trusts it and runs the
# delegated RPC under the asserted identity (profiling/exec never run anonymously).
_PEER_IDENTITY = VerifiedIdentity(user_id="parent-cluster", role="admin")

_CPU_PROFILE = job_pb2.ProfileType(cpu=job_pb2.CpuProfile())


class _ProxyPeerConnection:
    """A ``PeerConnection`` that delegates straight to a peer's in-process service.

    Counts the proxied on-demand calls so a test can assert a tombstoned handle
    short-circuits without a peer round-trip.
    """

    def __init__(self, service: ControllerServiceImpl):
        self._service = service
        self.profile_calls = 0
        self.exec_calls = 0

    def list_backends(self) -> list[controller_pb2.Controller.BackendSummary]:
        return []

    def shutdown(self) -> None:
        pass

    def launch_job(
        self, request: controller_pb2.Controller.LaunchJobRequest
    ) -> controller_pb2.Controller.LaunchJobResponse:
        with identity_scope(_PEER_IDENTITY):
            return self._service.launch_job(request, None)

    def federation_sync(
        self, request: controller_pb2.Controller.FederationSyncRequest
    ) -> controller_pb2.Controller.FederationSyncResponse:
        with identity_scope(_PEER_IDENTITY):
            return self._service.federation_sync(request, None)

    def terminate_job(self, job_id: JobName) -> None:
        with identity_scope(_PEER_IDENTITY):
            self._service.terminate_job(controller_pb2.Controller.TerminateJobRequest(job_id=job_id.to_wire()), None)

    def profile_task(self, request: job_pb2.ProfileTaskRequest) -> job_pb2.ProfileTaskResponse:
        self.profile_calls += 1
        with identity_scope(_PEER_IDENTITY):
            return self._service.profile_task(request, None)

    def exec_in_container(
        self, request: controller_pb2.Controller.ExecInContainerRequest
    ) -> controller_pb2.Controller.ExecInContainerResponse:
        self.exec_calls += 1
        with identity_scope(_PEER_IDENTITY):
            return self._service.exec_in_container(request, None)


def _make_service(
    stack: ExitStack, subdir: str, tmp_path, log_client
) -> tuple[ControllerServiceImpl, ControllerTestState]:
    state = stack.enter_context(make_controller_state())
    mock = MockController()
    mock.provider.health = state._health
    service = ControllerServiceImpl(
        controller=mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / subdir / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoint_service=EndpointServiceImpl(db=state._db),
    )
    return service, state


def _attach_federation(parent_service: ControllerServiceImpl, connection: _ProxyPeerConnection) -> FederationManager:
    peer = FederationPeer("cw", PeerConfig(controller_address="http://peer:10000"), connection)
    peer.probe()
    store = ControllerFederationStore(parent_service._db)
    manager = FederationManager([peer], threads=get_thread_container(), store=store, cluster_id="parent")
    parent_service._controller.federation = manager
    return manager


def _cluster_pinned_request(name: str, peer: str = "cw") -> controller_pb2.Controller.LaunchJobRequest:
    request = make_direct_job_request(name, replicas=1)
    request.constraints.append(Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value=peer).to_proto())
    return request


def _handle(state: ControllerTestState, job_id: JobName):
    with state._db.read_snapshot() as tx:
        return reads.federated_handle(tx, job_id)


def _handoff_and_mirror_running_task(
    parent_service: ControllerServiceImpl,
    parent_state: ControllerTestState,
    peer_state: ControllerTestState,
    manager: FederationManager,
    name: str = "fed-job",
) -> JobName:
    """Hand off a job, drive its peer task to RUNNING, and mirror it back.

    Returns the job id — cluster-invariant, so the parent and the peer name the
    job identically. The parent's mirror then holds a live federated task
    (``cluster`` set, no local worker), the exact row an on-demand RPC must proxy
    rather than resolve locally. The pinned submission only queues the job, so the
    control-tick promotion is driven here before the peer sees any task.
    """
    response = parent_service.launch_job(_cluster_pinned_request(name), None)
    job_id = JobName.from_wire(response.job_id)
    promote_queued_federation(manager, parent_state)

    worker = register_worker(peer_state, "w1", "w1:8080", job_pb2.WorkerMetadata(hostname="w1"))
    (task,) = query_tasks_for_job(peer_state, job_id)
    dispatch_task(peer_state, task, worker)
    manager.sync_once()
    return job_id


def test_profile_against_a_federated_task_runs_on_the_peer(tmp_path, log_client):
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        manager = _attach_federation(parent_service, _ProxyPeerConnection(peer_service))
        job_id = _handoff_and_mirror_running_task(parent_service, parent_state, peer_state, manager)

        peer_service._controller.provider.profile_task.return_value = job_pb2.ProfileTaskResponse(
            profile_data=b"peer-profile"
        )
        resp = parent_service.profile_task(
            job_pb2.ProfileTaskRequest(
                target=job_id.task(0).to_wire(),
                duration_seconds=1,
                profile_type=_CPU_PROFILE,
            ),
            None,
        )

        # The bytes could only have come from the peer's backend.
        assert resp.profile_data == b"peer-profile"
        # The federated task was never dispatched to the parent's local fallback backend.
        parent_service._controller.provider.profile_task.assert_not_called()
        # The peer resolved the task under the same, cluster-invariant job id.
        (call,) = peer_service._controller.provider.profile_task.call_args_list
        assert call.args[0].task_id == job_id.task(0).to_wire()


def test_profile_preserves_the_attempt_qualifier_when_proxying(tmp_path, log_client):
    """A ``:attempt`` target is forwarded verbatim (task index + attempt intact),
    so the peer profiles the exact attempt the user named."""
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        manager = _attach_federation(parent_service, _ProxyPeerConnection(peer_service))
        job_id = _handoff_and_mirror_running_task(parent_service, parent_state, peer_state, manager)

        peer_service._controller.provider.profile_task.return_value = job_pb2.ProfileTaskResponse(profile_data=b"ok")
        parent_service.profile_task(
            job_pb2.ProfileTaskRequest(
                target=f"{job_id.task(0).to_wire()}:0",
                duration_seconds=1,
                profile_type=_CPU_PROFILE,
            ),
            None,
        )

        (call,) = peer_service._controller.provider.profile_task.call_args_list
        # The forwarded request carries the target verbatim, with the attempt kept.
        assert call.args[1].target == f"{job_id.task(0).to_wire()}:0"


def test_exec_against_a_federated_task_runs_on_the_peer(tmp_path, log_client):
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        manager = _attach_federation(parent_service, _ProxyPeerConnection(peer_service))
        job_id = _handoff_and_mirror_running_task(parent_service, parent_state, peer_state, manager)

        peer_service._controller.provider.exec_in_container.return_value = worker_pb2.Worker.ExecInContainerResponse(
            exit_code=0, stdout="hello"
        )
        resp = parent_service.exec_in_container(
            controller_pb2.Controller.ExecInContainerRequest(
                task_id=job_id.task(0).to_wire(),
                command=["echo", "hi"],
                timeout_seconds=5,
            ),
            None,
        )

        assert resp.exit_code == 0
        assert resp.stdout == "hello"
        # Never dispatched to the parent's local fallback backend.
        parent_service._controller.provider.exec_in_container.assert_not_called()
        # The peer resolved the task, and its forwarded worker request both carry the
        # same, cluster-invariant task id.
        (call,) = peer_service._controller.provider.exec_in_container.call_args_list
        assert call.args[0].task_id == job_id.task(0).to_wire()
        assert call.args[1].task_id == job_id.task(0).to_wire()


def test_exec_forwards_a_task_id_whose_job_name_contains_a_colon(tmp_path, log_client):
    """A ':' in a job-name component is a legal name char, not an attempt separator
    for an exec task id — the parent must parse it as a JobName (not a TaskAttempt)
    to find the federated handle before forwarding the id verbatim to the peer."""
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        manager = _attach_federation(parent_service, _ProxyPeerConnection(peer_service))
        job_id = _handoff_and_mirror_running_task(parent_service, parent_state, peer_state, manager, name="train:debug")

        peer_service._controller.provider.exec_in_container.return_value = worker_pb2.Worker.ExecInContainerResponse(
            exit_code=0
        )
        resp = parent_service.exec_in_container(
            controller_pb2.Controller.ExecInContainerRequest(task_id=job_id.task(0).to_wire(), command=["true"]),
            None,
        )

        assert resp.exit_code == 0
        (call,) = peer_service._controller.provider.exec_in_container.call_args_list
        assert call.args[0].task_id == job_id.task(0).to_wire()


def test_exec_surfaces_the_peers_not_found_for_a_stale_mirror(tmp_path, log_client):
    """The parent's mirror is last-sync stale: an exec may target a task the peer has
    dropped. The peer is authoritative — its ``NOT_FOUND`` is surfaced verbatim, not
    guessed from the still-present cached row."""
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        manager = _attach_federation(parent_service, _ProxyPeerConnection(peer_service))
        job_id = _handoff_and_mirror_running_task(parent_service, parent_state, peer_state, manager)

        # The peer drops the job; the parent does NOT sync, so its mirror still shows
        # the task running.
        with peer_state._db.transaction() as cur:
            writes.delete_job(cur, job_id)

        with pytest.raises(ConnectError) as exc:
            parent_service.exec_in_container(
                controller_pb2.Controller.ExecInContainerRequest(task_id=job_id.task(0).to_wire(), command=["true"]),
                None,
            )
        assert exc.value.code == Code.NOT_FOUND
        # The peer's local backend was never reached — the task is gone on the peer.
        peer_service._controller.provider.exec_in_container.assert_not_called()


def test_tombstoned_handle_resolves_not_found_without_a_peer_round_trip(tmp_path, log_client):
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        connection = _ProxyPeerConnection(peer_service)
        manager = _attach_federation(parent_service, connection)
        job_id = _handoff_and_mirror_running_task(parent_service, parent_state, peer_state, manager)

        # The peer prunes the job and the parent syncs the tombstone: its handle and
        # the mirrored task rows are dropped.
        with peer_state._db.transaction() as cur:
            writes.delete_job(cur, job_id)
        manager.sync_once()
        assert _handle(parent_state, job_id) is None

        with pytest.raises(ConnectError) as exc:
            parent_service.profile_task(
                job_pb2.ProfileTaskRequest(
                    target=job_id.task(0).to_wire(),
                    duration_seconds=1,
                    profile_type=_CPU_PROFILE,
                ),
                None,
            )
        assert exc.value.code == Code.NOT_FOUND
        # The dropped mirror short-circuits to NOT_FOUND before any peer call.
        assert connection.profile_calls == 0
