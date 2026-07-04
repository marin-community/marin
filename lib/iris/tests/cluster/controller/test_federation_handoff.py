# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Federated handoff, FederationSync, and routed cancel end-to-end.

Wires two in-process controllers — a parent and a peer — through a delegating
``PeerConnection`` and drives the whole federated life cycle: a ``cluster``-pinned
job is handed off (no local tasks, a ``federated_jobs`` handle in ``HANDED_OFF``),
the peer materializes and owns it, one sync mirrors the peer's state back onto the
parent's handle, a routed cancel tombstones it, and the next sync drops the handle.
Also covers handoff admission/idempotency, the incremental-tombstone path, and the
full-resync set-replacement path.
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
from iris.cluster.controller.run_template import RunTemplateCache
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.federation.manager import FederationManager
from iris.cluster.federation.peer import FederationPeer
from iris.cluster.federation.store import HandoffAdmission, HandoffSpec, HandoffState
from iris.cluster.types import LOCAL_CLUSTER, AttemptUid, JobName
from iris.managed_thread import get_thread_container
from iris.rpc import controller_pb2, job_pb2
from rigging.server_auth import VerifiedIdentity, identity_scope

from ._test_support import ControllerTestState
from .conftest import (
    MockController,
    dispatch_task,
    make_controller_state,
    make_direct_job_request,
    query_job,
    query_tasks_for_job,
    register_worker,
    transition_task,
)

# The parent authenticates to the peer as itself; the peer trusts it (like a
# loopback admin) and attributes the job to the asserted owner_principal.
_PEER_IDENTITY = VerifiedIdentity(user_id="parent-cluster", role="admin")

_USER = "test-user"


class _InProcessPeerConnection:
    """A ``PeerConnection`` that delegates straight to a peer's in-process service.

    Each delegated call runs under an identity scope, mirroring an authenticated
    parent→peer RPC (``federation_sync`` requires an identity).
    """

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


class _UnreachablePeerConnection(_InProcessPeerConnection):
    """A connection whose LaunchJob always fails and whose TerminateJob 404s.

    Models a peer the handoff never reaches: delivery stays pending and a routed
    cancel finds nothing on the peer (NOT_FOUND == already satisfied)."""

    def launch_job(self, request):
        self.launch_calls += 1
        raise ConnectionError("peer unreachable")

    def terminate_job(self, job_id: JobName) -> None:
        raise ConnectError(Code.NOT_FOUND, "no such job")


def _make_service(
    stack: ExitStack, subdir: str, tmp_path, log_client
) -> tuple[ControllerServiceImpl, ControllerTestState]:
    state = stack.enter_context(make_controller_state())
    mock = MockController()
    mock.provider.health = state._health
    mock.provider.worker_attrs = state._worker_attrs
    service = ControllerServiceImpl(
        controller=mock,
        bundle_store=BundleStore(storage_dir=str(tmp_path / subdir / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoints=state._endpoints,
        endpoint_service=EndpointServiceImpl(db=state._db, endpoints=state._endpoints),
    )
    return service, state


def _attach_federation(
    parent_service: ControllerServiceImpl,
    connection: _InProcessPeerConnection,
) -> FederationManager:
    """Give ``parent_service`` a one-peer federation manager delegating to ``connection``."""
    peer = FederationPeer(
        "cw", PeerConfig(controller_address="http://peer:10000", dashboard_url="https://cw.dev"), connection
    )
    peer.probe()
    store = ControllerFederationStore(
        parent_service._db,
        run_template_cache=RunTemplateCache(256),
    )
    manager = FederationManager([peer], threads=get_thread_container(), store=store, cluster_id="parent")
    parent_service._controller.federation = manager
    return manager


def _cluster_pinned_request(name: str, peer: str = "cw") -> controller_pb2.Controller.LaunchJobRequest:
    request = make_direct_job_request(name, replicas=1)
    request.constraints.append(Constraint.create(key=CLUSTER_CONSTRAINT_KEY, op=ConstraintOp.EQ, value=peer).to_proto())
    return request


def _received_handoff_request(name: str, requester_id: str) -> controller_pb2.Controller.LaunchJobRequest:
    """A handoff request as a peer receives it: the federation field carries the
    requester (parent) cluster id and the asserted owner principal. The job id is
    cluster-invariant, so it is the plain ``/test-user/<name>`` the parent submitted."""
    request = make_direct_job_request(name, replicas=1)
    request.federation.requester_id = requester_id
    request.federation.owner_principal = _USER
    return request


def _handle(state: ControllerTestState, job_id: JobName):
    """The federated handle for ``job_id`` (or ``None``), via a scoped snapshot."""
    with state._db.read_snapshot() as tx:
        return reads.federated_handle(tx, job_id)


def _run_peer_task_to_success(peer_state: ControllerTestState, job_id: JobName) -> None:
    """Register a worker on the peer and drive the handed-off job's task to SUCCEEDED."""
    worker = register_worker(peer_state, "w1", "w1:8080", job_pb2.WorkerMetadata(hostname="w1"))
    (task,) = query_tasks_for_job(peer_state, job_id)
    dispatch_task(peer_state, task, worker)
    transition_task(peer_state, task.task_id, job_pb2.TASK_STATE_SUCCEEDED)


# ---------------------------------------------------------------------------
# handoff + sync
# ---------------------------------------------------------------------------


def test_handoff_materializes_on_peer_and_syncs_back(tmp_path, log_client):
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        manager = _attach_federation(parent_service, _InProcessPeerConnection(peer_service))

        response = parent_service.launch_job(_cluster_pinned_request("fed-job"), None)
        job_id = JobName.from_wire(response.job_id)

        # Parent side: a HANDED_OFF handle, and no local tasks (a federated root
        # owns none). Job ids are cluster-invariant, so the peer runs the same id.
        handle = _handle(parent_state, job_id)
        assert handle is not None
        assert handle.peer_id == "cw"
        assert handle.handoff_state == int(HandoffState.HANDED_OFF)
        assert handle.job_id == job_id
        assert query_tasks_for_job(parent_state, job_id) == []

        # Peer side: it materialized and OWNS the job (a RECEIVED federated_jobs
        # row, not a SENT handle) and expanded it into a task — under the same id.
        assert _handle(peer_state, job_id) is None
        assert len(query_tasks_for_job(peer_state, job_id)) == 1
        assert query_job(peer_state, job_id) is not None

        _run_peer_task_to_success(peer_state, job_id)
        manager.sync_once()

        # Parent's handle now mirrors the peer's terminal state and its task,
        # tagged with the owning peer.
        assert query_job(parent_state, job_id).state == job_pb2.JOB_STATE_SUCCEEDED
        (mirrored,) = query_tasks_for_job(parent_state, job_id)
        assert mirrored.state == job_pb2.TASK_STATE_SUCCEEDED
        assert mirrored.cluster == "cw"


def test_sync_mirrors_attempts_and_worker_identity_natively(tmp_path, log_client):
    """After sync-back the parent renders a federated task natively: the peer's
    attempt history is mirrored and the peer-side worker identity is surfaced (as
    display text — there is no local worker row), and the mirrored attempt stays
    off the worker-routing fold (its namespaced uid never resolves)."""
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        manager = _attach_federation(parent_service, _InProcessPeerConnection(peer_service))

        response = parent_service.launch_job(_cluster_pinned_request("fed-job"), None)
        job_id = JobName.from_wire(response.job_id)

        _run_peer_task_to_success(peer_state, job_id)
        manager.sync_once()

        (mirrored,) = query_tasks_for_job(parent_state, job_id)
        task = parent_service.get_task_status(
            controller_pb2.Controller.GetTaskStatusRequest(task_id=mirrored.task_id.to_wire()), None
        ).task

        # The peer's attempt renders natively, and the worker identity is surfaced
        # from the peer (the task has no local worker row, so it is display-only).
        assert task.cluster == "cw"
        assert task.worker_id == "w1"
        assert len(task.attempts) == 1
        assert task.attempts[0].state == job_pb2.TASK_STATE_SUCCEEDED
        assert task.attempts[0].worker_id == ""  # no local worker FK for a mirrored attempt

        # The mirrored uid is the peer's raw uid, written verbatim (no peer-prefix
        # rebasing — job ids are cluster-invariant). Because uid resolution is scoped
        # to local_tasks, it never resolves — so reconcile's worker-routing can never
        # act on a federated attempt.
        with peer_state._db.read_snapshot() as tx:
            (peer_attempt,) = reads.all_attempts_for_tasks(tx, [job_id.task(0)])[job_id.task(0)]
        with parent_state._db.read_snapshot() as tx:
            (attempt_row,) = reads.all_attempts_for_tasks(tx, [mirrored.task_id])[mirrored.task_id]
            assert attempt_row.attempt_uid == peer_attempt.attempt_uid
            assert "~" not in attempt_row.attempt_uid
            assert reads.resolve_attempt_uids(tx, [AttemptUid(attempt_row.attempt_uid)]) == {}


def test_dashboard_reads_expose_cluster_and_filter_by_it(tmp_path, log_client):
    """The dashboard reads see a federated job: GetJobStatus stamps ``cluster``, and
    the ListJobs ``cluster`` filter isolates federated jobs from local ones."""
    with ExitStack() as stack:
        parent_service, _ = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, _ = _make_service(stack, "peer", tmp_path, log_client)
        _attach_federation(parent_service, _InProcessPeerConnection(peer_service))

        fed = parent_service.launch_job(_cluster_pinned_request("fed-job"), None)
        local = parent_service.launch_job(make_direct_job_request("local-job", replicas=1), None)

        # GetJobStatus exposes the cluster coordinate: the owning peer for a federated job.
        fed_status = parent_service.get_job_status(
            controller_pb2.Controller.GetJobStatusRequest(job_id=fed.job_id), None
        ).job
        assert fed_status.cluster == "cw"
        # Handed off but not yet reported on: the pending reason names the peer,
        # not the local scheduler diagnostic (which never sees a federated job).
        assert fed_status.pending_reason == "Handed off to peer cw; awaiting first status report"

        # A local job carries the reserved 'local' sentinel and keeps the local
        # scheduler diagnostic.
        local_status = parent_service.get_job_status(
            controller_pb2.Controller.GetJobStatusRequest(job_id=local.job_id), None
        ).job
        assert local_status.cluster == LOCAL_CLUSTER
        assert "peer" not in local_status.pending_reason.lower()

        def _list(cluster: str) -> set[str]:
            resp = parent_service.list_jobs(
                controller_pb2.Controller.ListJobsRequest(query=controller_pb2.Controller.JobQuery(cluster=cluster)),
                None,
            )
            return {j.job_id for j in resp.jobs}

        # Unfiltered: both jobs. Filtered to the peer: only the federated one.
        assert {fed.job_id, local.job_id} <= _list("")
        assert _list("cw") == {fed.job_id}
        assert _list("no-such-peer") == set()


def test_federated_pending_reason_reflects_awaiting_acceptance(tmp_path, log_client):
    """While a handoff is undelivered (PENDING_HANDOFF), the pending reason says the
    job is awaiting the peer's acceptance — distinguishing it from a handed-off job."""
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, _ = _make_service(stack, "peer", tmp_path, log_client)
        _attach_federation(parent_service, _UnreachablePeerConnection(peer_service))

        response = parent_service.launch_job(_cluster_pinned_request("awaiting-ack"), None)
        assert _handle(parent_state, JobName.from_wire(response.job_id)).handoff_state == int(
            HandoffState.PENDING_HANDOFF
        )

        status = parent_service.get_job_status(
            controller_pb2.Controller.GetJobStatusRequest(job_id=response.job_id), None
        ).job
        assert status.pending_reason == "Awaiting acceptance by peer cw"


def test_cancel_routes_to_peer_and_tombstone_drops_the_handle(tmp_path, log_client):
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        manager = _attach_federation(parent_service, _InProcessPeerConnection(peer_service))

        response = parent_service.launch_job(_cluster_pinned_request("fed-job"), None)
        parent_job_id = JobName.from_wire(response.job_id)

        # Cancel the parent handle: it routes TerminateJob to the peer, which kills
        # the job there (the same, cluster-invariant id).
        parent_service.terminate_job(controller_pb2.Controller.TerminateJobRequest(job_id=parent_job_id.to_wire()), None)
        assert query_job(peer_state, parent_job_id).state == job_pb2.JOB_STATE_KILLED

        # The peer prunes the terminal job (writing a tombstone); the next sync
        # applies it and the parent drops the handle and its jobs row.
        with peer_state._db.transaction() as cur:
            writes.delete_job(cur, parent_job_id)
        manager.sync_once()

        assert _handle(parent_state, parent_job_id) is None
        assert query_job(parent_state, parent_job_id) is None


def test_full_resync_drops_a_handle_absent_from_the_peers_active_set(tmp_path, log_client):
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        manager = _attach_federation(parent_service, _InProcessPeerConnection(peer_service))

        response = parent_service.launch_job(_cluster_pinned_request("fed-job"), None)
        parent_job_id = JobName.from_wire(response.job_id)
        manager.sync_once()  # parent's cursor advances past the peer's current max seq
        assert query_job(parent_state, parent_job_id) is not None

        # The peer prunes the job, and the parent loses its cursor (reset to "", as
        # after a state reset / first contact). The next sync is therefore a full
        # resync, whose active set no longer contains the job — so the parent drops
        # it by set-replacement, not by a tombstone delta.
        with peer_state._db.transaction() as cur:
            writes.delete_job(cur, parent_job_id)
        with parent_state._db.transaction() as cur:
            writes.upsert_sync_cursor(cur, "cw", "")
        manager.sync_once()

        assert _handle(parent_state, parent_job_id) is None
        assert query_job(parent_state, parent_job_id) is None


def test_cancel_while_pending_handoff_is_never_delivered(tmp_path, log_client):
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, _peer_state = _make_service(stack, "peer", tmp_path, log_client)
        connection = _UnreachablePeerConnection(peer_service)
        manager = _attach_federation(parent_service, connection)

        # Delivery fails: the handle persists in PENDING_HANDOFF.
        response = parent_service.launch_job(_cluster_pinned_request("fed-job"), None)
        parent_job_id = JobName.from_wire(response.job_id)
        assert connection.launch_calls == 1
        handle = _handle(parent_state, parent_job_id)
        assert handle.handoff_state == int(HandoffState.PENDING_HANDOFF)

        # Cancelling a pending handoff bumps its intent; the sync loop's re-drive
        # must then never deliver the job the user already cancelled.
        parent_service.terminate_job(controller_pb2.Controller.TerminateJobRequest(job_id=parent_job_id.to_wire()), None)
        manager.sync_once()
        assert connection.launch_calls == 1  # no redelivery after cancel
        # The job the peer never received is terminated locally, not left pending.
        assert query_job(parent_state, parent_job_id).state == job_pb2.JOB_STATE_KILLED


def test_redrive_of_a_handle_the_peer_already_has_is_idempotent(tmp_path, log_client):
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        connection = _InProcessPeerConnection(peer_service)
        manager = _attach_federation(parent_service, connection)

        response = parent_service.launch_job(_cluster_pinned_request("fed-job"), None)
        parent_job_id = JobName.from_wire(response.job_id)
        assert connection.launch_calls == 1

        # Force the handle back to PENDING_HANDOFF (as if the parent crashed after
        # delivery but before recording it). The re-drive must re-send under the
        # same id and the peer's federation-aware admission dedups — no second job,
        # no error, and the handle settles in HANDED_OFF.
        with parent_state._db.transaction() as cur:
            writes.set_handoff_state(cur, parent_job_id, int(HandoffState.PENDING_HANDOFF))
        manager.sync_once()

        assert connection.launch_calls == 2  # re-sent once
        assert _handle(parent_state, parent_job_id).handoff_state == int(HandoffState.HANDED_OFF)
        assert len(query_tasks_for_job(peer_state, parent_job_id)) == 1  # idempotent re-drive — no duplicate


# ---------------------------------------------------------------------------
# admission + incremental tombstone
# ---------------------------------------------------------------------------


def test_admit_persists_a_pending_handle_and_is_idempotent(tmp_path, log_client):
    with ExitStack() as stack:
        _parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        store = ControllerFederationStore(parent_state._db, run_template_cache=RunTemplateCache(256))
        parent_job_id = JobName.root(_USER, "fed-job")
        spec = HandoffSpec(
            local_job_id=parent_job_id,
            peer_id="cw",
            owner_principal=_USER,
            request=make_direct_job_request("fed-job", replicas=1),
        )

        assert store.admit_and_persist_handoff(spec) is HandoffAdmission.ADMITTED
        handle = _handle(parent_state, parent_job_id)
        assert handle is not None
        assert handle.handoff_state == int(HandoffState.PENDING_HANDOFF)

        # A re-submit of the same job is idempotent — no second handle, no error.
        assert store.admit_and_persist_handoff(spec) is HandoffAdmission.ALREADY_EXISTS


def test_peer_admission_dedups_a_redrive_and_rejects_a_collision(tmp_path, log_client):
    """Peer-side handoff admission for a job id that already exists: a received handoff
    is idempotent when re-driven from the SAME requester (returns the job, no duplicate
    tasks), but is rejected with ``ALREADY_EXISTS`` when the existing job is local or
    was received from a different requester."""
    with ExitStack() as stack:
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)

        # First receipt from requester "parent": the peer materializes it locally as
        # an ordinary job with a RECEIVED handle recording the requester.
        with identity_scope(_PEER_IDENTITY):
            first = peer_service.launch_job(_received_handoff_request("fed-job", "parent"), None)
        job_id = JobName.from_wire(first.job_id)
        assert len(query_tasks_for_job(peer_state, job_id)) == 1

        # (a) A re-drive from the SAME requester is idempotent: same job, no dup tasks
        # (the boot-recovery / retry path).
        with identity_scope(_PEER_IDENTITY):
            again = peer_service.launch_job(_received_handoff_request("fed-job", "parent"), None)
        assert again.job_id == first.job_id
        assert len(query_tasks_for_job(peer_state, job_id)) == 1

        # (b) A handoff for the same id from a DIFFERENT requester is a genuine
        # collision the parent must see — rejected, not silently bound to the wrong job.
        with identity_scope(_PEER_IDENTITY):
            with pytest.raises(ConnectError) as exc:
                peer_service.launch_job(_received_handoff_request("fed-job", "other-parent"), None)
        assert exc.value.code == Code.ALREADY_EXISTS

        # (c) A handoff colliding with a purely LOCAL job (no RECEIVED handle) is
        # rejected too.
        peer_service.launch_job(make_direct_job_request("local-job", replicas=1), None)
        with identity_scope(_PEER_IDENTITY):
            with pytest.raises(ConnectError) as exc:
                peer_service.launch_job(_received_handoff_request("local-job", "parent"), None)
        assert exc.value.code == Code.ALREADY_EXISTS


def test_incremental_sync_delivers_a_tombstone_and_drops_the_handle(tmp_path, log_client):
    """A prune's tombstone reaches the parent on the INCREMENTAL path (cursor already
    advanced), not only via a full resync: each changelog row carries its requester,
    so the tombstone is attributable after the received job (and its row) is gone.
    """
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        manager = _attach_federation(parent_service, _InProcessPeerConnection(peer_service))

        response = parent_service.launch_job(_cluster_pinned_request("fed-job"), None)
        parent_job_id = JobName.from_wire(response.job_id)
        manager.sync_once()  # advance the parent's cursor past the peer's current max seq
        assert query_job(parent_state, parent_job_id) is not None

        # Prune on the peer AFTER the parent is caught up, so only the incremental
        # tombstone (not a first-contact full resync) can reclaim the handle.
        with peer_state._db.transaction() as cur:
            writes.delete_job(cur, parent_job_id)
        manager.sync_once()

        assert _handle(parent_state, parent_job_id) is None
        assert query_job(parent_state, parent_job_id) is None
