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
from iris.cluster.bundle import BundleStore, content_id
from iris.cluster.config import PeerConfig
from iris.cluster.constraints import CLUSTER_CONSTRAINT_KEY, Constraint, ConstraintOp
from iris.cluster.controller import reads, writes
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.federation_store import ControllerFederationStore
from iris.cluster.controller.service import WORKDIR_FILE_OFFLOAD_THRESHOLD, ControllerServiceImpl, _peer_status
from iris.cluster.federation.manager import FederationManager
from iris.cluster.federation.peer import FederationPeer
from iris.cluster.federation.store import HandoffAdmission, HandoffSpec, HandoffState
from iris.cluster.types import LOCAL_ADMIN_SUBMITTER, LOCAL_CLUSTER, AttemptUid, JobName
from iris.managed_thread import get_thread_container
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.auth import FEDERATION_PEER_ROLE
from rigging.server_auth import VerifiedIdentity, identity_scope

from ._test_support import ControllerTestState
from .conftest import (
    MockController,
    dispatch_task,
    make_controller_state,
    make_direct_job_request,
    query_job,
    query_task,
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


class _RefusingPeerConnection(_InProcessPeerConnection):
    """A connection whose LaunchJob answers with ``code`` (mutable between attempts).

    Models a peer that answers the handoff itself rather than dropping it: a
    terminal code is its verdict and repeats on every retry; a transient one may
    clear on a later attempt.
    """

    def __init__(self, service: ControllerServiceImpl, code: Code, message: str = "peer says no"):
        super().__init__(service)
        self.code = code
        self.message = message

    def launch_job(self, request):
        self.launch_calls += 1
        raise ConnectError(self.code, self.message)


def _make_service(
    stack: ExitStack, subdir: str, tmp_path, log_client, auth: ControllerAuth | None = None
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
        auth=auth,
    )
    return service, state


def _attach_federation(
    parent_service: ControllerServiceImpl,
    connection: _InProcessPeerConnection,
) -> FederationManager:
    """Give ``parent_service`` a one-peer federation manager delegating to ``connection``."""
    peer = FederationPeer("cw", PeerConfig(controller_address="http://peer:10000"), connection)
    peer.probe()
    store = ControllerFederationStore(
        parent_service._db,
    )
    manager = FederationManager(
        [peer],
        threads=get_thread_container(),
        store=store,
        bundles=parent_service._bundle_store,
        cluster_id="parent",
    )
    parent_service._controller.federation = manager
    return manager


def _cluster_pinned_request(
    name: str, peer: str = "cw", replicas: int = 1
) -> controller_pb2.Controller.LaunchJobRequest:
    request = make_direct_job_request(name, replicas=replicas)
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


def _peer_status_of(service: ControllerServiceImpl, job_id: JobName) -> int:
    """The ``peer_status`` GetJobStatus reports for ``job_id``."""
    return service.get_job_status(
        controller_pb2.Controller.GetJobStatusRequest(job_id=job_id.to_wire()), None
    ).job.peer_status


def _run_peer_task_to_success(peer_state: ControllerTestState, job_id: JobName) -> None:
    """Register a worker on the peer and drive the handed-off job's task to SUCCEEDED."""
    worker = register_worker(peer_state, "w1", "w1:8080", job_pb2.WorkerMetadata(hostname="w1"))
    (task,) = query_tasks_for_job(peer_state, job_id)
    dispatch_task(peer_state, task, worker)
    transition_task(peer_state, task.task_id, job_pb2.TASK_STATE_SUCCEEDED)


# ---------------------------------------------------------------------------
# blobs: a content id resolves only against the store that minted it
# ---------------------------------------------------------------------------


def test_a_federated_job_carries_its_workspace_bundle_to_the_peer(tmp_path, log_client):
    """The peer's tasks fetch the bundle from the peer's own store, so the handoff
    carries the bytes rather than the parent's content id."""
    blob = b"PK\x03\x04 pretend workspace zip"
    with ExitStack() as stack:
        parent_service, _ = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, _ = _make_service(stack, "peer", tmp_path, log_client)
        _attach_federation(parent_service, _InProcessPeerConnection(peer_service))

        request = _cluster_pinned_request("fed-bundle")
        request.bundle_blob = blob
        parent_service.launch_job(request, None)

        assert peer_service._bundle_store.get(content_id(blob)) == blob


def test_a_federated_job_carries_its_externalized_workdir_files_to_the_peer(tmp_path, log_client):
    """A workdir file large enough to be externalized becomes a content id in the
    parent's store; the peer must receive the bytes, not that id."""
    big = b"x" * (WORKDIR_FILE_OFFLOAD_THRESHOLD + 1)
    with ExitStack() as stack:
        parent_service, _ = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, _ = _make_service(stack, "peer", tmp_path, log_client)
        _attach_federation(parent_service, _InProcessPeerConnection(peer_service))

        request = _cluster_pinned_request("fed-workdir")
        request.entrypoint.workdir_files["big.bin"] = big
        parent_service.launch_job(request, None)

        assert peer_service._bundle_store.get(content_id(big)) == big


# ---------------------------------------------------------------------------
# inbound admission: an enforcing peer gates who may hand off
# ---------------------------------------------------------------------------

# An enforcing peer (a provider is configured) that admits only openathena submitters.
_ENFORCING_AUTH = ControllerAuth(provider="cidr", allowed_submitters=("*@openathena.ai",))


def _peer_handoff_request(name: str, requester_id: str, submitting_user: str):
    """A handoff request carrying the peer-verified requester and the asserted submitter."""
    request = _received_handoff_request(name, requester_id)
    request.federation.submitting_user = submitting_user
    return request


def test_inbound_handoff_admits_a_verified_peer_for_an_allowed_submitter(tmp_path, log_client):
    """An enforcing peer admits a handoff whose federation-peer identity matches the
    asserted requester and whose submitter the allowlist permits."""
    with ExitStack() as stack:
        peer_service, _ = _make_service(stack, "peer", tmp_path, log_client, auth=_ENFORCING_AUTH)
        request = _peer_handoff_request("fed-job", "parent-cluster", "alice@openathena.ai")
        with identity_scope(VerifiedIdentity("parent-cluster", FEDERATION_PEER_ROLE)):
            response = peer_service.launch_job(request, None)
        assert JobName.from_wire(response.job_id).name == "fed-job"


def test_inbound_handoff_rejects_a_submitter_outside_the_allowlist(tmp_path, log_client):
    """A verified peer cannot federate a submitter the receiving cluster does not admit."""
    with ExitStack() as stack:
        peer_service, _ = _make_service(stack, "peer", tmp_path, log_client, auth=_ENFORCING_AUTH)
        request = _peer_handoff_request("fed-job", "parent-cluster", "eve@gmail.com")
        with identity_scope(VerifiedIdentity("parent-cluster", FEDERATION_PEER_ROLE)):
            with pytest.raises(ConnectError) as exc:
                peer_service.launch_job(request, None)
        assert exc.value.code == Code.PERMISSION_DENIED


def test_inbound_handoff_rejects_a_requester_that_mismatches_the_peer(tmp_path, log_client):
    """The asserted requester must equal the authenticated peer — a peer cannot relay a
    handoff under another cluster's requester id."""
    with ExitStack() as stack:
        peer_service, _ = _make_service(stack, "peer", tmp_path, log_client, auth=_ENFORCING_AUTH)
        request = _peer_handoff_request("fed-job", "other-cluster", "alice@openathena.ai")
        with identity_scope(VerifiedIdentity("parent-cluster", FEDERATION_PEER_ROLE)):
            with pytest.raises(ConnectError) as exc:
                peer_service.launch_job(request, None)
        assert exc.value.code == Code.PERMISSION_DENIED


def test_inbound_handoff_rejects_a_non_peer_identity(tmp_path, log_client):
    """An ordinary authenticated user cannot forge a handoff by setting the field."""
    with ExitStack() as stack:
        peer_service, _ = _make_service(stack, "peer", tmp_path, log_client, auth=_ENFORCING_AUTH)
        request = _peer_handoff_request("fed-job", "parent-cluster", "alice@openathena.ai")
        with identity_scope(VerifiedIdentity("alice@openathena.ai", "user")):
            with pytest.raises(ConnectError) as exc:
                peer_service.launch_job(request, None)
        assert exc.value.code == Code.PERMISSION_DENIED


def test_enforcing_parent_refuses_to_federate_a_local_admin_submission(tmp_path, log_client):
    """With auth on, a local_admin (CIDR/loopback) submission is refused before handoff
    with a clear message — even to a peer whose policy would admit it — because a
    federated job must carry an authenticated user."""
    with ExitStack() as stack:
        parent_service, _ = _make_service(stack, "parent", tmp_path, log_client, auth=_ENFORCING_AUTH)
        peer_service, _ = _make_service(stack, "peer", tmp_path, log_client)
        _attach_federation(parent_service, _InProcessPeerConnection(peer_service))
        # No identity scope: an unauthenticated (CIDR/loopback) caller resolves to local_admin.
        with pytest.raises(ConnectError) as exc:
            parent_service.launch_job(_cluster_pinned_request("fed-job"), None)
        assert exc.value.code == Code.PERMISSION_DENIED


def test_a_child_job_a_peer_could_host_is_refused_as_unfederatable(tmp_path, log_client):
    """A sub-job dispatched from inside a running job never crosses to a peer.

    Its submitter is the worker running the parent, which authenticates by network location
    as local_admin. The refusal names the structural limit (INVALID_ARGUMENT), not the
    identity gate (PERMISSION_DENIED) that gates a root submission, and reaches no peer.
    """
    with ExitStack() as stack:
        parent_service, _ = _make_service(stack, "parent", tmp_path, log_client, auth=_ENFORCING_AUTH)
        peer_service, _ = _make_service(stack, "peer", tmp_path, log_client)
        connection = _InProcessPeerConnection(peer_service)
        _attach_federation(parent_service, connection)

        root = JobName.root(_USER, "root-job")
        with identity_scope(VerifiedIdentity(_USER, "admin")):
            parent_service.launch_job(make_direct_job_request("root-job"), None)

        child = _cluster_pinned_request("gpu-child")
        child.name = root.child("gpu-child").to_wire()
        with pytest.raises(ConnectError) as exc:
            parent_service.launch_job(child, None)

        assert exc.value.code == Code.INVALID_ARGUMENT
        assert connection.launch_calls == 0


def test_inbound_handoff_rejects_a_local_admin_submitter(tmp_path, log_client):
    """A local_admin (CIDR/loopback) identity is never a valid federation submitter,
    even for a verified peer — rejected regardless of the allowlist."""
    with ExitStack() as stack:
        peer_service, _ = _make_service(stack, "peer", tmp_path, log_client, auth=_ENFORCING_AUTH)
        request = _peer_handoff_request("fed-job", "parent-cluster", LOCAL_ADMIN_SUBMITTER)
        with identity_scope(VerifiedIdentity("parent-cluster", FEDERATION_PEER_ROLE)):
            with pytest.raises(ConnectError) as exc:
                peer_service.launch_job(request, None)
        assert exc.value.code == Code.PERMISSION_DENIED


def test_federation_sync_binds_the_requester_to_the_authenticated_peer(tmp_path, log_client):
    """A peer may sync only its own requester set; another requester's is denied and its
    own is authorized."""
    with ExitStack() as stack:
        peer_service, _ = _make_service(stack, "peer", tmp_path, log_client, auth=_ENFORCING_AUTH)
        with identity_scope(VerifiedIdentity("parent-cluster", FEDERATION_PEER_ROLE)):
            with pytest.raises(ConnectError) as exc:
                peer_service.federation_sync(
                    controller_pb2.Controller.FederationSyncRequest(requester_id="other-cluster"), None
                )
            assert exc.value.code == Code.PERMISSION_DENIED
            # Its own requester is authorized (an empty set, but not denied).
            peer_service.federation_sync(
                controller_pb2.Controller.FederationSyncRequest(requester_id="parent-cluster"), None
            )


def test_federation_sync_rejects_an_ordinary_user(tmp_path, log_client):
    """Only a federation peer (or admin) may sync — an ordinary user cannot read another
    requester's federated set."""
    with ExitStack() as stack:
        peer_service, _ = _make_service(stack, "peer", tmp_path, log_client, auth=_ENFORCING_AUTH)
        with identity_scope(VerifiedIdentity("alice", "user")):
            with pytest.raises(ConnectError) as exc:
                peer_service.federation_sync(
                    controller_pb2.Controller.FederationSyncRequest(requester_id="parent-cluster"), None
                )
        assert exc.value.code == Code.PERMISSION_DENIED


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

        # Before the first sync the parent knows the peer accepted the handoff but
        # has no mirrored tasks yet: PEER_STATUS_ASSIGNED.
        assert _peer_status_of(parent_service, job_id) == job_pb2.PEER_STATUS_ASSIGNED

        _run_peer_task_to_success(peer_state, job_id)
        manager.sync_once()

        # Parent's handle now mirrors the peer's terminal state and its task,
        # tagged with the owning peer; the posture advances to PEER_STATUS_SYNCED.
        assert query_job(parent_state, job_id).state == job_pb2.JOB_STATE_SUCCEEDED
        (mirrored,) = query_tasks_for_job(parent_state, job_id)
        assert mirrored.state == job_pb2.TASK_STATE_SUCCEEDED
        assert mirrored.cluster == "cw"
        assert _peer_status_of(parent_service, job_id) == job_pb2.PEER_STATUS_SYNCED


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


def test_sync_mirrors_submit_time_and_preemptions_faithfully(tmp_path, log_client):
    """The mirror carries the peer's real submit time and preemption counter: a
    not-yet-started federated task keeps the peer's submitted_at (not epoch 0 or
    the started_at fallback), and a preemption on the peer shows up in the
    mirrored row's counter."""
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, peer_state = _make_service(stack, "peer", tmp_path, log_client)
        manager = _attach_federation(parent_service, _InProcessPeerConnection(peer_service))

        request = _cluster_pinned_request("fed-timing")
        request.max_retries_preemption = 1
        response = parent_service.launch_job(request, None)
        job_id = JobName.from_wire(response.job_id)

        # Sync while the peer's task is still pending: no attempt has started,
        # yet the peer's submit time survives the mirror (not epoch 0).
        manager.sync_once()
        (peer_task,) = query_tasks_for_job(peer_state, job_id)
        (mirrored,) = query_tasks_for_job(parent_state, job_id)
        assert peer_task.submitted_at_ms.epoch_ms() > 0
        assert mirrored.submitted_at_ms == peer_task.submitted_at_ms
        assert mirrored.started_at_ms is None

        # One preemption (worker failure with budget to retry), then success, on
        # the peer; the mirrored row carries the peer's real counter, not 0.
        worker = register_worker(peer_state, "w1", "w1:8080", job_pb2.WorkerMetadata(hostname="w1"))
        dispatch_task(peer_state, peer_task, worker)
        transition_task(peer_state, peer_task.task_id, job_pb2.TASK_STATE_WORKER_FAILED)
        (peer_task,) = query_tasks_for_job(peer_state, job_id)
        # preemption_count is derived from the peer's attempt rows, not a stored column.
        assert query_task(peer_state, peer_task.task_id).preemption_count == 1
        dispatch_task(peer_state, peer_task, worker)
        transition_task(peer_state, peer_task.task_id, job_pb2.TASK_STATE_SUCCEEDED)
        manager.sync_once()

        (mirrored,) = query_tasks_for_job(parent_state, job_id)
        assert mirrored.state == job_pb2.TASK_STATE_SUCCEEDED
        # The parent derives the mirrored count from the mirrored attempt rows —
        # it matches the peer without any scalar on the sync wire.
        assert query_task(parent_state, mirrored.task_id).preemption_count == 1


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
        # Handed off but not yet reported on: the peer accepted the handoff but has
        # mirrored no tasks, so the posture is ASSIGNED and the pending reason names
        # the peer, not the local scheduler diagnostic (never sees a federated job).
        assert fed_status.peer_status == job_pb2.PEER_STATUS_ASSIGNED
        assert fed_status.pending_reason == "Handed off to peer cw; awaiting first status report"

        # A local job carries the reserved 'local' sentinel, no peer posture, and
        # keeps the local scheduler diagnostic.
        local_status = parent_service.get_job_status(
            controller_pb2.Controller.GetJobStatusRequest(job_id=local.job_id), None
        ).job
        assert local_status.cluster == LOCAL_CLUSTER
        assert local_status.peer_status == job_pb2.PEER_STATUS_NONE
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
    """While a handoff is undelivered (PENDING_HANDOFF), the status RPCs expose the
    pre-registration state: the posture says the peer has not accepted, the task
    count is the requested replica count (there are no task rows yet), and the
    pending reason says the job is awaiting the peer's acceptance — on both the
    detail path and the batch-loading list path."""
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, _ = _make_service(stack, "peer", tmp_path, log_client)
        _attach_federation(parent_service, _UnreachablePeerConnection(peer_service))

        response = parent_service.launch_job(_cluster_pinned_request("awaiting-ack", replicas=3), None)
        job_id = JobName.from_wire(response.job_id)
        assert _handle(parent_state, job_id).handoff_state == int(HandoffState.PENDING_HANDOFF)
        assert query_tasks_for_job(parent_state, job_id) == []

        status = parent_service.get_job_status(
            controller_pb2.Controller.GetJobStatusRequest(job_id=response.job_id), None
        ).job
        assert status.peer_status == job_pb2.PEER_STATUS_PENDING_SCHEDULING
        assert status.task_count == 3
        assert status.pending_reason == "Awaiting acceptance by peer cw"

        # The list path loads the handles in bulk and reports the same posture,
        # count, and reason.
        (listed,) = parent_service.list_jobs(
            controller_pb2.Controller.ListJobsRequest(query=controller_pb2.Controller.JobQuery(cluster="cw")),
            None,
        ).jobs
        assert listed.peer_status == job_pb2.PEER_STATUS_PENDING_SCHEDULING
        assert listed.task_count == 3
        assert listed.pending_reason == "Awaiting acceptance by peer cw"


@pytest.mark.parametrize(
    ("cluster", "handoff_state", "has_reported_tasks", "expected"),
    [
        # A local job is never a peer job, whatever the other inputs.
        (LOCAL_CLUSTER, None, False, job_pb2.PEER_STATUS_NONE),
        (LOCAL_CLUSTER, None, True, job_pb2.PEER_STATUS_NONE),
        # Handed off, peer has not acked yet, nothing mirrored.
        ("cw", int(HandoffState.PENDING_HANDOFF), False, job_pb2.PEER_STATUS_PENDING_SCHEDULING),
        # Peer acked, but no task set mirrored back yet.
        ("cw", int(HandoffState.HANDED_OFF), False, job_pb2.PEER_STATUS_ASSIGNED),
        # Peer acked and its task set is mirrored.
        ("cw", int(HandoffState.HANDED_OFF), True, job_pb2.PEER_STATUS_SYNCED),
        # Mirrored tasks win over a stale PENDING_HANDOFF handle: the sync loop can
        # mirror a job's state before a transient RPC failure lets mark_handed_off
        # run, so tasks-present is the more current signal (never "awaiting peer").
        ("cw", int(HandoffState.PENDING_HANDOFF), True, job_pb2.PEER_STATUS_SYNCED),
        # Peer rejected the handoff (id collision); terminal.
        ("cw", int(HandoffState.HANDOFF_REJECTED), False, job_pb2.PEER_STATUS_REJECTED),
        # A missing handle (can't-happen for a live federated job) reads as handed off.
        ("cw", None, False, job_pb2.PEER_STATUS_ASSIGNED),
    ],
)
def test_peer_status_derivation(cluster, handoff_state, has_reported_tasks, expected):
    """The full truth table of the PeerStatus derivation, including REJECTED and
    the mirrored-tasks-beat-a-stale-PENDING_HANDOFF ordering."""
    assert _peer_status(cluster, handoff_state, has_reported_tasks) == expected


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


@pytest.mark.parametrize("code", [Code.PERMISSION_DENIED, Code.INVALID_ARGUMENT])
def test_a_handoff_the_peer_refuses_fails_the_submission_and_stops_the_redrive(code, tmp_path, log_client):
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, _peer_state = _make_service(stack, "peer", tmp_path, log_client)
        connection = _RefusingPeerConnection(peer_service, code, "submitter not in allowlist")
        manager = _attach_federation(parent_service, connection)

        # The peer's verdict is the submitter's answer, not a job that silently never lands.
        with pytest.raises(ConnectError) as exc:
            parent_service.launch_job(_cluster_pinned_request("fed-job"), None)
        assert exc.value.code == code
        assert "submitter not in allowlist" in exc.value.message

        parent_job_id = JobName.root(_USER, "fed-job")
        handle = _handle(parent_state, parent_job_id)
        assert handle.handoff_state == int(HandoffState.HANDOFF_REJECTED)
        job = query_job(parent_state, parent_job_id)
        assert job.state == job_pb2.JOB_STATE_KILLED
        assert "submitter not in allowlist" in job.error

        # Terminalized, so the sync loop has nothing left to re-drive.
        manager.sync_once()
        assert connection.launch_calls == 1


def test_a_refused_handoff_does_not_propagate_out_of_the_redrive(tmp_path, log_client):
    # _deliver_handoff runs on the sync thread too, which dies on an uncaught
    # exception. A peer that starts refusing between delivery attempts must
    # terminalize the handle there, not take the whole sync loop down with it.
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, _peer_state = _make_service(stack, "peer", tmp_path, log_client)
        connection = _RefusingPeerConnection(peer_service, Code.UNAVAILABLE, "peer is booting")
        manager = _attach_federation(parent_service, connection)

        response = parent_service.launch_job(_cluster_pinned_request("fed-job"), None)
        parent_job_id = JobName.from_wire(response.job_id)
        assert _handle(parent_state, parent_job_id).handoff_state == int(HandoffState.PENDING_HANDOFF)

        connection.code = Code.PERMISSION_DENIED
        manager.sync_once()

        assert _handle(parent_state, parent_job_id).handoff_state == int(HandoffState.HANDOFF_REJECTED)
        assert query_job(parent_state, parent_job_id).state == job_pb2.JOB_STATE_KILLED


def test_a_handoff_the_peer_could_not_authenticate_stays_pending(tmp_path, log_client):
    # UNAUTHENTICATED is a key/clock/rollout transient — the federation bearer is
    # minted per request, so a later attempt can clear it. The handle stays pending
    # and the submission succeeds.
    with ExitStack() as stack:
        parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        peer_service, _peer_state = _make_service(stack, "peer", tmp_path, log_client)
        connection = _RefusingPeerConnection(peer_service, Code.UNAUTHENTICATED, "bad token")
        manager = _attach_federation(parent_service, connection)

        response = parent_service.launch_job(_cluster_pinned_request("fed-job"), None)
        parent_job_id = JobName.from_wire(response.job_id)
        assert _handle(parent_state, parent_job_id).handoff_state == int(HandoffState.PENDING_HANDOFF)

        manager.sync_once()
        assert connection.launch_calls == 2  # re-driven, still pending
        assert _handle(parent_state, parent_job_id).handoff_state == int(HandoffState.PENDING_HANDOFF)


# ---------------------------------------------------------------------------
# admission + incremental tombstone
# ---------------------------------------------------------------------------


def test_admit_persists_a_pending_handle_and_is_idempotent(tmp_path, log_client):
    with ExitStack() as stack:
        _parent_service, parent_state = _make_service(stack, "parent", tmp_path, log_client)
        store = ControllerFederationStore(parent_state._db)
        parent_job_id = JobName.root(_USER, "fed-job")
        spec = HandoffSpec(
            local_job_id=parent_job_id,
            peer_id="cw",
            owner_principal=_USER,
            submitting_user=_USER,
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
