# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regional allocation reads preserve resource shapes and attempt lifetimes."""

from contextlib import ExitStack

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.config import AuthConfig
from iris.cluster.controller import writes
from iris.cluster.controller.auth import FederationTokenProvider, create_controller_auth, request_auth_policy
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.types import JobName, gpu_device
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.async_adapter import AsyncServiceAdapter
from iris.rpc.auth import FEDERATION_PEER_ROLE, authorize_method
from iris.rpc.controller_connect import ControllerServiceASGIApplication
from iris.testing.controller import make_job_request
from iris.testing.federation import InProcessPeerConnection, attach_federation, make_service
from iris.testing.transitions import commit_dispatch_updates
from rigging.server_auth import PolicyAuthInterceptor, VerifiedIdentity, identity_scope
from rigging.timing import Timestamp
from rigging.token_authority import generate_ed25519_keypair
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def admin_identity():
    with identity_scope(VerifiedIdentity(user_id="operator", role="admin")):
        yield


def _job(service, name, gpus):
    request = make_job_request(name, max_retries_preemption=2)
    if gpus:
        request.resources.device.CopyFrom(gpu_device("H100", gpus))
    return JobName.from_wire(service.launch_job(request, None).job_id)


def _dispatch(state, job, attempt, band, at):
    with state._db.transaction() as tx:
        writes.promote_for_dispatch(tx, job.task(0), attempt, at, priority_band=band)


def _observe(state, job, attempt, phase, at):
    with state._db.transaction() as tx:
        commit_dispatch_updates(
            tx, [TaskUpdate(task_id=job.task(0), attempt_id=attempt, new_state=phase)], now=Timestamp.from_ms(at)
        )


def test_metadata_reads_mixed_children_with_applied_band_and_unstarted_attempt(controller_service, state):
    root = _job(controller_service, "mixed", 0)
    small = _job(controller_service, root.child("small").to_wire(), 2)
    large = _job(controller_service, root.child("large").to_wire(), 8)
    at = Timestamp.now().epoch_ms() + 100
    _dispatch(state, small, 0, job_pb2.PRIORITY_BAND_BATCH, at)
    _observe(state, small, 0, job_pb2.TASK_STATE_RUNNING, at + 100)
    _dispatch(state, large, 0, job_pb2.PRIORITY_BAND_INTERACTIVE, at)

    response = controller_service.get_gpu_allocation_metadata(
        controller_pb2.Controller.GetGpuAllocationMetadataRequest(
            from_ms=at, to_ms=at + 1000, root_job_ids=[root.to_wire()]
        ),
        None,
    )
    rows = {row.task_id: row for row in response.attempts}
    assert {task: row.gpu_count for task, row in rows.items()} == {
        root.task(0).to_wire(): 0,
        small.task(0).to_wire(): 2,
        large.task(0).to_wire(): 8,
    }
    assert rows[small.task(0).to_wire()].requested_priority == job_pb2.PRIORITY_BAND_INTERACTIVE
    assert rows[small.task(0).to_wire()].current_applied_priority == job_pb2.PRIORITY_BAND_BATCH
    assert rows[small.task(0).to_wire()].started_at_ms == at + 100
    assert rows[large.task(0).to_wire()].created_at_ms == at
    assert not rows[large.task(0).to_wire()].HasField("started_at_ms")
    assert not rows[root.task(0).to_wire()].HasField("created_at_ms")


def test_metadata_retains_old_attempt_lifetime_and_limits_without_truncation(controller_service, state):
    job = _job(controller_service, "retried", 8)
    at = Timestamp.now().epoch_ms() + 100
    _dispatch(state, job, 0, job_pb2.PRIORITY_BAND_BATCH, at)
    _observe(state, job, 0, job_pb2.TASK_STATE_RUNNING, at + 100)
    _observe(state, job, 0, job_pb2.TASK_STATE_PREEMPTED, at + 200)
    _dispatch(state, job, 1, job_pb2.PRIORITY_BAND_INTERACTIVE, at + 300)
    _observe(state, job, 1, job_pb2.TASK_STATE_RUNNING, at + 400)
    request = controller_pb2.Controller.GetGpuAllocationMetadataRequest(from_ms=at, to_ms=at + 1000)
    rows = {a.attempt_id: a for a in controller_service.get_gpu_allocation_metadata(request, None).attempts}
    assert rows[0].started_at_ms == at + 100
    assert rows[0].finished_at_ms == at + 200
    assert rows[0].current_attempt_id == 1
    assert rows[1].started_at_ms == at + 400
    assert not rows[1].HasField("finished_at_ms")
    request.max_rows = 1
    with pytest.raises(ConnectError) as error:
        controller_service.get_gpu_allocation_metadata(request, None)
    assert error.value.code == Code.RESOURCE_EXHAUSTED
    request.max_rows = 2
    request.from_ms = at + 250
    assert [a.attempt_id for a in controller_service.get_gpu_allocation_metadata(request, None).attempts] == [1]


def test_metadata_parent_reads_local_regional_jobs_absent_from_its_mirror(tmp_path, log_client):
    with ExitStack() as stack:
        parent, _ = make_service(stack, "parent", tmp_path, log_client)
        peer, state = make_service(stack, "peer", tmp_path, log_client)
        attach_federation(parent, InProcessPeerConnection(peer))
        job = _job(peer, "regional-only", 4)
        at = Timestamp.now().epoch_ms() + 100
        _dispatch(state, job, 0, job_pb2.PRIORITY_BAND_BATCH, at)
        response = parent.get_gpu_allocation_metadata(
            controller_pb2.Controller.GetGpuAllocationMetadataRequest(cluster="cw", from_ms=at, to_ms=at + 1000), None
        )
        assert [(row.task_id, row.gpu_count) for row in response.attempts] == [(job.task(0).to_wire(), 4)]


@pytest.mark.parametrize(
    "identity",
    [
        VerifiedIdentity(user_id="observer", role="dashboard"),
        VerifiedIdentity(user_id="worker", role="worker"),
        VerifiedIdentity(user_id="endpoint", role="admin", audience="/endpoint"),
    ],
)
def test_metadata_rejects_unprivileged_and_endpoint_scoped_readers(controller_service, identity):
    with identity_scope(identity), pytest.raises(ConnectError) as error:
        controller_service.get_gpu_allocation_metadata(
            controller_pb2.Controller.GetGpuAllocationMetadataRequest(from_ms=1, to_ms=2), None
        )
    assert error.value.code == Code.PERMISSION_DENIED


def test_metadata_peer_cannot_forward_a_read_to_another_peer(controller_service):
    identity = VerifiedIdentity(user_id="trusted-parent", role=FEDERATION_PEER_ROLE)
    with identity_scope(identity), pytest.raises(ConnectError) as error:
        controller_service.get_gpu_allocation_metadata(
            controller_pb2.Controller.GetGpuAllocationMetadataRequest(cluster="other", from_ms=1, to_ms=2), None
        )
    assert error.value.code == Code.PERMISSION_DENIED


def test_metadata_signed_peer_rpc_reads_regional_jobs_but_cannot_run_sql(controller_service):
    _job(controller_service, "regional-wire", 4)
    key = generate_ed25519_keypair()
    parent = create_controller_auth(AuthConfig(), cluster_name="parent", signing_key_pem=key.private_pem)
    regional = create_controller_auth(
        AuthConfig(trusted_cidrs=["10.0.0.0/8"], federation_peers={"parent": key.public_pem}),
        cluster_name="regional",
    )
    token = FederationTokenProvider("parent", parent.jwt_manager).get_token()
    interceptor = PolicyAuthInterceptor(request_auth_policy(regional), authorize=authorize_method)
    app = ControllerServiceASGIApplication(AsyncServiceAdapter(controller_service), interceptors=[interceptor])
    headers = {"Authorization": f"Bearer {token}", "Connect-Protocol-Version": "1"}
    at = Timestamp.now().epoch_ms()
    with TestClient(app) as client:
        response = client.post(
            "/iris.cluster.ControllerService/GetGpuAllocationMetadata",
            headers=headers,
            json={"fromMs": str(at - 60_000), "toMs": str(at + 1000)},
        )
        assert response.status_code == 200
        row = response.json()["attempts"][0]
        assert row["gpuCount"] == 4 and row["gpuVariant"] == "H100"
        assert row["currentAppliedPriority"] == "PRIORITY_BAND_INTERACTIVE"
        assert "createdAtMs" not in row
        forbidden = client.post(
            "/iris.cluster.ControllerService/ExecuteRawQuery", headers=headers, json={"sql": "SELECT * FROM job_config"}
        )
        assert forbidden.status_code == 403
