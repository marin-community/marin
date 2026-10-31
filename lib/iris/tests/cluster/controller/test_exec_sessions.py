# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grant issuance crosses real controller persistence and the K8s adapter."""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.bundle import BundleStore
from iris.cluster.config import ExecGatewayConfig
from iris.cluster.controller.auth import ControllerAuth, JwtTokenManager
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.reconcile import dispatch
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.exec_gateway.auth import exec_audience, verify_exec_grant
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.types import K8sResource
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.auth import FEDERATION_PEER_ROLE
from iris.testing.controller import MockController, make_controller_state, make_direct_job_request, make_test_entrypoint
from iris.testing.k8s import make_batch, make_kueue_provider
from iris.testing.transitions import commit_dispatch_updates
from rigging.server_auth import VerifiedIdentity, identity_scope
from rigging.timing import Timestamp
from rigging.token_authority import JwksVerifier, JwtSigner, generate_ed25519_keypair, signing_key_from_private_pem


@pytest.fixture
def execution(log_client, tmp_path):
    key = generate_ed25519_keypair()
    signer = JwtSigner(signing_key_from_private_pem(key.private_pem), issuer="controller")
    verifier = JwksVerifier(
        issuers={"controller": [key.public_pem]},
        expected_audiences=frozenset({exec_audience("gateway")}),
        leeway_seconds=0,
    )
    manager = JwtTokenManager(signer, verifier)
    k8s = InMemoryK8sService()
    provider = make_kueue_provider(k8s)
    runtime = MockController()
    runtime.backend = provider
    with make_controller_state() as state:
        service = ControllerServiceImpl(
            controller=runtime,
            bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
            log_client=log_client,
            db=state._db,
            endpoint_service=EndpointServiceImpl(db=state._db),
            auth=ControllerAuth(provider="static", jwt_manager=manager),
            exec_gateway=ExecGatewayConfig(address="https://exec.test", gateway_id="gateway"),
        )
        request = make_direct_job_request("exec-session")
        request.ClearField("resources")
        request.ClearField("entrypoint")
        request.environment.primary_service = "agent"
        for name in ("agent", "api"):
            member = request.environment.services.add(
                name=name,
                image="python:3.12",
                workdir="/work",
                resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            )
            member.runtime_entrypoint.CopyFrom(make_test_entrypoint())
        with identity_scope(VerifiedIdentity(user_id="test-user", role="user")):
            launched = service.launch_job(request, None)
        with state._db.transaction() as tx:
            batch = dispatch.drain_for_dispatch(tx)
        provider.sync(make_batch(tasks_to_run=list(batch.tasks_to_run)))
        [pod] = k8s.list_json(K8sResource.PODS)
        pod["metadata"]["uid"] = "actual-pod-uid"
        pod["status"] = {
            "phase": "Running",
            "containerStatuses": [
                {"name": container["name"], "state": {"running": {}}} for container in pod["spec"]["containers"]
            ],
        }
        k8s.seed_resource(K8sResource.PODS, pod["metadata"]["name"], pod)
        task = JobName.from_wire(launched.job_id).task(0)
        with state._db.transaction() as tx:
            commit_dispatch_updates(
                tx, [TaskUpdate(task_id=task, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING)], now=Timestamp.now()
            )
        try:
            yield service, verifier, task.to_wire(), batch.tasks_to_run[0].attempt_uid, k8s, pod
        finally:
            provider.close()


@pytest.mark.parametrize(("requested", "physical"), [("agent", "task"), ("task", "task"), ("api", "service-api")])
def test_open_exec_session_binds_actual_pod_attempt_and_canonical_container(execution, requested, physical):
    service, verifier, task, uid, _, pod = execution
    request = controller_pb2.Controller.OpenExecSessionRequest(
        task_id=task, attempt_uid=uid, container_name=requested, command=["python", "-u"]
    )
    with identity_scope(VerifiedIdentity(user_id="test-user", role="user")):
        response = service.open_exec_session(request, None)
    grant = verify_exec_grant(verifier, response.grant)
    assert response.gateway_address == "https://exec.test"
    assert response.attempt_uid == uid
    assert grant.target.pod_name == pod["metadata"]["name"]
    assert grant.target.pod_uid == "actual-pod-uid"
    assert grant.target.attempt_uid == uid
    assert grant.target.container_name == physical
    assert grant.command == ("python", "-u")


@pytest.mark.parametrize(
    ("user", "role", "container", "attempt", "code"),
    [
        ("mallory", "user", "api", "", Code.PERMISSION_DENIED),
        ("peer", FEDERATION_PEER_ROLE, "api", "", Code.PERMISSION_DENIED),
        ("test-user", "user", "logship", "", Code.INVALID_ARGUMENT),
        ("test-user", "user", "api", "stale-attempt", Code.FAILED_PRECONDITION),
    ],
)
def test_open_exec_session_rejects_other_owners_peers_containers_and_attempts(
    execution, user, role, container, attempt, code
):
    service, _, task, _, _, _ = execution
    request = controller_pb2.Controller.OpenExecSessionRequest(
        task_id=task, attempt_uid=attempt, container_name=container, command=["python"]
    )
    with identity_scope(VerifiedIdentity(user_id=user, role=role)), pytest.raises(ConnectError) as rejected:
        service.open_exec_session(request, None)
    assert rejected.value.code == code


@pytest.mark.parametrize("change", ["attempt", "terminating", "container"])
def test_open_exec_session_rejects_stale_or_stopped_pod(execution, change):
    service, _, task, _, k8s, pod = execution
    if change == "attempt":
        pod["metadata"]["labels"]["iris.attempt_uid"] = "replacement"
    elif change == "terminating":
        pod["metadata"]["deletionTimestamp"] = "2026-01-01T00:00:00Z"
    else:
        pod["status"]["containerStatuses"] = []
    k8s.seed_resource(K8sResource.PODS, pod["metadata"]["name"], pod)
    request = controller_pb2.Controller.OpenExecSessionRequest(task_id=task, container_name="api", command=["python"])
    with identity_scope(VerifiedIdentity(user_id="test-user", role="user")), pytest.raises(ConnectError) as rejected:
        service.open_exec_session(request, None)
    assert rejected.value.code == Code.FAILED_PRECONDITION
