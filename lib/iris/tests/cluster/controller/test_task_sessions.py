# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Session authority survives restart and fences rollover, revocation and late provider replies."""

import dataclasses
import io
import ipaddress
import json
import subprocess
import sys
from pathlib import Path

import pytest
import uvicorn
from connectrpc.errors import ConnectError
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID
from finelog.client import LogClient
from finelog.rpc import finelog_stats_pb2, logging_pb2
from iris.cluster import credential_process
from iris.cluster.backends.k8s.tasks import K8sTaskProvider
from iris.cluster.bundle import BundleStore
from iris.cluster.config import AuthConfig, CredentialBrokerConfig, StorageCredentialProfile
from iris.cluster.controller import writes
from iris.cluster.controller.auth import create_controller_auth
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.projections.endpoints import EndpointRow, EndpointsProjection
from iris.cluster.controller.schema import jobs_table, task_attempts_table, tasks_table
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.task_sessions import TaskSessionBroker
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.types import K8sResource
from iris.cluster.runtime.env import TASK_OUTPUT_FINALIZING_STATUS
from iris.cluster.task_credentials import task_credentials
from iris.cluster.types import JobName
from iris.managed_thread import ThreadContainer
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.auth import authorize_resource_owner
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.testing.controller import make_job_request
from iris.testing.controller_state import submit_job_in_tx
from iris.testing.k8s import k8s_backend_descriptor, pod_config
from rigging.server_auth import JwtAuthenticator, RequestAuthPolicy, VerifiedIdentity, identity_scope
from rigging.timing import Duration, ExponentialBackoff, Timestamp
from sqlalchemy import update
from starlette.applications import Starlette
from starlette.responses import Response
from starlette.routing import Mount, Route


@pytest.fixture
def session_setup(state):
    job = JobName.root("training", "session-test")
    with state._db.transaction() as tx:
        submit_job_in_tx(
            tx, job_id=job, request=make_job_request(job.to_wire()), submitting_user="submitter@example.test"
        )
        uid = writes.insert_attempt(
            tx,
            task_id=job.task(0),
            attempt_id=0,
            worker_id=None,
            state=job_pb2.TASK_STATE_ASSIGNED,
            created_at_ms=Timestamp.now().epoch_ms(),
        )
        tx.execute(update(tasks_table).where(tasks_table.c.task_id == job.task(0)).values(current_attempt_id=0))
    config = CredentialBrokerConfig(
        capability_key="capability-key-that-is-at-least-32-bytes",
        tls_secret="controller-tls",
        profiles={
            "training": StorageCredentialProfile(
                api_token="provider-root",
                allowed_principals=["submitter@example.test"],
                endpoint="https://cwobject.com",
                region="US-EAST-02A",
                buckets=["training"],
            )
        },
    )
    auth = create_controller_auth(AuthConfig(required=True), cluster_name="test-cluster")
    assert auth.jwt_manager is not None
    broker = TaskSessionBroker(state._db, config, auth.jwt_manager, 300)
    published = []
    broker.attach_projection(lambda payload: published.append(json.loads(payload)))
    dispatch = broker.prepare([job_pb2.RunTaskRequest(task_id=job.task(0).to_wire(), attempt_id=0, attempt_uid=uid)])[0]
    assert dispatch is not None
    broker.publish()
    renew = controller_pb2.Controller.RenewTaskSessionRequest(
        attempt_uid=uid,
        refresh_capability=dispatch.task_refresh_capability,
    )
    return broker, auth.jwt_manager, renew, job, published, config


def test_session_uses_submitter_and_separate_namespace_grant(session_setup):
    broker, jwt, renew, job, _, _ = session_setup
    response = broker.renew(renew)
    identity = jwt.verify(response.token)
    assert identity.user_id == "submitter@example.test"
    assert identity.owner_id == "training"
    with identity_scope(identity):
        assert authorize_resource_owner("training") == identity
    broker.authorize_request(
        identity, "LaunchJob", controller_pb2.Controller.LaunchJobRequest(name=job.child("child").to_wire())
    )
    with pytest.raises(ConnectError, match="root job"):
        broker.authorize_request(
            identity, "LaunchJob", controller_pb2.Controller.LaunchJobRequest(name="/training/other/child")
        )
    with pytest.raises(ConnectError):
        broker.authorize_request(dataclasses.replace(identity, owner_id="other"), "GetCurrentUser", job_pb2.Empty())


@pytest.mark.parametrize(
    ("method", "build_request"),
    [
        ("LaunchJob", lambda j: controller_pb2.Controller.LaunchJobRequest(name=j.child("child").to_wire())),
        ("GetJobStatus", lambda j: controller_pb2.Controller.GetJobStatusRequest(job_id=j.to_wire())),
        ("GetJobState", lambda j: controller_pb2.Controller.GetJobStateRequest(job_ids=[j.to_wire()])),
        ("TerminateJob", lambda j: controller_pb2.Controller.TerminateJobRequest(job_id=j.to_wire())),
        ("CompleteJob", lambda j: controller_pb2.Controller.CompleteJobRequest(job_id=j.to_wire())),
        ("ListTasks", lambda j: controller_pb2.Controller.ListTasksRequest(job_id=j.to_wire())),
        ("GetTaskStatus", lambda j: controller_pb2.Controller.GetTaskStatusRequest(task_id=j.task(0).to_wire())),
        ("ExecInContainer", lambda j: controller_pb2.Controller.ExecInContainerRequest(task_id=j.task(0).to_wire())),
        ("GetProcessStatus", lambda j: job_pb2.GetProcessStatusRequest(target=j.task(0).to_wire())),
        ("ProfileTask", lambda j: job_pb2.ProfileTaskRequest(target=j.task(0).to_wire())),
        ("KickTasks", lambda j: controller_pb2.Controller.KickTasksRequest(targets=[j.task(0).to_wire()])),
        (
            "RegisterEndpoint",
            lambda j: controller_pb2.Controller.RegisterEndpointRequest(
                name="/serve/alias", task_id=j.task(0).to_wire()
            ),
        ),
        ("UnregisterEndpoint", lambda j: controller_pb2.Controller.UnregisterEndpointRequest(endpoint_id=j.name)),
        (
            "MintEndpointToken",
            lambda j: controller_pb2.Controller.MintEndpointTokenRequest(endpoint_name="/serve/" + j.name),
        ),
        (
            "ListJobs",
            lambda j: controller_pb2.Controller.ListJobsRequest(
                query=controller_pb2.Controller.JobQuery(job_id_prefix=j.to_wire() + "/")
            ),
        ),
    ],
)
def test_task_rpc_resource_scope_accepts_own_root_and_rejects_other_roots(session_setup, state, method, build_request):
    broker, jwt, renew, job, _, _ = session_setup
    other = JobName.root(job.user, "foreign")
    with state._db.transaction() as tx:
        submit_job_in_tx(tx, job_id=other, request=make_job_request(other.to_wire()))
        for owner in (job, other):
            tx.caches[EndpointsProjection].add(
                tx,
                EndpointRow(
                    endpoint_id=owner.name,
                    name="/serve/" + owner.name,
                    address="127.0.0.1:8000",
                    task_id=owner.task(0),
                    metadata={},
                    registered_at=Timestamp.now(),
                ),
            )
    identity = jwt.verify(broker.renew(renew).token)
    # Both jobs share an owner, so only the session's root boundary distinguishes them.
    broker.authorize_request(identity, method, build_request(job))
    with pytest.raises(ConnectError, match="root job"):
        broker.authorize_request(identity, method, build_request(other))


def test_non_broker_task_tokens_keep_namespace_authority(session_setup):
    _, jwt, _, job, _, _ = session_setup
    identity = jwt.verify(jwt.create_task_token(job))
    with identity_scope(identity):
        assert authorize_resource_owner(job.user) == identity


def test_broker_rejects_anonymous_launch_and_fails_ineligible_dispatch(
    session_setup,
    state,
    make_controller,
    log_client,
    mock_controller,
    tmp_path,
):
    broker, _, _, job, _, config = session_setup
    service = ControllerServiceImpl(
        controller=mock_controller,
        bundle_store=BundleStore(str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoint_service=EndpointServiceImpl(db=state._db),
        task_sessions=broker,
    )
    with (
        identity_scope(VerifiedIdentity("anonymous", "admin")),
        pytest.raises(ConnectError, match="authenticated submitting user"),
    ):
        service.launch_job(make_job_request("/training/anonymous"), None)
    # A pre-cutover record must finish visibly, rather than remain ASSIGNED with no Pod.
    with state._db.transaction() as tx:
        tx.execute(update(jobs_table).where(jobs_table.c.job_id == job).values(submitting_user="local_admin"))
    kubernetes = InMemoryK8sService(namespace="iris")
    provider = K8sTaskProvider(
        descriptor=k8s_backend_descriptor(),
        kubectl=kubernetes,
        pods=pod_config(),
        cluster_scan_interval=0.0,
    )
    controller = make_controller(
        db=state._db,
        provider=provider,
        credential_broker=config,
        auth=create_controller_auth(AuthConfig(required=True), cluster_name="test"),
    )
    controller.run_control_tick()
    status = controller.get_job_status(job.to_wire()).job
    assert status.state == job_pb2.JOB_STATE_FAILED
    assert "Credential broker denied dispatch" in status.error
    assert not kubernetes.list_json(K8sResource.PODS)


def test_rollover_and_disable_fence_cached_tokens_and_renewal_across_restart(session_setup, state):
    broker, jwt, renew, _, published, config = session_setup
    old_identity = jwt.verify(broker.renew(renew).token)
    management = controller_pb2.Controller.ManageTaskSessionsRequest
    with identity_scope(VerifiedIdentity("operator", "admin")):
        broker.manage(management(attempt_uid=renew.attempt_uid, action=management.ROLLOVER))
    with pytest.raises(ConnectError, match="rolled over"):
        broker.authorize_request(old_identity, "GetCurrentUser", job_pb2.Empty())
    renewed = jwt.verify(broker.renew(renew).token)
    assert renewed.session_generation > old_identity.session_generation
    with identity_scope(VerifiedIdentity("operator", "admin")):
        broker.manage(management(principal=renewed.user_id, action=management.DISABLE))
    assert renew.attempt_uid not in published[-1]
    restarted = TaskSessionBroker(state._db, config, jwt, 300)
    with pytest.raises(ConnectError, match="disabled"):
        restarted.renew(renew)


def test_superseded_and_closed_attempts_cannot_renew(session_setup, state):
    broker, _, renew, job, _, _ = session_setup
    with state._db.transaction() as tx:
        tx.execute(
            update(task_attempts_table)
            .where(task_attempts_table.c.attempt_uid == renew.attempt_uid)
            .values(finished_at_ms=Timestamp.from_seconds(Timestamp.now().epoch_seconds() - 301))
        )
    with pytest.raises(ConnectError, match="closed"):
        broker.renew(renew)
    with state._db.transaction() as tx:
        tx.execute(
            update(task_attempts_table)
            .where(task_attempts_table.c.attempt_uid == renew.attempt_uid)
            .values(finished_at_ms=None)
        )
        tx.execute(update(tasks_table).where(tasks_table.c.task_id == job.task(0)).values(current_attempt_id=1))
    with pytest.raises(ConnectError, match="superseded"):
        broker.renew(renew)


def test_refresh_capability_cannot_authenticate_another_attempt(session_setup):
    broker, _, renew, _, _, _ = session_setup
    forged = controller_pb2.Controller.RenewTaskSessionRequest(
        attempt_uid="another-attempt",
        refresh_capability=renew.refresh_capability,
    )
    with pytest.raises(ConnectError, match="Invalid"):
        broker.renew(forged)


def test_provider_exchange_racing_disable_never_releases_credentials(session_setup, monkeypatch):
    broker, jwt, renew, _, _, _ = session_setup
    identity = jwt.verify(broker.renew(renew).token)
    management = controller_pb2.Controller.ManageTaskSessionsRequest

    def provider_response(request, *, timeout):
        assert request.full_url == "https://api.coreweave.com/v1/cwobject/temporary-credentials/api-token"
        assert request.get_header("Authorization") == "Bearer provider-root"
        with identity_scope(VerifiedIdentity("operator", "admin")):
            broker.manage(management(attempt_uid=renew.attempt_uid, action=management.DISABLE))
        expiry = Timestamp.from_seconds(Timestamp.now().epoch_seconds() + 1800).as_formatted_date()
        return io.BytesIO(
            json.dumps(
                {
                    "AccessKeyId": "temporary-access",
                    "SecretAccessKey": "temporary-secret",
                    "Token": "session-token",
                    "Expiration": expiry,
                }
            ).encode()
        )

    monkeypatch.setattr("iris.cluster.controller.storage_credentials.urlopen", provider_response)
    with identity_scope(identity), pytest.raises(ConnectError, match="disabled"):
        broker.credentials(controller_pb2.Controller.GetTaskCredentialsRequest(profile="training"))


def test_finalization_keeps_storage_authority_but_closes_mutating_access(session_setup, state):
    broker, jwt, renew, job, published, _ = session_setup
    with state._db.transaction() as tx:
        tx.execute(
            update(tasks_table)
            .where(tasks_table.c.task_id == job.task(0))
            .values(status_message=TASK_OUTPUT_FINALIZING_STATUS)
        )
    broker.publish()
    identity = jwt.verify(broker.renew(renew).token)
    broker.authorize_request(identity, "GetTaskCredentials", controller_pb2.Controller.GetTaskCredentialsRequest())
    assert published[-1][renew.attempt_uid]["finalizing"]
    with pytest.raises(ConnectError, match="Finalizing"):
        broker.authorize_request(
            identity, "LaunchJob", controller_pb2.Controller.LaunchJobRequest(name=job.child("late").to_wire())
        )
    with state._db.transaction() as tx:
        tx.execute(
            update(task_attempts_table)
            .where(task_attempts_table.c.attempt_uid == renew.attempt_uid)
            .values(finished_at_ms=Timestamp.now())
        )
    with pytest.raises(ConnectError, match="closed"):
        broker.renew(renew)


def test_task_client_verifies_private_tls_and_recovers_from_access_token_rollover(
    session_setup,
    state,
    log_client,
    mock_controller,
    tmp_path,
    monkeypatch,
):
    broker, jwt, renew, _, _, _ = session_setup
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-controller")])
    now = Timestamp.now().epoch_seconds()
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(Timestamp.from_seconds(now - 60).as_naive_utc())
        .not_valid_after(Timestamp.from_seconds(now + 3600).as_naive_utc())
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
        .issuer_name(name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(Timestamp.from_seconds(now - 60).as_naive_utc())
        .not_valid_after(Timestamp.from_seconds(now + 3600).as_naive_utc())
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
        .sign(key, hashes.SHA256())
    )
    ca_file, key_file = tmp_path / "ca.pem", tmp_path / "key.pem"
    leaf_file = tmp_path / "leaf.pem"
    ca_file.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    leaf_file.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        leaf_key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
    )
    service = ControllerServiceImpl(
        controller=mock_controller,
        bundle_store=BundleStore(str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        endpoint_service=EndpointServiceImpl(db=state._db),
        task_sessions=broker,
    )
    policy = RequestAuthPolicy(authenticators=(JwtAuthenticator(jwt),), verifier=jwt)
    dashboard = ControllerDashboard(service, auth_policy=policy, task_sessions=broker)

    async def log_response(request):
        assert request.headers["authorization"].startswith("Bearer ")
        message = (
            logging_pb2.FetchLogsResponse(cursor=42)
            if request.url.path.endswith("FetchLogs")
            else finelog_stats_pb2.ListNamespacesResponse()
        )
        return Response(message.SerializeToString(), media_type="application/proto")

    app = Starlette(
        routes=[
            Route("/finelog.stats.StatsService/ListNamespaces", log_response, methods=["POST"]),
            Route("/finelog.logging.LogService/FetchLogs", log_response, methods=["POST"]),
            Mount("/", app=dashboard.app),
        ]
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            log_level="warning",
            log_config=None,
            ssl_certfile=str(leaf_file),
            ssl_keyfile=str(key_file),
        )
    )
    threads = ThreadContainer()
    try:
        threads.spawn_server(server, name="task-credential-tls")
        ExponentialBackoff(initial=0.01, maximum=0.1).wait_until(
            lambda: server.started, timeout=Duration.from_seconds(5)
        )
        assert server.servers
        port = server.servers[0].sockets[0].getsockname()[1]
        address = f"https://127.0.0.1:{port}"
        private_file = tmp_path / "session.json"
        private_file.write_text(
            json.dumps(
                {
                    "controller": address,
                    "attempt_uid": renew.attempt_uid,
                    "refresh_capability": renew.refresh_capability,
                    "ca_file": str(ca_file),
                    "profiles": {
                        "training": {
                            "region": "US-EAST-02A",
                            "endpoint": "https://cwobject.com",
                            "addressing_style": "virtual",
                        }
                    },
                }
            )
        )
        credentials = task_credentials(str(private_file))
        with ControllerServiceClientSync(
            address, interceptors=credentials.interceptors(), http_client=credentials.http_client()
        ) as client:
            first = client.get_current_user(job_pb2.GetCurrentUserRequest())
            assert first.user_id == "submitter@example.test"
            management = controller_pb2.Controller.ManageTaskSessionsRequest
            with identity_scope(VerifiedIdentity("operator", "admin")):
                broker.manage(management(attempt_uid=renew.attempt_uid, action=management.ROLLOVER))
            assert client.get_current_user(job_pb2.GetCurrentUserRequest()).user_id == first.user_id
        logs = LogClient.connect(address, interceptors=credentials.interceptors(), ca_bundle=ca_file.read_bytes())
        try:
            assert logs.list_namespaces() == []
            assert logs.fetch_logs(logging_pb2.FetchLogsRequest()).cursor == 42
        finally:
            logs.close()
        exchanges = []
        expiry = Timestamp.from_seconds(Timestamp.now().epoch_seconds() + 1800)

        def provider_response(request, *, timeout):
            exchanges.append(request.full_url)
            return io.BytesIO(
                json.dumps(
                    {
                        "AccessKeyId": "temporary-access",
                        "SecretAccessKey": "temporary-secret",
                        "Token": "session-token",
                        "Expiration": expiry.as_formatted_date(),
                    }
                ).encode()
            )

        monkeypatch.setattr("iris.cluster.controller.storage_credentials.urlopen", provider_response)
        # -I -S removes the checkout and site-packages: the same mounted script
        # must work in a user environment without Iris or any third-party library.
        helper = Path(credential_process.__file__)
        for _ in range(2):
            process = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    str(helper),
                    "--file",
                    str(private_file),
                    "--profile",
                    "training",
                    "--cache-dir",
                    str(tmp_path / "credentials"),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(process.stdout)
            assert result["Version"] == 1
            assert result["AccessKeyId"] == "temporary-access"
            assert result["SessionToken"] == "session-token"
        assert len(exchanges) == 1
        with ControllerServiceClientSync(address, timeout_ms=1000) as untrusted:
            with pytest.raises(ConnectError):
                untrusted.renew_task_session(renew)
    finally:
        threads.stop()
