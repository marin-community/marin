# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import time

import httpx
import jwt
import pytest
from google.protobuf.json_format import MessageToDict
from iris.cluster.backends.rpc.backend import RpcTaskBackend
from iris.cluster.bundle import BundleStore
from iris.cluster.config import AuthConfig, SharedAttemptConfig
from iris.cluster.controller.auth import create_controller_auth, request_auth_policy
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.shared_attempts import SHARED_GRANT_TOKEN_SECRET, AttemptCredentials, SharedAttemptClient
from iris.rpc import controller_pb2, job_pb2, worker_pb2
from iris.testing.controller import make_job_request, make_worker_metadata, worker_backend_descriptor
from iris.testing.journeys.backend import ScriptedObservation, ScriptedTaskBackend
from iris.version import client_revision_date
from rigging.grants import GrantDenied
from rigging.server_auth import ANONYMOUS_ADMIN, identity_scope
from rigging.shared_grants import SharedGrantConfig, SharedGrantUnavailable
from rigging.token_authority import generate_ed25519_keypair
from starlette.testclient import TestClient


class AttemptAuthority:
    """External broker boundary, recording authority rather than internal calls."""

    def __init__(self, *, now=time.time):
        self.active = {}
        self.unavailable = False
        self.denied = False
        self.now = now
        self.deadlines = {}

    def issue(self, *, grant_id, resource, task_id, attempt_uid):
        if self.unavailable:
            raise SharedGrantUnavailable("Synthetic broker outage")
        if self.denied:
            raise GrantDenied("Synthetic standing permission withdrawal")
        deadline = self.deadlines.setdefault(attempt_uid, self.now() + 30)
        if self.now() >= deadline:
            raise GrantDenied("The attempt grant has expired")
        self.active[attempt_uid] = (grant_id, resource, task_id)
        return AttemptCredentials(
            grant_id="child-" + attempt_uid,
            parent_grant_id=grant_id,
            actor="attempt-" + attempt_uid,
            attempt_uid=attempt_uid,
            access_token="synthetic-attempt-" + attempt_uid,
            expires_at=deadline,
            auth_url="https://auth.example",
        )

    def retire(self, *, grant_id, resource, task_id, attempt_uid):
        if self.unavailable:
            raise SharedGrantUnavailable("Synthetic broker outage")
        binding = self.active.pop(attempt_uid, None)
        if binding is not None:
            assert binding == (grant_id, resource, task_id)


class CredentialExecutionBackend(ScriptedTaskBackend):
    def __init__(self):
        super().__init__()
        self.launched_credentials = []
        self.lose_next_response = False

    def reconcile(self, request):
        for task in request.tasks_to_run:
            delivered = job_pb2.RunTaskRequest()
            delivered.CopyFrom(task)
            self.launched_credentials.append(delivered)
        observation = super().reconcile(request)
        if self.lose_next_response:
            self.lose_next_response = False
            raise ConnectionError("Synthetic lost response after the task started")
        return observation


class CredentialWorker:
    """Worker RPC boundary with explicit accepted launch requests."""

    def __init__(self):
        self.accepted = []

    def get_stub(self, address):
        return self

    def evict(self, address):
        pass

    def close(self):
        pass

    async def reconcile(self, request):
        observed = []
        for desired in request.desired:
            state = job_pb2.TASK_STATE_MISSING
            if desired.HasField("run") and desired.run.HasField("request"):
                accepted = job_pb2.RunTaskRequest()
                accepted.CopyFrom(desired.run.request)
                self.accepted.append(accepted)
                state = job_pb2.TASK_STATE_RUNNING
            observed.append(worker_pb2.Worker.AttemptObservation(attempt_uid=desired.attempt_uid, state=state))
        return worker_pb2.Worker.ReconcileResponse(
            worker_id="credential-worker", observed=observed, health=worker_pb2.Worker.WorkerHealth(healthy=True)
        )


@pytest.fixture
def shared_caller(monkeypatch):
    key = generate_ed25519_keypair()
    trust = SharedGrantConfig(
        issuer="https://auth.example", public_keys=(key.public_pem,), authority_url="https://auth.example"
    )
    auth = create_controller_auth(AuthConfig(shared_grants=trust), cluster_name="staging")
    now = int(time.time())
    token = jwt.encode(
        {
            "iss": trust.issuer,
            "aud": "iris",
            "sub": "session",
            "act": {"sub": "session"},
            "owner": "research",
            "grant_id": "submission-grant",
            "policy_revision": "one",
            "grant_version": 1,
            "permissions": [
                {
                    "service": "iris",
                    "actions": ["job.submit", "job.read", "job.cancel"],
                    "resource": "/clusters/staging/jobs/research/",
                }
            ],
            "iat": now,
            "exp": now + 300,
            "jti": "one",
        },
        key.private_pem,
        algorithm="EdDSA",
        headers={"typ": "at+jwt"},
    )

    def authorize(_url, **_kwargs):
        return httpx.Response(
            200,
            json={
                "grant_id": "submission-grant",
                "actor": "session",
                "owner": "research",
                "policy_revision": "one",
                "expires_at": now + 300,
            },
        )

    monkeypatch.setattr("rigging.shared_grants.requests.post", authorize)
    return auth, token


def _dashboard(controller, db, auth, log_client, tmp_path):
    service = ControllerServiceImpl(
        controller=controller,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=db,
        auth=auth,
        endpoint_service=EndpointServiceImpl(db=db),
    )
    return ControllerDashboard(service, auth_policy=request_auth_policy(auth))


@pytest.mark.parametrize("terminal_action", ["success", "cancel"])
def test_shared_submission_delivers_attempt_authority_and_retries_retirement_after_restart(
    make_controller, log_client, tmp_path, shared_caller, terminal_action
):
    auth, token = shared_caller
    broker = AttemptAuthority()
    auth.shared_attempts = broker
    backend = CredentialExecutionBackend()
    db_dir = tmp_path / "authority-db"
    db = ControllerDB(db_dir=db_dir)
    controller = make_controller(db=db, provider=backend, auth=auth)
    dashboard = _dashboard(controller, db, auth, log_client, tmp_path)
    prefix = "/iris.cluster.ControllerService/"
    request = make_job_request("/research/proof", task_image="registry.invalid/proof:synthetic")
    request.client_revision_date = client_revision_date()
    request.container_profile = job_pb2.CONTAINER_PROFILE_SANDBOX
    with TestClient(dashboard.app) as client:
        client.headers["Authorization"] = f"Bearer {token}"
        submitted = client.post(prefix + "LaunchJob", json=MessageToDict(request))
        assert submitted.status_code == 200, submitted.text
        controller.run_control_tick()
        assert len(backend.launched_credentials) == 1
        delivered = backend.launched_credentials[0]
        assert delivered.task_token == ""
        assert delivered.environment.secrets[SHARED_GRANT_TOKEN_SECRET] == "synthetic-attempt-" + delivered.attempt_uid
        assert broker.active == {
            delivered.attempt_uid: ("submission-grant", "/clusters/staging/jobs/research/proof/", "/research/proof/0")
        }
        broker.unavailable = True
        if terminal_action == "success":
            backend.queue_observation("/research/proof/0", ScriptedObservation(job_pb2.TASK_STATE_SUCCEEDED))
        else:
            canceled = client.post(prefix + "TerminateJob", json={"jobId": "/research/proof"})
            assert canceled.status_code == 200, canceled.text
        controller.run_control_tick()
        assert delivered.attempt_uid in broker.active
    controller.stop()
    replacement_db = ControllerDB(db_dir=db_dir)
    replacement = make_controller(db=replacement_db, provider=CredentialExecutionBackend(), auth=auth)
    broker.unavailable = False
    replacement.run_control_tick()
    assert broker.active == {}


def test_shared_job_without_attempt_broker_cannot_acquire_a_native_task_token(
    make_controller, log_client, tmp_path, shared_caller
):
    auth, token = shared_caller
    backend = CredentialExecutionBackend()
    db = ControllerDB(db_dir=tmp_path / "authority-db")
    controller = make_controller(db=db, provider=backend, auth=auth)
    dashboard = _dashboard(controller, db, auth, log_client, tmp_path)
    request = make_job_request("/research/proof", task_image="registry.invalid/proof:synthetic")
    request.client_revision_date = client_revision_date()
    request.container_profile = job_pb2.CONTAINER_PROFILE_SANDBOX
    with TestClient(dashboard.app) as client:
        submitted = client.post(
            "/iris.cluster.ControllerService/LaunchJob",
            json=MessageToDict(request),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert submitted.status_code == 200, submitted.text
    controller.run_control_tick()
    assert backend.launched_credentials == []
    with identity_scope(ANONYMOUS_ADMIN):
        assert controller.get_task_status("/research/proof/0").task.state == job_pb2.TASK_STATE_FAILED


@pytest.mark.parametrize("fault", ["unavailable", "denied", "expired"])
def test_shared_issuance_failure_preserves_unrelated_progress_and_recovers_or_fails_closed(
    make_controller, log_client, tmp_path, shared_caller, fault
):
    auth, token = shared_caller
    clock = [time.time()]
    broker = AttemptAuthority(now=lambda: clock[0])
    auth.shared_attempts = broker
    backend = CredentialExecutionBackend()
    db = ControllerDB(db_dir=tmp_path / "authority-db")
    controller = make_controller(db=db, provider=backend, auth=auth)
    with identity_scope(ANONYMOUS_ADMIN):
        controller.launch_job(make_job_request("/research/native", task_image="registry.invalid/native:synthetic"))
    controller.run_control_tick()
    backend.queue_observation("/research/native/0", ScriptedObservation(job_pb2.TASK_STATE_SUCCEEDED))
    dashboard = _dashboard(controller, db, auth, log_client, tmp_path)
    request = make_job_request("/research/shared", task_image="registry.invalid/shared:synthetic")
    request.client_revision_date = client_revision_date()
    request.container_profile = job_pb2.CONTAINER_PROFILE_SANDBOX
    with TestClient(dashboard.app) as client:
        submitted = client.post(
            "/iris.cluster.ControllerService/LaunchJob",
            json=MessageToDict(request),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert submitted.status_code == 200, submitted.text
    if fault == "expired":
        # The credential response landed, then dispatch failed before the backend accepted it.
        backend.fail_reconcile(times=1)
        with pytest.raises(ConnectionError):
            controller.run_control_tick()
        clock[0] += 61
    broker.unavailable = fault == "unavailable"
    broker.denied = fault == "denied"
    controller.run_control_tick()
    with identity_scope(ANONYMOUS_ADMIN):
        assert controller.get_job_status("/research/native").job.state == job_pb2.JOB_STATE_SUCCEEDED
        shared = controller.get_task_status("/research/shared/0").task
    shared_launches = [
        event for event in backend.events if event.kind == "launched" and event.task_id == request.name + "/0"
    ]
    assert shared_launches == []
    if fault != "unavailable":
        assert shared.state == job_pb2.TASK_STATE_FAILED
        assert broker.active == {}
        controller.run_control_tick()
        with identity_scope(ANONYMOUS_ADMIN):
            assert controller.get_task_status("/research/shared/0").task.state == job_pb2.TASK_STATE_FAILED
        return
    assert shared.state == job_pb2.TASK_STATE_ASSIGNED
    attempt_id = shared.attempts[0].attempt_id
    controller.stop()
    controller = make_controller(
        db=ControllerDB(db_dir=tmp_path / "authority-db"), provider=CredentialExecutionBackend(), auth=auth
    )
    broker.unavailable = False
    controller.run_control_tick()
    with identity_scope(ANONYMOUS_ADMIN):
        resumed = controller.get_task_status("/research/shared/0").task
    assert resumed.state == job_pb2.TASK_STATE_RUNNING
    assert [attempt.attempt_id for attempt in resumed.attempts] == [attempt_id]
    assert len(broker.active) == 1


@pytest.mark.parametrize("fault", ["unavailable", "denied"])
def test_worker_fleet_shared_issuance_failure_withholds_launch_and_recovers_or_stops(
    make_controller, log_client, tmp_path, shared_caller, fault
):
    auth, token = shared_caller
    broker = AttemptAuthority()
    broker.unavailable = fault == "unavailable"
    broker.denied = fault == "denied"
    auth.shared_attempts = broker
    worker = CredentialWorker()
    backend = RpcTaskBackend(descriptor=worker_backend_descriptor(), stub_factory=worker)
    db = ControllerDB(db_dir=tmp_path / "authority-db")
    controller = make_controller(db=db, provider=backend, auth=auth)
    with identity_scope(ANONYMOUS_ADMIN):
        registered = controller.register_worker(
            controller_pb2.Controller.RegisterRequest(
                worker_id="credential-worker", address="worker.invalid:1234", metadata=make_worker_metadata()
            )
        )
    assert registered.accepted
    dashboard = _dashboard(controller, db, auth, log_client, tmp_path)
    request = make_job_request("/research/shared", task_image="registry.invalid/shared:synthetic")
    request.client_revision_date = client_revision_date()
    request.container_profile = job_pb2.CONTAINER_PROFILE_SANDBOX
    with TestClient(dashboard.app) as client:
        submitted = client.post(
            "/iris.cluster.ControllerService/LaunchJob",
            json=MessageToDict(request),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert submitted.status_code == 200, submitted.text
    controller.run_control_tick()
    controller.run_control_tick()
    assert worker.accepted == []
    with identity_scope(ANONYMOUS_ADMIN):
        task = controller.get_task_status("/research/shared/0").task
    if fault == "denied":
        assert task.state == job_pb2.TASK_STATE_FAILED
        controller.run_control_tick()
        with identity_scope(ANONYMOUS_ADMIN):
            stopped = controller.get_task_status("/research/shared/0").task
        assert stopped.attempts[0].HasField("finished_at")
        assert worker.accepted == []
        return
    assert task.state == job_pb2.TASK_STATE_ASSIGNED
    broker.unavailable = False
    controller.run_control_tick()
    with identity_scope(ANONYMOUS_ADMIN):
        assert controller.get_task_status("/research/shared/0").task.state == job_pb2.TASK_STATE_RUNNING
    assert len(worker.accepted) == 1
    launched = worker.accepted[0]
    assert launched.task_token == ""
    assert launched.environment.secrets[SHARED_GRANT_TOKEN_SECRET] == "synthetic-attempt-" + launched.attempt_uid


@pytest.mark.parametrize("terminal_state", [job_pb2.TASK_STATE_SUCCEEDED, job_pb2.TASK_STATE_FAILED])
def test_shared_attempt_terminal_observation_during_issuance_outage_survives_lost_dispatch_response(
    make_controller, log_client, tmp_path, shared_caller, terminal_state
):
    auth, token = shared_caller
    broker = AttemptAuthority()
    auth.shared_attempts = broker
    backend = CredentialExecutionBackend()
    db = ControllerDB(db_dir=tmp_path / "authority-db")
    controller = make_controller(db=db, provider=backend, auth=auth)
    dashboard = _dashboard(controller, db, auth, log_client, tmp_path)
    request = make_job_request("/research/shared", task_image="registry.invalid/shared:synthetic")
    request.client_revision_date = client_revision_date()
    request.container_profile = job_pb2.CONTAINER_PROFILE_SANDBOX
    with TestClient(dashboard.app) as client:
        submitted = client.post(
            "/iris.cluster.ControllerService/LaunchJob",
            json=MessageToDict(request),
            headers={"Authorization": f"Bearer {token}"},
        )
        assert submitted.status_code == 200, submitted.text
    backend.lose_next_response = True
    with pytest.raises(ConnectionError):
        controller.run_control_tick()
    with identity_scope(ANONYMOUS_ADMIN):
        dispatched = controller.get_task_status("/research/shared/0").task
    assert dispatched.state == job_pb2.TASK_STATE_ASSIGNED
    assert len(backend.launched_credentials) == 1
    broker.unavailable = True
    backend.queue_observation("/research/shared/0", ScriptedObservation(terminal_state))
    controller.run_control_tick()
    with identity_scope(ANONYMOUS_ADMIN):
        completed = controller.get_task_status("/research/shared/0").task
    assert completed.state == terminal_state
    assert len(completed.attempts) == 1
    assert broker.active
    broker.unavailable = False
    controller.run_control_tick()
    assert broker.active == {}
    assert len(backend.launched_credentials) == 1
    with identity_scope(ANONYMOUS_ADMIN):
        assert controller.get_task_status("/research/shared/0").task.state == terminal_state


def test_missing_controller_identity_defers_issuance(tmp_path):
    trust = SharedGrantConfig(
        issuer="https://auth.example", public_keys=("synthetic-public-key",), authority_url="https://auth.example"
    )
    client = SharedAttemptClient(trust, SharedAttemptConfig(identity_token_file=tmp_path / "missing.token"))
    with pytest.raises(SharedGrantUnavailable, match="workload identity is unavailable"):
        client.issue(
            grant_id="parent",
            resource="/clusters/staging/jobs/proof/job/",
            task_id="/proof/job/0",
            attempt_uid="0123456789abcdef",
        )


@pytest.mark.parametrize("malformed_json", [False, True])
def test_malformed_attempt_credentials_never_expose_bearers_in_error_logs(tmp_path, monkeypatch, caplog, malformed_json):
    identity = tmp_path / "identity.token"
    identity.write_text("synthetic-controller-identity")
    secret = "synthetic-secret-that-must-not-appear-in-logs"
    trust = SharedGrantConfig(
        issuer="https://auth.example", public_keys=("synthetic-public-key",), authority_url="https://auth.example"
    )
    # The public-key verifier is unnecessary for this transport-only boundary.
    client = SharedAttemptClient(trust, SharedAttemptConfig(identity_token_file=identity))

    def response(*_args, **_kwargs):
        if malformed_json:
            return httpx.Response(200, content='{"access_token":"' + secret)
        return httpx.Response(200, json={"access_token": secret})

    monkeypatch.setattr("iris.cluster.controller.shared_attempts.requests.post", response)
    with pytest.raises(SharedGrantUnavailable):
        try:
            client.issue(
                grant_id="parent",
                resource="/clusters/staging/jobs/proof/job/",
                task_id="/proof/job/0",
                attempt_uid="0123456789abcdef",
            )
        except SharedGrantUnavailable:
            logging.getLogger(__name__).exception("Attempt dispatch remains unavailable")
            raise
    assert secret not in caplog.text
    assert identity.read_text() not in caplog.text
    credentials = AttemptCredentials(
        grant_id="child",
        parent_grant_id="parent",
        actor="attempt",
        attempt_uid="0123456789abcdef",
        access_token=secret,
        expires_at=time.time() + 30,
        auth_url="https://auth.example",
    )
    assert secret not in repr(credentials)
