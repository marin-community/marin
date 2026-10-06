# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The per-job task token: who it names, which tasks get it, and what it may launch."""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.bundle import BundleStore
from iris.cluster.config import AuthConfig
from iris.cluster.controller.auth import ControllerAuth, create_controller_auth
from iris.cluster.controller.controller import with_task_token
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.types import JobName
from iris.rpc import job_pb2
from iris.testing.controller import make_job_request
from rigging.server_auth import VerifiedIdentity, identity_scope
from rigging.token_authority import generate_ed25519_keypair

_JOB = JobName.from_wire("/alice/train")
_ADMIN = VerifiedIdentity(user_id="operator", role="admin")
_ALICE_TASK = VerifiedIdentity(user_id="alice", role="task")


def _enforcing_auth() -> ControllerAuth:
    return create_controller_auth(
        AuthConfig(trusted_cidrs=["10.0.0.0/8"]),
        cluster_name="test-cluster",
        signing_key_pem=generate_ed25519_keypair().private_pem,
    )


def _run_request(profile: int = job_pb2.CONTAINER_PROFILE_DEFAULT) -> job_pb2.RunTaskRequest:
    return job_pb2.RunTaskRequest(task_id=_JOB.task(0).to_wire(), container_profile=profile)


def test_task_token_authenticates_as_the_job_owner_without_admin():
    auth = _enforcing_auth()
    assert auth.jwt_manager is not None

    stamped = with_task_token(_run_request(), _JOB, auth)

    identity = auth.jwt_manager.verify(stamped.task_token)
    assert (identity.user_id, identity.role) == ("alice", "task")


@pytest.mark.parametrize(
    ("profile", "auth"),
    [
        pytest.param(job_pb2.CONTAINER_PROFILE_SANDBOX, _enforcing_auth(), id="sandbox"),
        pytest.param(
            job_pb2.CONTAINER_PROFILE_DEFAULT,
            create_controller_auth(None, cluster_name="test-cluster", signing_key_pem=None),
            id="null-auth",
        ),
    ],
)
def test_no_task_token_for_sandbox_or_null_auth(profile, auth):
    assert with_task_token(_run_request(profile), _JOB, auth).task_token == ""


@pytest.fixture
def auth_service(state, mock_controller, tmp_path, log_client):
    return ControllerServiceImpl(
        controller=mock_controller,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        auth=ControllerAuth(provider="static"),
        endpoint_service=EndpointServiceImpl(db=state._db),
    )


def _launch(service, name: str, *, profile: int = 0, band: int = 0):
    request = make_job_request(name, priority_band=band)
    request.container_profile = profile
    return service.launch_job(request, None)


@pytest.mark.parametrize(
    ("parent_profile", "allowed"),
    [
        pytest.param(job_pb2.CONTAINER_PROFILE_PRIVILEGED, True, id="parent-privileged"),
        pytest.param(job_pb2.CONTAINER_PROFILE_DEFAULT, False, id="parent-default"),
    ],
)
def test_task_launches_an_elevated_child_only_when_its_parent_holds_that_profile(
    auth_service, parent_profile, allowed
):
    with identity_scope(_ADMIN):
        _launch(auth_service, "/alice/parent", profile=parent_profile)

    with identity_scope(_ALICE_TASK):
        if allowed:
            _launch(auth_service, "/alice/parent/child", profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED)
            return
        with pytest.raises(ConnectError) as exc:
            _launch(auth_service, "/alice/parent/child", profile=job_pb2.CONTAINER_PROFILE_PRIVILEGED)
    assert exc.value.code == Code.PERMISSION_DENIED


@pytest.mark.parametrize(
    ("parent_band", "allowed"),
    [
        pytest.param(job_pb2.PRIORITY_BAND_PRODUCTION, True, id="parent-production"),
        pytest.param(job_pb2.PRIORITY_BAND_INTERACTIVE, False, id="parent-interactive"),
    ],
)
def test_task_launches_a_production_child_only_under_a_production_parent(auth_service, parent_band, allowed):
    with identity_scope(_ADMIN):
        _launch(auth_service, "/alice/parent", band=parent_band)

    with identity_scope(_ALICE_TASK):
        if allowed:
            _launch(auth_service, "/alice/parent/child", band=job_pb2.PRIORITY_BAND_PRODUCTION)
            return
        with pytest.raises(ConnectError) as exc:
            _launch(auth_service, "/alice/parent/child", band=job_pb2.PRIORITY_BAND_PRODUCTION)
    assert exc.value.code == Code.PERMISSION_DENIED


def test_task_cannot_launch_under_another_users_job(auth_service):
    with identity_scope(_ADMIN):
        _launch(auth_service, "/bob/parent")

    with identity_scope(_ALICE_TASK), pytest.raises(ConnectError) as exc:
        _launch(auth_service, "/bob/parent/child")
    assert exc.value.code == Code.PERMISSION_DENIED
