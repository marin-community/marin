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
_ALICE_TASK = VerifiedIdentity(user_id="alice", role="task", job="/alice/parent")


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
    assert (identity.user_id, identity.role, identity.job) == ("alice", "task", _JOB.to_wire())


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


_PRIVILEGED = {"profile": job_pb2.CONTAINER_PROFILE_PRIVILEGED}
_PRODUCTION = {"band": job_pb2.PRIORITY_BAND_PRODUCTION}


@pytest.mark.parametrize(
    ("parent", "child", "allowed"),
    [
        pytest.param(_PRIVILEGED, _PRIVILEGED, True, id="privileged-under-privileged"),
        pytest.param({"profile": job_pb2.CONTAINER_PROFILE_DEFAULT}, _PRIVILEGED, False, id="privileged-under-default"),
        pytest.param(_PRODUCTION, _PRODUCTION, True, id="production-under-production"),
        pytest.param({"band": job_pb2.PRIORITY_BAND_INTERACTIVE}, _PRODUCTION, False, id="production-under-interactive"),
    ],
)
def test_task_gives_a_child_an_admin_gated_setting_only_when_its_parent_holds_it(auth_service, parent, child, allowed):
    with identity_scope(_ADMIN):
        _launch(auth_service, "/alice/parent", **parent)

    with identity_scope(_ALICE_TASK):
        if allowed:
            _launch(auth_service, "/alice/parent/child", **child)
            return
        with pytest.raises(ConnectError) as exc:
            _launch(auth_service, "/alice/parent/child", **child)
    assert exc.value.code == Code.PERMISSION_DENIED


def test_task_of_another_job_cannot_borrow_a_privileged_parent(auth_service):
    with identity_scope(_ADMIN):
        _launch(auth_service, "/alice/parent", **_PRIVILEGED)

    other_task = VerifiedIdentity(user_id="alice", role="task", job="/alice/other")
    with identity_scope(other_task), pytest.raises(ConnectError) as exc:
        _launch(auth_service, "/alice/parent/child", **_PRIVILEGED)
    assert exc.value.code == Code.PERMISSION_DENIED


def test_task_cannot_launch_under_another_users_job(auth_service):
    with identity_scope(_ADMIN):
        _launch(auth_service, "/bob/parent")

    with identity_scope(_ALICE_TASK), pytest.raises(ConnectError) as exc:
        _launch(auth_service, "/bob/parent/child")
    assert exc.value.code == Code.PERMISSION_DENIED
