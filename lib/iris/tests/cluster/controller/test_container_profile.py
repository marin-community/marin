# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authorization and persistence for container security profiles.

Elevated profiles (DOCKER_ACCESS, PRIVILEGED) are host-root-equivalent, so
``launch_job`` gates them on the admin role when an auth provider is configured.
RESTRICTED and DEFAULT are unprivileged and need no authorization. The accepted
profile is persisted on ``job_config`` and stamped onto each dispatched
``RunTaskRequest``.
"""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.bundle import BundleStore
from iris.cluster.controller import reads
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.backend import BackendCapability
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.projections.run_templates import RunTemplatesProjection
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from rigging.server_auth import VerifiedIdentity, _verified_identity
from tests.cluster.controller.conftest import (
    MockController,
    make_controller_state,
    make_test_entrypoint,
)

PRIVILEGED = job_pb2.CONTAINER_PROFILE_PRIVILEGED
DOCKER_ACCESS = job_pb2.CONTAINER_PROFILE_DOCKER_ACCESS
RESTRICTED = job_pb2.CONTAINER_PROFILE_RESTRICTED
DEFAULT = job_pb2.CONTAINER_PROFILE_DEFAULT


@pytest.fixture
def state():
    with make_controller_state() as s:
        yield s


def _make_service(state, tmp_path, log_client, auth: ControllerAuth) -> ControllerServiceImpl:
    return ControllerServiceImpl(
        controller=MockController(),
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        auth=auth,
        endpoint_service=EndpointServiceImpl(db=state._db),
    )


@pytest.fixture
def service(state, tmp_path, log_client) -> ControllerServiceImpl:
    """Service with a configured auth provider (so elevation gates on admin)."""
    return _make_service(state, tmp_path, log_client, ControllerAuth(provider="static"))


def _as(role: str, user_id: str, fn, *args, **kwargs):
    reset = _verified_identity.set(VerifiedIdentity(user_id=user_id, role=role))
    try:
        return fn(*args, **kwargs)
    finally:
        _verified_identity.reset(reset)


def _launch(name: str, profile: int) -> controller_pb2.Controller.LaunchJobRequest:
    return controller_pb2.Controller.LaunchJobRequest(
        name=name,
        entrypoint=make_test_entrypoint(),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        container_profile=profile,
    )


@pytest.mark.parametrize("profile", [PRIVILEGED, DOCKER_ACCESS])
def test_non_admin_cannot_use_elevated_profile(service, profile):
    with pytest.raises(ConnectError) as exc:
        _as("user", "alice", service.launch_job, _launch("/alice/job", profile), None)
    assert exc.value.code == Code.PERMISSION_DENIED


@pytest.mark.parametrize("profile", [PRIVILEGED, DOCKER_ACCESS])
def test_admin_can_use_elevated_profile(service, profile):
    resp = _as("admin", "admin", service.launch_job, _launch("/admin/job", profile), None)
    assert resp.job_id == "/admin/job"


@pytest.mark.parametrize("profile", [RESTRICTED, DEFAULT, job_pb2.CONTAINER_PROFILE_UNSPECIFIED])
def test_non_admin_can_use_unprivileged_profile(service, profile):
    """RESTRICTED/DEFAULT/UNSPECIFIED need no authorization."""
    resp = _as("user", "alice", service.launch_job, _launch("/alice/job", profile), None)
    assert resp.job_id == "/alice/job"


def test_docker_access_rejected_on_cluster_backend(state, tmp_path, log_client):
    """DOCKER_ACCESS needs the docker worker backend; a CLUSTER_VIEW (k8s)
    backend rejects it at submit so it never stalls the reconcile tick."""
    service = _make_service(state, tmp_path, log_client, ControllerAuth(provider="static"))
    service._controller.capabilities = frozenset({BackendCapability.CLUSTER_VIEW})
    with pytest.raises(ConnectError) as exc:
        _as("admin", "admin", service.launch_job, _launch("/admin/job", DOCKER_ACCESS), None)
    assert exc.value.code == Code.INVALID_ARGUMENT
    assert "docker_access" in str(exc.value.message).lower()


def test_null_auth_allows_elevated(state, tmp_path, log_client):
    """No auth provider: every caller is the anonymous admin, so the elevation
    gate is a no-op (the operator has opted into an untrusted cluster)."""
    service = _make_service(state, tmp_path, log_client, ControllerAuth())
    resp = _as("admin", "anonymous", service.launch_job, _launch("/anonymous/job", PRIVILEGED), None)
    assert resp.job_id == "/anonymous/job"


def test_profile_persisted_and_stamped_on_run_request(service, state):
    """An accepted PRIVILEGED profile lands on job_config and on the RunTaskRequest."""
    job_id = JobName.from_wire("/admin/job")
    _as("admin", "admin", service.launch_job, _launch(job_id.to_wire(), PRIVILEGED), None)

    with state._db.read_snapshot() as snap:
        detail = reads.get_job_detail(snap, job_id)
        assert detail is not None
        assert detail.container_profile == PRIVILEGED

        template = snap.caches[RunTemplatesProjection].get(snap, job_id)
    assert template is not None
    assert template.container_profile == PRIVILEGED
