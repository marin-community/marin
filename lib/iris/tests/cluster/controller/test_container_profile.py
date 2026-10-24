# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authorization and persistence for container security profiles.

Elevated profiles (DOCKER_ACCESS, PRIVILEGED) are host-root-equivalent, so
``launch_job`` gates them on the admin role when an auth provider is configured.
RESTRICTED and DEFAULT are unprivileged and need no authorization. The accepted
profile is persisted on ``job_config`` and stamped onto each dispatched
``RunTaskRequest``.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.backends.k8s.tasks import K8sTaskProvider
from iris.cluster.bundle import BundleStore
from iris.cluster.config import EgressAllowlistConfig
from iris.cluster.controller import reads
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.projections.run_templates import RunTemplatesProjection
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.egress import MAX_EGRESS_RULES, format_egress_rule, parse_egress_rules
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2, worker_pb2
from iris.testing.controller import (
    MockController,
    make_controller_state,
    make_test_entrypoint,
)
from iris.testing.k8s import k8s_backend_descriptor, pod_config
from rigging.server_auth import VerifiedIdentity, _verified_identity

PRIVILEGED = job_pb2.CONTAINER_PROFILE_PRIVILEGED
DOCKER_ACCESS = job_pb2.CONTAINER_PROFILE_DOCKER_ACCESS
RESTRICTED = job_pb2.CONTAINER_PROFILE_RESTRICTED
DEFAULT = job_pb2.CONTAINER_PROFILE_DEFAULT
SANDBOX = job_pb2.CONTAINER_PROFILE_SANDBOX


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


def _launch(
    name: str,
    profile: int,
    egress: int = job_pb2.EGRESS_POLICY_UNSPECIFIED,
    rules: Sequence[str] = (),
    replicas: int = 1,
    gpus: int = 0,
) -> controller_pb2.Controller.LaunchJobRequest:
    resources = job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3)
    if gpus:
        resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="H100", count=gpus))
    return controller_pb2.Controller.LaunchJobRequest(
        name=name,
        entrypoint=make_test_entrypoint(),
        environment=job_pb2.EnvironmentConfig(),
        replicas=replicas,
        resources=resources,
        container_profile=profile,
        egress_policy=egress,
        egress_allow=parse_egress_rules(rules),
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


@pytest.mark.parametrize("profile", [RESTRICTED, DEFAULT, SANDBOX, job_pb2.CONTAINER_PROFILE_UNSPECIFIED])
def test_non_admin_can_use_unprivileged_profile(service, profile):
    """RESTRICTED/DEFAULT/SANDBOX/UNSPECIFIED need no authorization.

    SANDBOX is un-gated on purpose: gVisor gives in-guest capabilities while
    isolating the host, so it is safe to hand out without the admin role.
    """
    resp = _as("user", "alice", service.launch_job, _launch("/alice/job", profile), None)
    assert resp.job_id == "/alice/job"


@pytest.mark.parametrize("profile", [job_pb2.CONTAINER_PROFILE_GVISOR, SANDBOX])
def test_gvisor_profiles_rejected_on_accelerator_task(service, profile):
    """gVisor cannot pass a GPU/TPU through, so an accelerator task is rejected."""
    req = _launch("/alice/job", profile)
    req.resources.device.gpu.count = 1
    with pytest.raises(ConnectError) as exc:
        _as("user", "alice", service.launch_job, req, None)
    assert exc.value.code == Code.INVALID_ARGUMENT
    assert "cpu-only" in str(exc.value.message).lower()


def test_legacy_gvisor_submission_preserves_bundle_environment_and_profile(service):
    # An already-running client sends the old wire value and a workspace bundle.
    req = _launch("/alice/job", 5)
    req.bundle_blob = b"legacy-workspace"
    req.environment.env_vars["TASK_VAR"] = "legacy-value"
    response = _as("user", "alice", service.launch_job, req, None)

    stored = _as(
        "user",
        "alice",
        service.get_job_status,
        controller_pb2.Controller.GetJobStatusRequest(job_id=response.job_id),
        None,
    ).request
    assert stored.container_profile == job_pb2.CONTAINER_PROFILE_GVISOR
    assert stored.egress_policy == job_pb2.EGRESS_POLICY_CLUSTER
    assert service.bundle_zip(stored.bundle_id) == b"legacy-workspace"
    assert stored.environment.env_vars["TASK_VAR"] == "legacy-value"


def test_sandbox_rejects_workspace_bundle(service):
    req = _launch("/alice/job", SANDBOX)
    req.bundle_blob = b"workspace"
    with pytest.raises(ConnectError) as exc:
        _as("user", "alice", service.launch_job, req, None)
    assert exc.value.code == Code.INVALID_ARGUMENT
    assert "bundle" in str(exc.value.message)


def _service_group_request():
    request = _launch("/alice/group", SANDBOX, job_pb2.EGRESS_POLICY_NONE)
    request.task_image = "example.test/task@sha256:" + "a" * 64
    request.resources.cpu_millicores = 2000
    request.resources.memory_bytes = 2 * 1024**3
    request.environment.sandbox_services.add(
        name="runtime",
        image="example.test/runtime@sha256:" + "b" * 64,
        workdir="/app",
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
    )
    return request


def test_sandbox_service_group_persists_on_admission(service):
    service._controller.backend.descriptor = k8s_backend_descriptor()
    response = _as("user", "alice", service.launch_job, _service_group_request(), None)
    stored = _as(
        "user",
        "alice",
        service.get_job_status,
        controller_pb2.Controller.GetJobStatusRequest(job_id=response.job_id),
        None,
    ).request
    assert stored.resources.cpu_millicores == 2000
    assert stored.environment.sandbox_services[0].name == "runtime"


def test_service_exec_cannot_select_infrastructure_containers(service):
    service._controller.backend.descriptor = k8s_backend_descriptor()
    _as("user", "alice", service.launch_job, _service_group_request(), None)
    backend = service._controller.backend
    backend.exec_in_container.return_value = worker_pb2.Worker.ExecInContainerResponse(stdout="runtime")
    request = controller_pb2.Controller.ExecInContainerRequest(
        task_id="/alice/group/0",
        container_name="runtime",
        command=["true"],
    )
    result = _as("user", "alice", service.exec_in_container, request, None)
    assert result.stdout == "runtime"
    assert backend.exec_in_container.call_args.args[1].container_name == "runtime"
    backend.exec_in_container.reset_mock()
    request.container_name = "output-uploader"
    with pytest.raises(ConnectError) as error:
        _as("user", "alice", service.exec_in_container, request, None)
    assert error.value.code == Code.INVALID_ARGUMENT
    backend.exec_in_container.assert_not_called()


@pytest.mark.parametrize("location", ["task", "service", "secret", "egress"])
def test_service_groups_reject_credential_and_network_escalation_at_admission(service, location):
    service._controller.backend.descriptor = k8s_backend_descriptor()
    request = _service_group_request()
    if location == "task":
        request.environment.env_vars["AWS_SECRET_ACCESS_KEY"] = "private"
    elif location == "service":
        request.environment.sandbox_services[0].env_vars["GOOGLE_APPLICATION_CREDENTIALS"] = "/host/key.json"
    elif location == "secret":
        request.environment.secrets["CW_KEY_SECRET"] = "private"
    else:
        request.egress_policy = job_pb2.EGRESS_POLICY_INTERNET
    with pytest.raises(ConnectError) as error:
        _as("user", "alice", service.launch_job, request, None)
    assert error.value.code == Code.INVALID_ARGUMENT


def test_unknown_profile_rejected(service):
    """A profile this controller does not know must not fall back to DEFAULT."""
    with pytest.raises(ConnectError) as exc:
        _as("user", "alice", service.launch_job, _launch("/alice/job", 99), None)
    assert exc.value.code == Code.INVALID_ARGUMENT


def test_docker_access_rejected_on_cluster_backend(state, tmp_path, log_client):
    """DOCKER_ACCESS needs the docker worker backend; a Kubernetes
    backend rejects it at submit so it never stalls the reconcile tick."""
    service = _make_service(state, tmp_path, log_client, ControllerAuth(provider="static"))
    service._controller.backend.descriptor = k8s_backend_descriptor()
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


def _stored_egress(state, job_id: JobName) -> tuple[int, int]:
    with state._db.read_snapshot() as snap:
        detail = reads.get_job_detail(snap, job_id)
        template = snap.caches[RunTemplatesProjection].get(snap, job_id)
    assert detail is not None and template is not None
    return detail.egress_policy, template.egress_policy


@pytest.mark.parametrize(
    ("profile", "requested", "resolved"),
    [
        pytest.param(SANDBOX, job_pb2.EGRESS_POLICY_UNSPECIFIED, job_pb2.EGRESS_POLICY_INTERNET, id="sandbox-default"),
        pytest.param(DEFAULT, job_pb2.EGRESS_POLICY_UNSPECIFIED, job_pb2.EGRESS_POLICY_CLUSTER, id="default-default"),
        pytest.param(DEFAULT, job_pb2.EGRESS_POLICY_NONE, job_pb2.EGRESS_POLICY_NONE, id="default-none"),
        pytest.param(SANDBOX, job_pb2.EGRESS_POLICY_INTERNET, job_pb2.EGRESS_POLICY_INTERNET, id="sandbox-internet"),
    ],
)
def test_egress_policy_resolved_and_stored(service, state, profile, requested, resolved):
    _as("user", "alice", service.launch_job, _launch("/alice/job", profile, requested), None)

    assert _stored_egress(state, JobName.from_wire("/alice/job")) == (resolved, resolved)


@pytest.mark.parametrize(
    ("profile", "egress", "rules", "message"),
    [
        pytest.param(
            SANDBOX, job_pb2.EGRESS_POLICY_CLUSTER, (), "cannot use egress policy cluster", id="sandbox-cluster"
        ),
        pytest.param(
            SANDBOX, job_pb2.EGRESS_POLICY_ALLOWLIST, (), "cannot use egress policy allowlist", id="sandbox-allowlist"
        ),
        pytest.param(
            DEFAULT, job_pb2.EGRESS_POLICY_CLUSTER, ("pypi.org",), "require egress policy allowlist", id="rules-cluster"
        ),
        pytest.param(DEFAULT, 99, (), "Unknown egress policy", id="unknown"),
    ],
)
def test_egress_policy_rejected(service, profile, egress, rules, message):
    with pytest.raises(ConnectError) as exc:
        _as("user", "alice", service.launch_job, _launch("/alice/job", profile, egress, rules), None)
    assert exc.value.code == Code.INVALID_ARGUMENT
    assert message in str(exc.value.message)


def _stored_rules(state, job_id: JobName) -> list[str]:
    with state._db.read_snapshot() as snap:
        template = snap.caches[RunTemplatesProjection].get(snap, job_id)
    assert template is not None
    return [format_egress_rule(rule) for rule in template.egress_allow]


@pytest.mark.parametrize(
    ("service_fixture", "profile", "resolved"),
    [
        pytest.param("service", DEFAULT, job_pb2.EGRESS_POLICY_CLUSTER, id="open-cluster"),
        pytest.param("allowlist_default_service", DEFAULT, job_pb2.EGRESS_POLICY_ALLOWLIST, id="allowlist-cluster"),
        pytest.param("allowlist_default_service", SANDBOX, job_pb2.EGRESS_POLICY_INTERNET, id="allowlist-sandbox"),
        pytest.param("no_cilium_service", DEFAULT, job_pb2.EGRESS_POLICY_CLUSTER, id="no-cilium-cluster"),
    ],
)
def test_egress_rules_keep_the_clusters_default_policy(request, state, service_fixture, profile, resolved):
    """A launcher declares rules on every cluster; they never move a job onto ALLOWLIST."""
    service = request.getfixturevalue(service_fixture)
    job_id = JobName.from_wire("/alice/job")
    _as("user", "alice", service.launch_job, _launch(job_id.to_wire(), profile, rules=["api.openai.com:443"]), None)

    assert _stored_egress(state, job_id) == (resolved, resolved)
    assert _stored_rules(state, job_id) == ["api.openai.com:443"]


@contextmanager
def _k8s_backend(service: ControllerServiceImpl, egress_allowlist: EgressAllowlistConfig | None) -> Iterator[None]:
    """Route the service's egress resolution and job validation to a Kubernetes backend."""
    backend = K8sTaskProvider(
        descriptor=k8s_backend_descriptor(),
        kubectl=InMemoryK8sService(namespace="iris"),
        pods=pod_config(host_network=True, egress_allowlist=egress_allowlist),
    )
    service._controller.backend.egress_allowlist_mode.side_effect = backend.egress_allowlist_mode
    service._controller.backend.validate_job.side_effect = backend.validate_job
    try:
        yield
    finally:
        backend.close()


@pytest.fixture
def allowlist_default_service(service):
    """The service in front of a host-network Kubernetes backend whose cluster defaults to the allowlist."""
    with _k8s_backend(service, EgressAllowlistConfig(default=True)):
        yield service


@pytest.fixture
def no_cilium_service(service):
    """The service in front of a Kubernetes backend without Cilium, which cannot enforce the allowlist."""
    with _k8s_backend(service, None):
        yield service


def test_allowlist_runs_on_the_cluster_network_where_the_cluster_cannot_enforce_it(no_cilium_service, state):
    """Launchers can request the allowlist before every cluster enforces it."""
    job_id = JobName.from_wire("/alice/job")
    request = _launch(job_id.to_wire(), DEFAULT, job_pb2.EGRESS_POLICY_ALLOWLIST, ["api.openai.com:443"])
    _as("user", "alice", no_cilium_service.launch_job, request, None)

    cluster = job_pb2.EGRESS_POLICY_CLUSTER
    assert _stored_egress(state, job_id) == (cluster, cluster)
    assert _stored_rules(state, job_id) == ["api.openai.com:443"]


@pytest.mark.parametrize(
    ("profile", "resolved"),
    [
        pytest.param(DEFAULT, job_pb2.EGRESS_POLICY_ALLOWLIST, id="default"),
        pytest.param(SANDBOX, job_pb2.EGRESS_POLICY_INTERNET, id="sandbox"),
    ],
)
def test_cluster_allowlist_default_applies_to_every_job_but_a_sandbox(
    allowlist_default_service, state, profile, resolved
):
    _as("user", "alice", allowlist_default_service.launch_job, _launch("/alice/job", profile), None)

    assert _stored_egress(state, JobName.from_wire("/alice/job")) == (resolved, resolved)


@pytest.mark.parametrize(
    ("parent_egress", "parent_rules"),
    [
        pytest.param(job_pb2.EGRESS_POLICY_CLUSTER, (), id="opted-out"),
        pytest.param(job_pb2.EGRESS_POLICY_UNSPECIFIED, ("api.openai.com:443", "203.0.113.0/24"), id="extra-rules"),
    ],
)
def test_child_job_inherits_its_parents_egress(allowlist_default_service, state, parent_egress, parent_rules):
    """A tree that opted out of the allowlist, or added rules, keeps that choice in every child."""
    service = allowlist_default_service
    _as("user", "alice", service.launch_job, _launch("/alice/parent", DEFAULT, parent_egress, parent_rules), None)
    _as("user", "alice", service.launch_job, _launch("/alice/parent/child", DEFAULT), None)

    parent, child = JobName.from_wire("/alice/parent"), JobName.from_wire("/alice/parent/child")
    assert _stored_egress(state, child) == _stored_egress(state, parent)
    assert _stored_rules(state, child) == list(parent_rules)


def test_child_rules_and_inherited_rules_share_the_rule_cap(allowlist_default_service, state):
    service = allowlist_default_service
    parent_rules = [f"host{i}.example.com" for i in range(MAX_EGRESS_RULES)]
    _as("user", "alice", service.launch_job, _launch("/alice/parent", DEFAULT, rules=parent_rules), None)

    repeat = JobName.from_wire("/alice/parent/repeat")
    _as("user", "alice", service.launch_job, _launch(repeat.to_wire(), DEFAULT, rules=parent_rules[:1]), None)
    assert sorted(_stored_rules(state, repeat)) == sorted(parent_rules)

    with pytest.raises(ConnectError) as exc:
        _as("user", "alice", service.launch_job, _launch("/alice/parent/extra", DEFAULT, rules=["pypi.org"]), None)
    assert exc.value.code == Code.INVALID_ARGUMENT
    assert f"at most {MAX_EGRESS_RULES}" in str(exc.value.message)


@pytest.mark.parametrize(
    ("replicas", "gpus", "egress", "resolved"),
    [
        pytest.param(1, 8, job_pb2.EGRESS_POLICY_UNSPECIFIED, job_pb2.EGRESS_POLICY_ALLOWLIST, id="single-host-gpu"),
        pytest.param(2, 0, job_pb2.EGRESS_POLICY_UNSPECIFIED, job_pb2.EGRESS_POLICY_ALLOWLIST, id="multi-host-cpu"),
        pytest.param(2, 8, job_pb2.EGRESS_POLICY_UNSPECIFIED, job_pb2.EGRESS_POLICY_CLUSTER, id="multi-host-gpu"),
        pytest.param(2, 8, job_pb2.EGRESS_POLICY_ALLOWLIST, job_pb2.EGRESS_POLICY_CLUSTER, id="multi-host-gpu-asked"),
    ],
)
def test_multi_host_gpu_job_is_recorded_on_the_cluster_network(
    allowlist_default_service, state, replicas, gpus, egress, resolved
):
    """NCCL needs host networking for InfiniBand, which pod policy cannot restrict; the stored policy says so."""
    job_id = JobName.from_wire("/alice/job")
    request = _launch(job_id.to_wire(), DEFAULT, egress, replicas=replicas, gpus=gpus)
    _as("user", "alice", allowlist_default_service.launch_job, request, None)

    assert _stored_egress(state, job_id) == (resolved, resolved)
