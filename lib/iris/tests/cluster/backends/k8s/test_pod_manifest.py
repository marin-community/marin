# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for pod manifest building: naming, env vars, volumes, constraints, init containers."""

import json

import pytest
from iris.cluster.backends.k8s.tasks import (
    _INFRASTRUCTURE_FAILURE_REASONS,
    _KUEUE_POD_GROUP_NAME,
    _KUEUE_POD_GROUP_POD_INDEX,
    _KUEUE_POD_GROUP_TOTAL,
    _KUEUE_PREFERRED_TOPOLOGY,
    _KUEUE_PRIORITY_CLASS,
    _KUEUE_QUEUE_NAME,
    _KUEUE_REQUIRED_TOPOLOGY,
    _KUEUE_SLICE_REQUIRED_TOPOLOGY,
    _KUEUE_SLICE_SIZE,
    _LABEL_JOB_ID,
    _LABEL_TASK_HASH,
    _build_init_container_spec,
    _build_pdb_manifest,
    _build_pod_manifest,
    _build_task_script,
    _build_volumes_and_mounts,
    _constraints_to_node_selector,
    _is_coordinator_task,
    _is_infrastructure_failure,
    _job_id_from_task,
    _pod_group_name,
    _pod_name,
    _sanitize_label_value,
    _security_context,
    _task_hash,
    _task_update_from_pod,
)
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.platforms.k8s.coreweave_topology import (
    NVL72_GPUS_PER_NODE,
    RACK_SIZE,
    SCHEDULABLE_RACK_NODES,
    KueueTopologyBinding,
    TopologyMode,
)
from iris.cluster.platforms.k8s.types import parse_k8s_quantity
from iris.cluster.runtime.env import STANDARD_MOUNTS
from iris.cluster.runtime.types import MountKind
from iris.cluster.types import JobName
from iris.rpc import job_pb2

from .conftest import add_eq_constraint, common_env_from_req, make_pod, make_run_req, pod_config

# ---------------------------------------------------------------------------
# Pod naming
# ---------------------------------------------------------------------------


def test_pod_name_sanitizes_slashes():
    name = _pod_name(JobName.from_wire("/smoke-job/0"), 1)
    assert "/" not in name
    assert name.startswith("iris-")
    assert name.islower()


def test_pod_name_length_limit():
    long_task = "/a" * 50
    name = _pod_name(JobName.from_wire(long_task), 0)
    assert len(name) <= 63


def test_pod_name_deterministic():
    task = JobName.from_wire("/test-job/42")
    assert _pod_name(task, 0) == _pod_name(task, 0)
    assert _pod_name(task, 0) != _pod_name(task, 1)


def test_pod_name_preserves_attempt_suffix_with_long_task_id():
    long_task = JobName.from_wire("/a" * 40)
    name_0 = _pod_name(long_task, 0)
    name_1 = _pod_name(long_task, 1)
    name_999 = _pod_name(long_task, 999)
    assert len(name_0) <= 63
    assert len(name_1) <= 63
    assert len(name_999) <= 63
    assert name_0 != name_1, "different attempts must produce different pod names"
    assert name_0.endswith("-0")
    assert name_1.endswith("-1")
    assert name_999.endswith("-999")


def test_pod_name_different_tasks_never_collide():
    task_a = JobName.from_wire("/a" * 40 + "-suffix-1")
    task_b = JobName.from_wire("/a" * 40 + "-suffix-2")
    assert _pod_name(task_a, 1) != _pod_name(
        task_b, 1
    ), "sibling tasks with the same long prefix must have different pod names"


# ---------------------------------------------------------------------------
# Pod manifest building
# ---------------------------------------------------------------------------


def test_build_pod_manifest_fields():
    req = make_run_req("/test-job/0", attempt_id=2)
    manifest = _build_pod_manifest(req, pod_config())

    assert manifest["kind"] == "Pod"
    assert manifest["metadata"]["namespace"] == "iris"
    assert manifest["spec"]["restartPolicy"] == "Never"

    container = manifest["spec"]["containers"][0]
    assert container["image"] == "myrepo/iris:latest"
    assert container["command"][0] == "bash"
    assert container["command"][1] == "-lc"
    assert "exec python train.py" in container["command"][2]

    # CPU is requested only (no limit) so containers can burst onto idle node
    # CPU; memory is both requested and limited (overshoot is fatal).
    assert container["resources"]["requests"]["cpu"] == "1000m"
    assert "cpu" not in container["resources"].get("limits", {})
    assert container["resources"]["limits"]["memory"] == str(4 * 1024**3)
    assert container["resources"]["requests"]["memory"] == str(4 * 1024**3)


def test_build_pod_manifest_defaults_image_when_no_override():
    req = make_run_req("/test-job/0")
    manifest = _build_pod_manifest(req, pod_config(default_image="myrepo/iris:latest"))
    assert manifest["spec"]["containers"][0]["image"] == "myrepo/iris:latest"


def test_build_pod_manifest_honors_task_image_override():
    """RunTaskRequest.task_image overrides the task container image. The init
    container keeps default_image (see _build_init_container_spec) since it runs
    iris's own bundle_fetch tooling."""
    req = make_run_req("/test-job/0")
    req.task_image = "myrepo/custom:v9"
    manifest = _build_pod_manifest(req, pod_config(default_image="myrepo/iris:latest"))
    assert manifest["spec"]["containers"][0]["image"] == "myrepo/custom:v9"


def test_build_pod_manifest_env_vars():
    req = make_run_req("/test-job/0")
    req.environment.env_vars["MY_VAR"] = "hello"
    manifest = _build_pod_manifest(req, pod_config())
    env_names = {e["name"] for e in manifest["spec"]["containers"][0]["env"]}
    assert "MY_VAR" in env_names
    assert "IRIS_JOB_ID" in env_names
    assert "IRIS_TASK_ID" in env_names
    assert "IRIS_NUM_TASKS" in env_names
    assert "IRIS_BIND_HOST" in env_names
    assert "IRIS_WORKDIR" in env_names
    assert "IRIS_ADVERTISE_HOST" in env_names


def test_build_pod_manifest_env_secret_adds_envfrom():
    req = make_run_req("/test-job/0")
    manifest = _build_pod_manifest(req, pod_config(env_secret_name="iris-task-env"))
    container = manifest["spec"]["containers"][0]
    assert container["envFrom"] == [{"secretRef": {"name": "iris-task-env", "optional": True}}]


def test_build_pod_manifest_no_env_secret_omits_envfrom():
    req = make_run_req("/test-job/0")
    manifest = _build_pod_manifest(req, pod_config())
    assert "envFrom" not in manifest["spec"]["containers"][0]


def test_build_pod_manifest_task_container_falls_back_to_logs_on_error():
    """The task container captures its tail log output into terminated.message
    on a non-zero exit, instead of leaving operators with a bare "Error" reason
    and no clue what actually happened."""
    req = make_run_req("/test-job/0")
    manifest = _build_pod_manifest(req, pod_config())
    container = manifest["spec"]["containers"][0]
    assert container["terminationMessagePolicy"] == "FallbackToLogsOnError"


def test_build_pod_manifest_gpu():
    req = make_run_req("/test-job/0")
    req.resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="A100", count=4))
    manifest = _build_pod_manifest(req, pod_config())
    limits = manifest["spec"]["containers"][0]["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == "4"
    assert "rdma/ib" not in limits


def test_build_pod_manifest_gpu_host_network_requests_rdma():
    req = make_run_req("/test-job/0")
    req.resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="A100", count=4))
    manifest = _build_pod_manifest(req, pod_config(host_network=True))
    limits = manifest["spec"]["containers"][0]["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == "4"
    assert limits["rdma/ib"] == "4"


def test_build_pod_manifest_runtime_label():
    req = make_run_req("/test-job/0")
    manifest = _build_pod_manifest(req, pod_config())
    assert manifest["metadata"]["labels"]["iris.runtime"] == "iris-kubernetes"


def test_build_pod_manifest_task_hash_label():
    req = make_run_req("/test-job/0")
    manifest = _build_pod_manifest(req, pod_config())
    labels = manifest["metadata"]["labels"]
    assert labels[_LABEL_TASK_HASH] == _task_hash("/test-job/0")
    assert len(labels[_LABEL_TASK_HASH]) <= 63
    assert labels[_LABEL_TASK_HASH].isalnum()


def test_task_hash_distinct_for_sanitization_collisions():
    base = "a" * 63
    id_a = base + "X"
    id_b = base + "Y"
    assert _sanitize_label_value(id_a) == _sanitize_label_value(id_b), "precondition: same sanitized value"
    assert _task_hash(id_a) != _task_hash(id_b), "hashes must be distinct"


# ---------------------------------------------------------------------------
# Phase -> state mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phase,expected_state",
    [
        ("Pending", job_pb2.TASK_STATE_BUILDING),
        ("Running", job_pb2.TASK_STATE_RUNNING),
        ("Succeeded", job_pb2.TASK_STATE_SUCCEEDED),
        ("Failed", job_pb2.TASK_STATE_FAILED),
        ("Unknown", job_pb2.TASK_STATE_FAILED),
    ],
)
def test_task_update_from_pod_phases(phase, expected_state):
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    pod = make_pod("iris-job-0-0", phase, exit_code=1 if phase == "Failed" else None)
    update = _task_update_from_pod(entry, pod)
    assert update.new_state == expected_state


def test_task_update_failed_has_exit_code():
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    pod = make_pod("iris-job-0-0", "Failed", exit_code=42, reason="Error")
    update = _task_update_from_pod(entry, pod)
    assert update.exit_code == 42
    assert update.new_state == job_pb2.TASK_STATE_FAILED


@pytest.mark.parametrize("reason", sorted(_INFRASTRUCTURE_FAILURE_REASONS))
def test_task_update_infrastructure_failure_is_worker_failed(reason):
    """Evicted, Preempting, etc. should be WORKER_FAILED, not FAILED."""
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    pod = make_pod("iris-job-0-0", "Failed", exit_code=137, reason=reason)
    update = _task_update_from_pod(entry, pod)
    assert update.new_state == job_pb2.TASK_STATE_WORKER_FAILED
    assert update.exit_code == 137


def test_task_update_oom_killed_is_application_failure():
    """OOMKilled is a misconfiguration, not infrastructure — should be FAILED."""
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    pod = make_pod("iris-job-0-0", "Failed", exit_code=137, reason="OOMKilled")
    update = _task_update_from_pod(entry, pod)
    assert update.new_state == job_pb2.TASK_STATE_FAILED
    assert update.exit_code == 137


def test_task_update_application_error_is_failed():
    """Non-zero exit with reason 'Error' is an application failure, not infrastructure."""
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    pod = make_pod("iris-job-0-0", "Failed", exit_code=1, reason="Error")
    update = _task_update_from_pod(entry, pod)
    assert update.new_state == job_pb2.TASK_STATE_FAILED
    assert update.exit_code == 1


def test_task_update_error_prefers_termination_message_over_bare_reason():
    """With terminationMessagePolicy: FallbackToLogsOnError, the kubelet fills in
    ``message`` with the container's tail log output on a non-zero exit. This is
    the real payoff of that manifest field: _extract_error already prefers a
    non-empty message over the generic "Error" reason, so the actual crash
    (traceback, fatal-error banner, ...) reaches the task/job error instead."""
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    pod = make_pod(
        "iris-job-0-0",
        "Failed",
        exit_code=1,
        reason="Error",
        message="RuntimeError: CUDA error: an illegal memory access was encountered",
    )
    update = _task_update_from_pod(entry, pod)
    assert update.error == "RuntimeError: CUDA error: an illegal memory access was encountered"


def test_is_infrastructure_failure_with_pod_level_reason():
    """Pod-level eviction (no container statuses) is detected as infrastructure failure."""
    pod: dict = {
        "metadata": {"name": "test"},
        "status": {"phase": "Failed", "reason": "Evicted", "containerStatuses": []},
    }
    assert _is_infrastructure_failure(pod)


def _add_condition(pod: dict, type_: str, status: str, reason: str = "") -> dict:
    pod["status"].setdefault("conditions", []).append({"type": type_, "status": status, "reason": reason})
    return pod


@pytest.mark.parametrize("reason", ["PreemptionByScheduler", "TerminationByKubelet", "EvictionByEvictionAPI"])
def test_task_update_disruption_target_is_worker_failed(reason):
    """A preemption SIGKILLed after grace surfaces as reason='Error' exit 137 — not in
    the reason whitelist — but the control plane's DisruptionTarget condition marks it
    as infrastructure, so it must be WORKER_FAILED (preemption budget), not FAILED."""
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    pod = make_pod("iris-job-0-0", "Failed", exit_code=137, reason="Error")
    _add_condition(pod, "DisruptionTarget", "True", reason)
    update = _task_update_from_pod(entry, pod)
    assert update.new_state == job_pb2.TASK_STATE_WORKER_FAILED
    assert update.exit_code == 137


def test_task_update_oom_killed_without_disruption_target_stays_application_failure():
    """A self-inflicted cgroup OOM carries no DisruptionTarget condition, so it stays a
    FAILED (misconfigured job) even though it also exits 137 — the condition, not the
    exit code, is what distinguishes preemption from OOM guilt."""
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    pod = make_pod("iris-job-0-0", "Failed", exit_code=137, reason="OOMKilled")
    update = _task_update_from_pod(entry, pod)
    assert update.new_state == job_pb2.TASK_STATE_FAILED


def test_disruption_target_condition_status_false_is_not_infrastructure():
    """A DisruptionTarget condition with status != 'True' does not mark a disruption."""
    pod = make_pod("iris-job-0-0", "Failed", exit_code=1, reason="Error")
    _add_condition(pod, "DisruptionTarget", "False", "")
    assert not _is_infrastructure_failure(pod)


# ---------------------------------------------------------------------------
# Node resource parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("2", 2),
        ("500m", 500),
        ("4Gi", 4 * 1024**3),
        ("1024Mi", 1024 * 1024**2),
        ("100Ki", 100 * 1024),
        ("2G", 2 * 10**9),
        ("0", 0),
        ("", 0),
    ],
)
def test_parse_k8s_quantity(value, expected):
    assert parse_k8s_quantity(value) == expected


def test_parse_k8s_quantity_decimal():
    """Decimal quantities like '1.5' are parsed correctly."""
    assert parse_k8s_quantity("1.5") == 1
    assert parse_k8s_quantity("0.5Gi") == 0.5 * 1024**3


# ---------------------------------------------------------------------------
# Constraint -> nodeSelector mapping
# ---------------------------------------------------------------------------


def test_constraints_to_node_selector_pool():
    req = make_run_req("/my-job/task-0", attempt_id=1)
    add_eq_constraint(req, "pool", "h100-8x")

    manifest = _build_pod_manifest(req, pod_config())
    assert manifest["spec"]["nodeSelector"] == {"iris.pool": "h100-8x"}


def test_constraints_to_node_selector_region():
    req = make_run_req("/my-job/task-0")
    add_eq_constraint(req, "region", "US-WEST-04A")

    manifest = _build_pod_manifest(req, pod_config())
    assert manifest["spec"]["nodeSelector"] == {"iris.region": "US-WEST-04A"}


def test_constraints_to_node_selector_multiple():
    req = make_run_req("/my-job/task-0", attempt_id=1)
    add_eq_constraint(req, "pool", "h100-8x")
    add_eq_constraint(req, "region", "US-WEST-04A")

    manifest = _build_pod_manifest(req, pod_config())
    assert manifest["spec"]["nodeSelector"] == {
        "iris.pool": "h100-8x",
        "iris.region": "US-WEST-04A",
    }


def test_constraints_unknown_key_ignored():
    req = make_run_req("/my-job/task-0")
    add_eq_constraint(req, "custom_key", "foo")

    manifest = _build_pod_manifest(req, pod_config())
    assert "nodeSelector" not in manifest["spec"]


def test_constraints_non_eq_op_raises():
    c = job_pb2.Constraint(key="pool", op=job_pb2.CONSTRAINT_OP_NE)
    c.value.string_value = "h100-8x"

    with pytest.raises(ValueError, match=r"Unsupported constraint op.*pool.*CONSTRAINT_OP_EQ"):
        _constraints_to_node_selector([c])


def test_constraints_to_node_selector_function_directly():
    """Unit test the helper in isolation."""
    c = job_pb2.Constraint(key="pool", op=job_pb2.CONSTRAINT_OP_EQ)
    c.value.string_value = "a100-4x"
    assert _constraints_to_node_selector([c]) == {"iris.pool": "a100-4x"}


def test_constraints_to_node_selector_empty():
    assert _constraints_to_node_selector([]) == {}


# ---------------------------------------------------------------------------
# GPU tolerations
# ---------------------------------------------------------------------------


def test_build_pod_manifest_no_gpu_no_toleration():
    req = make_run_req("/my-job/task-0")

    manifest = _build_pod_manifest(req, pod_config())
    assert "tolerations" not in manifest["spec"]


def test_nvidia_gpu_toleration_added():
    """GPU pods tolerate both the NVIDIA GPU taint and CoreWeave interruptable capacity."""
    req = make_run_req("/my-job/task-0")
    req.resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="A100", count=4))

    manifest = _build_pod_manifest(req, pod_config())
    tolerations = manifest["spec"].get("tolerations", [])
    toleration_keys = {t.get("key") for t in tolerations}
    assert "nvidia.com/gpu" in toleration_keys
    assert "qos.coreweave.cloud/interruptable" in toleration_keys


def test_coreweave_constraints_end_to_end():
    """Constraints from a coreweave h100-8x scale group map to correct nodeSelector."""
    req = make_run_req("/my-job/task-0", attempt_id=1)
    req.resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="H100", count=8))
    add_eq_constraint(req, "pool", "h100-8x")
    add_eq_constraint(req, "region", "US-WEST-04A")

    manifest = _build_pod_manifest(req, pod_config(default_image="ghcr.io/marin-community/iris-task:latest"))
    spec = manifest["spec"]

    assert spec["nodeSelector"]["iris.pool"] == "h100-8x"
    assert spec["nodeSelector"]["iris.region"] == "US-WEST-04A"
    assert any(t.get("key") == "qos.coreweave.cloud/interruptable" for t in spec["tolerations"])


# ---------------------------------------------------------------------------
# No non-Kueue colocation: it's Kueue or nothing
# ---------------------------------------------------------------------------


def test_multi_task_non_coscheduled_job_has_no_affinity():
    """A plain multi-task job (no coscheduling) gets no podAffinity: there is no
    non-Kueue colocation fallback. Topology placement comes only via Kueue."""
    req = make_run_req("/my-job/task-0", attempt_id=1, num_tasks=4)
    manifest = _build_pod_manifest(req, pod_config(default_image="img:latest"))
    assert "affinity" not in manifest["spec"]


def test_job_id_label_on_pod():
    """Pod metadata includes iris.job_id label derived from the task's parent path."""
    req = make_run_req("/my-job/task-0", attempt_id=1)
    manifest = _build_pod_manifest(req, pod_config(default_image="img:latest"))
    job_id = manifest["metadata"]["labels"][_LABEL_JOB_ID]
    assert "my-job" in job_id
    assert "task-0" not in job_id


def test_job_id_from_task_strips_task_suffix():
    """_job_id_from_task extracts the parent path from a task wire ID."""
    task_id = JobName.from_wire("/my-job/task-0")
    job_id = _job_id_from_task(task_id)
    assert "task-0" not in job_id
    assert "my-job" in job_id


def test_job_id_shared_across_sibling_tasks():
    """Sibling tasks from the same job produce the same job_id label."""
    task_0 = JobName.from_wire("/training-run/task-0")
    task_1 = JobName.from_wire("/training-run/task-1")
    assert _job_id_from_task(task_0) == _job_id_from_task(task_1)


# ---------------------------------------------------------------------------
# Timeout -> activeDeadlineSeconds
# ---------------------------------------------------------------------------


def test_timeout_sets_active_deadline_seconds():
    req = make_run_req("/my-job/task-0")
    req.timeout.milliseconds = 3600_000  # 1 hour
    manifest = _build_pod_manifest(req, pod_config(default_image="img:latest"))
    assert manifest["spec"]["activeDeadlineSeconds"] == 3600


def test_timeout_rounds_down_to_at_least_one_second():
    req = make_run_req("/my-job/task-0")
    req.timeout.milliseconds = 500  # sub-second
    manifest = _build_pod_manifest(req, pod_config(default_image="img:latest"))
    assert manifest["spec"]["activeDeadlineSeconds"] == 1


def test_no_timeout_no_deadline():
    req = make_run_req("/my-job/task-0")
    manifest = _build_pod_manifest(req, pod_config(default_image="img:latest"))
    assert "activeDeadlineSeconds" not in manifest["spec"]


def test_zero_timeout_no_deadline():
    req = make_run_req("/my-job/task-0")
    req.timeout.milliseconds = 0
    manifest = _build_pod_manifest(req, pod_config(default_image="img:latest"))
    assert "activeDeadlineSeconds" not in manifest["spec"]


# ---------------------------------------------------------------------------
# Volumes and mounts
# ---------------------------------------------------------------------------


def test_pod_manifest_volumes_and_mounts_are_consistent():
    """No dangling mounts and no orphaned volumes.

    A mount naming an undeclared volume is rejected by the API server outright.
    An orphan is quieter and worse: the volume exists but reaches no container,
    so whatever it backs silently falls through to the container layer.
    """
    req = make_run_req("/test-job/0", attempt_id=1)
    req.bundle_id = "bundle-abc"
    manifest = _build_pod_manifest(req, pod_config(controller_address="http://ctrl:8080"))
    spec = manifest["spec"]

    declared = {v["name"] for v in spec["volumes"]}
    mounted = {m["name"] for c in spec["containers"] + spec.get("initContainers", []) for m in c.get("volumeMounts", [])}

    assert mounted - declared == set(), "volumeMount names a volume the pod does not declare"
    assert declared - mounted == set(), "volume reaches no container"


def test_task_container_does_not_mount_the_log_shipper_host_path():
    """varlogpods stays the sidecar's.

    It is a hostPath onto the node's pod log directory; mounting it into the task
    would hand every task a read of every other pod's logs on that node.
    """
    manifest = _build_pod_manifest(make_run_req("/test-job/0"), pod_config())

    assert "varlogpods" in {v["name"] for v in manifest["spec"]["volumes"]}
    assert "varlogpods" not in {m["name"] for m in manifest["spec"]["containers"][0]["volumeMounts"]}


def test_cache_env_points_at_mounted_cache_volumes():
    """Every cache env var names a path the pod actually mounts.

    The pod spec carries these rather than the task image, so that a task
    bringing its own image writes to the shared cache volumes too.
    """
    manifest = _build_pod_manifest(make_run_req("/test-job/0"), pod_config())
    container = manifest["spec"]["containers"][0]

    env = {e["name"]: e.get("value") for e in container["env"]}
    host_backed = {v["name"] for v in manifest["spec"]["volumes"] if "hostPath" in v}
    cache_mounts = [m["mountPath"] for m in container["volumeMounts"] if m["name"] in host_backed]

    # Each var must resolve inside a node-persistent cache mount. A var pointing
    # anywhere else lands on the container layer and re-downloads every task.
    for var in ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "HF_HUB_CACHE", "CARGO_HOME", "CARGO_TARGET_DIR"):
        value = env[var]
        assert any(
            value == mount or value.startswith(f"{mount}/") for mount in cache_mounts
        ), f"{var}={value} is not under a cache mount ({cache_mounts}); it would land on the container layer"

    # HF_HOME carries the submitter's HF_TOKEN, so it must NOT be redirected onto
    # a node-shared cache directory that every other task on the node can read.
    assert "HF_HOME" not in env


@pytest.mark.parametrize("device", ["gpu", "tpu", None])
def test_shm_is_raised_above_the_docker_default_only_for_accelerators(device):
    """Accelerator pods get a raised /dev/shm; plain CPU pods keep the default.

    Multi-process NCCL and TPU runtimes exchange buffers through /dev/shm and
    fail on the container default (64MB), so the limit tracks the accelerator,
    not the exact ceiling.
    """
    req = make_run_req("/test-job/0")
    if device == "gpu":
        req.resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="A100", count=4))
    elif device == "tpu":
        req.resources.device.tpu.CopyFrom(job_pb2.TpuDevice(variant="v4", count=4))
    manifest = _build_pod_manifest(req, pod_config())

    dshm_volumes = [v for v in manifest["spec"]["volumes"] if v["name"] == "dshm"]
    assert len(dshm_volumes) == 1
    empty_dir = dshm_volumes[0]["emptyDir"]

    # Memory-backed: /dev/shm on disk would silently gut collective throughput.
    assert empty_dir["medium"] == "Memory"

    if device is None:
        assert "sizeLimit" not in empty_dir
    else:
        assert parse_k8s_quantity(empty_dir["sizeLimit"]) > 64 * 1024**2


def test_tpu_adds_sys_resource_capability():
    """TPU pods get SYS_RESOURCE capability for memlock ulimits."""
    req = make_run_req("/test-job/0")
    req.resources.device.tpu.CopyFrom(job_pb2.TpuDevice(variant="v4", count=4))
    manifest = _build_pod_manifest(req, pod_config())

    caps = manifest["spec"]["containers"][0]["securityContext"]["capabilities"]["add"]
    assert "SYS_PTRACE" in caps
    assert "SYS_RESOURCE" in caps


def test_cache_mounts_are_host_backed_and_the_rest_are_not():
    """Only CACHE mounts get a hostPath, and they land under cache_dir.

    hostPath is what makes a cache outlive its pod; an emptyDir here would be
    deleted with the pod and re-downloaded by the next task.
    """
    volumes, _mounts = _build_volumes_and_mounts("/my-cache", has_accelerator=False)
    by_name = {v["name"]: v for v in volumes}

    for mount in STANDARD_MOUNTS:
        volume = by_name[mount.name]
        if mount.kind is MountKind.CACHE:
            assert volume["hostPath"]["path"].startswith("/my-cache/")
            assert volume["hostPath"]["type"] == "DirectoryOrCreate"
        else:
            assert "emptyDir" in volume

    # Distinct host dirs, or two caches would collide on the node.
    host_paths = [v["hostPath"]["path"] for v in volumes if "hostPath" in v]
    assert len(host_paths) == len(set(host_paths))


# ---------------------------------------------------------------------------
# SYS_PTRACE security context
# ---------------------------------------------------------------------------


def test_sys_ptrace_capability():
    """Container gets SYS_PTRACE capability for profiling."""
    req = make_run_req("/my-job/task-0")
    manifest = _build_pod_manifest(req, pod_config())
    container = manifest["spec"]["containers"][0]
    assert "SYS_PTRACE" in container["securityContext"]["capabilities"]["add"]


# ---------------------------------------------------------------------------
# Container security profiles
# ---------------------------------------------------------------------------


def test_default_profile_matches_baseline():
    """UNSPECIFIED resolves to DEFAULT: today's SYS_PTRACE-only context."""
    ctx = _security_context(job_pb2.CONTAINER_PROFILE_UNSPECIFIED, has_tpu=False)
    assert ctx == {"capabilities": {"add": ["SYS_PTRACE"]}}


def test_restricted_profile_drops_all_caps():
    ctx = _security_context(job_pb2.CONTAINER_PROFILE_RESTRICTED, has_tpu=False)
    assert ctx["capabilities"] == {"drop": ["ALL"], "add": []}
    assert ctx["allowPrivilegeEscalation"] is False
    assert ctx["seccompProfile"] == {"type": "RuntimeDefault"}
    assert "privileged" not in ctx


def test_restricted_profile_omits_tpu_cap():
    """RESTRICTED must not leak the SYS_RESOURCE device cap, even on TPU."""
    ctx = _security_context(job_pb2.CONTAINER_PROFILE_RESTRICTED, has_tpu=True)
    assert ctx["capabilities"] == {"drop": ["ALL"], "add": []}


def test_privileged_profile_sets_privileged():
    ctx = _security_context(job_pb2.CONTAINER_PROFILE_PRIVILEGED, has_tpu=False)
    assert ctx["privileged"] is True
    assert ctx["allowPrivilegeEscalation"] is True
    assert "SYS_PTRACE" in ctx["capabilities"]["add"]


def test_docker_access_rejected_on_k8s():
    """DOCKER_ACCESS has no host docker socket on k8s nodes; fail fast."""
    with pytest.raises(ValueError, match="DOCKER_ACCESS is not supported"):
        _security_context(job_pb2.CONTAINER_PROFILE_DOCKER_ACCESS, has_tpu=False)


def test_privileged_profile_applied_to_pod_manifest():
    """A PRIVILEGED RunTaskRequest produces a privileged container securityContext."""
    req = make_run_req("/my-job/task-0")
    req.container_profile = job_pb2.CONTAINER_PROFILE_PRIVILEGED
    manifest = _build_pod_manifest(req, pod_config())
    assert manifest["spec"]["containers"][0]["securityContext"]["privileged"] is True


def test_docker_access_pod_manifest_raises():
    req = make_run_req("/my-job/task-0")
    req.container_profile = job_pb2.CONTAINER_PROFILE_DOCKER_ACCESS
    with pytest.raises(ValueError, match="DOCKER_ACCESS is not supported"):
        _build_pod_manifest(req, pod_config())


def test_gvisor_profile_sets_runtime_class_and_benign_context():
    """GVISOR sets the pod runtimeClassName and a non-privileged securityContext."""
    req = make_run_req("/my-job/task-0")
    req.container_profile = job_pb2.CONTAINER_PROFILE_GVISOR
    manifest = _build_pod_manifest(req, pod_config())
    assert manifest["spec"]["runtimeClassName"] == "gvisor"
    ctx = manifest["spec"]["containers"][0]["securityContext"]
    assert "privileged" not in ctx
    assert ctx["capabilities"]["add"] == ["SYS_PTRACE"]


# ---------------------------------------------------------------------------
# Service account
# ---------------------------------------------------------------------------


def test_service_account_set():
    """serviceAccountName is set in spec when service_account is provided."""
    req = make_run_req("/my-job/task-0")
    manifest = _build_pod_manifest(req, pod_config(service_account="my-sa"))
    assert manifest["spec"]["serviceAccountName"] == "my-sa"


def test_service_account_omitted_when_empty():
    """serviceAccountName is absent from spec when service_account is empty."""
    req = make_run_req("/my-job/task-0")
    manifest = _build_pod_manifest(req, pod_config(service_account=""))
    assert "serviceAccountName" not in manifest["spec"]


# ---------------------------------------------------------------------------
# Host networking
# ---------------------------------------------------------------------------


def test_host_network_mode():
    """hostNetwork and dnsPolicy are set when host_network is enabled."""
    req = make_run_req("/my-job/task-0")
    manifest = _build_pod_manifest(req, pod_config(host_network=True))
    assert manifest["spec"]["hostNetwork"] is True
    assert manifest["spec"]["dnsPolicy"] == "ClusterFirstWithHostNet"


def test_host_network_omitted_when_disabled():
    """hostNetwork and dnsPolicy are absent when host_network is False."""
    req = make_run_req("/my-job/task-0")
    manifest = _build_pod_manifest(req, pod_config(host_network=False))
    assert "hostNetwork" not in manifest["spec"]
    assert "dnsPolicy" not in manifest["spec"]


# ---------------------------------------------------------------------------
# Iris env vars and task script
# ---------------------------------------------------------------------------


def test_iris_env_vars_injected():
    """Pod manifest includes IRIS_TASK_ID, IRIS_NUM_TASKS, and other system vars."""
    req = make_run_req("/test-job/0")
    req.num_tasks = 4
    req.bundle_id = "bundle-abc"
    manifest = _build_pod_manifest(req, pod_config(controller_address="http://ctrl:8080"))

    env_by_name = {e["name"]: e for e in manifest["spec"]["containers"][0]["env"]}
    assert env_by_name["IRIS_TASK_ID"]["value"] == "/test-job/0:0"
    assert env_by_name["IRIS_NUM_TASKS"]["value"] == "4"
    assert env_by_name["IRIS_BUNDLE_ID"]["value"] == "bundle-abc"
    assert env_by_name["IRIS_CONTROLLER_ADDRESS"]["value"] == "http://ctrl:8080"
    assert env_by_name["IRIS_CONTROLLER_URL"]["value"] == "http://ctrl:8080"
    # Tasks must listen on all interfaces: a peer or the controller reaching the
    # pod by IP cannot reach a loopback bind.
    assert env_by_name["IRIS_BIND_HOST"]["value"] == "0.0.0.0"


def test_advertise_host_uses_downward_api():
    """IRIS_ADVERTISE_HOST is populated via the k8s downward API (status.podIP)."""
    req = make_run_req("/test-job/0")
    manifest = _build_pod_manifest(req, pod_config())

    env_by_name = {e["name"]: e for e in manifest["spec"]["containers"][0]["env"]}
    adv = env_by_name["IRIS_ADVERTISE_HOST"]
    assert "valueFrom" in adv
    assert adv["valueFrom"]["fieldRef"]["fieldPath"] == "status.podIP"


def test_device_env_vars_tpu():
    """TPU device resources inject JAX_PLATFORMS, PJRT_DEVICE, JAX_FORCE_TPU_INIT."""
    req = make_run_req("/test-job/0")
    req.resources.device.tpu.CopyFrom(job_pb2.TpuDevice(variant="v4-8", count=4))
    manifest = _build_pod_manifest(req, pod_config())

    env_by_name = {e["name"]: e.get("value") for e in manifest["spec"]["containers"][0]["env"]}
    assert env_by_name["JAX_PLATFORMS"] == "tpu,cpu"
    assert env_by_name["PJRT_DEVICE"] == "TPU"
    assert env_by_name["JAX_FORCE_TPU_INIT"] == "1"


def test_iris_env_overrides_user_env():
    """Iris system vars override user-supplied vars with the same key."""
    req = make_run_req("/test-job/0")
    req.environment.env_vars["IRIS_TASK_ID"] = "wrong-value"
    manifest = _build_pod_manifest(req, pod_config())

    env_by_name = {e["name"]: e.get("value") for e in manifest["spec"]["containers"][0]["env"]}
    assert env_by_name["IRIS_TASK_ID"] == "/test-job/0:0"


def test_task_script_runs_each_setup_command_before_exec():
    """Each setup command runs as its own step, before the run command."""
    req = make_run_req("/test-job/0")
    req.entrypoint.setup_commands.extend(["pip install foo", "export BAR=1"])
    script = _build_task_script(req)
    lines = script.split("\n")
    # render_setup_steps materializes each command to its own file and runs it.
    step_runs = [i for i, l in enumerate(lines) if l.startswith("bash /tmp/iris-setup-step-")]
    exec_idx = next(i for i, l in enumerate(lines) if l.startswith("exec "))
    assert len(step_runs) == 2
    assert max(step_runs) < exec_idx


def test_task_script_exec_run_command():
    """Run command is exec'd as the last line of the task script."""
    req = make_run_req("/test-job/0")
    script = _build_task_script(req)
    lines = script.split("\n")
    assert lines[-1] == "exec python train.py"


def test_build_common_iris_env_no_controller_address():
    """Controller address env vars are omitted when controller_address is None."""
    req = make_run_req("/test-job/0")
    env = common_env_from_req(req, controller_address=None)
    assert "IRIS_CONTROLLER_ADDRESS" not in env
    assert "IRIS_CONTROLLER_URL" not in env
    assert "IRIS_TASK_ID" in env


def test_build_common_iris_env_serializes_user_env_as_iris_job_env():
    """User env vars are serialized into IRIS_JOB_ENV for child job inheritance."""
    req = make_run_req("/test-job/0")
    env = common_env_from_req(req, controller_address=None)
    job_env = json.loads(env["IRIS_JOB_ENV"])
    assert job_env["IRIS_JOB_ID"] == "test-job"


def test_build_common_iris_env_includes_attempt_suffix_on_retry():
    """IRIS_TASK_ID includes :attempt_id suffix for retried tasks."""
    req = make_run_req("/test-job/0", attempt_id=3)
    env = common_env_from_req(req, controller_address=None)
    assert env["IRIS_TASK_ID"] == "/test-job/0:3"


def test_build_common_iris_env_includes_attempt_suffix_for_first_attempt():
    """IRIS_TASK_ID carries the :0 suffix on the first attempt, matching retries."""
    req = make_run_req("/test-job/0", attempt_id=0)
    env = common_env_from_req(req, controller_address=None)
    assert env["IRIS_TASK_ID"] == "/test-job/0:0"


# ---------------------------------------------------------------------------
# Init containers: bundle fetch and workdir files
# ---------------------------------------------------------------------------


def test_init_container_created_when_bundle_id_present():
    """Setting bundle_id + controller_address produces an init container."""
    req = make_run_req("/my-job/task-0")
    req.bundle_id = "bundle-abc"

    init_containers, extra_volumes, configmap_name = _build_init_container_spec(
        req,
        "iris-my-job-task-0-abcd1234-0",
        "myrepo/iris:latest",
        "http://ctrl:8080",
    )

    assert len(init_containers) == 1
    ic = init_containers[0]
    assert ic["name"] == "stage-workdir"
    assert ic["image"] == "myrepo/iris:latest"
    env_by_name = {e["name"]: e["value"] for e in ic["env"]}
    assert env_by_name["IRIS_BUNDLE_ID"] == "bundle-abc"
    assert env_by_name["IRIS_CONTROLLER_URL"] == "http://ctrl:8080"
    assert configmap_name is None
    assert extra_volumes == []


def test_no_init_container_when_no_bundle_or_files():
    """No init containers when neither bundle_id nor workdir_files are set."""
    req = make_run_req("/my-job/task-0")
    req.bundle_id = ""

    init_containers, extra_volumes, configmap_name = _build_init_container_spec(
        req,
        "iris-pod-name",
        "myrepo/iris:latest",
        "http://ctrl:8080",
    )

    assert init_containers == []
    assert extra_volumes == []
    assert configmap_name is None


def test_init_container_for_workdir_files():
    """Workdir files produce a ConfigMap volume and init container with IRIS_WORKDIR_FILES_SRC."""
    req = make_run_req("/my-job/task-0")
    req.entrypoint.workdir_files["config.yaml"] = b"key: value"
    req.entrypoint.workdir_files["sub/data.txt"] = b"hello"

    init_containers, extra_volumes, configmap_name = _build_init_container_spec(
        req,
        "iris-pod-name",
        "myrepo/iris:latest",
        None,
    )

    assert len(init_containers) == 1
    assert configmap_name == "iris-pod-name-wf"
    assert len(extra_volumes) == 1
    assert extra_volumes[0]["name"] == "workdir-files"
    assert extra_volumes[0]["configMap"]["name"] == configmap_name

    ic = init_containers[0]
    env_by_name = {e["name"]: e["value"] for e in ic["env"]}
    assert env_by_name["IRIS_WORKDIR_FILES_SRC"] == "/iris/staged-workdir-files"

    mount_by_name = {m["name"]: m for m in ic["volumeMounts"]}
    assert "workdir-files" in mount_by_name
    assert mount_by_name["workdir-files"]["readOnly"] is True


def test_init_container_bundle_and_workdir_files():
    """Both bundle and workdir files produce a single init container with all env vars."""
    req = make_run_req("/my-job/task-0")
    req.bundle_id = "bundle-xyz"
    req.entrypoint.workdir_files["run.sh"] = b"#!/bin/bash"

    init_containers, extra_volumes, configmap_name = _build_init_container_spec(
        req,
        "iris-pod-name",
        "myrepo/iris:latest",
        "http://ctrl:8080",
    )

    assert len(init_containers) == 1
    ic = init_containers[0]
    env_by_name = {e["name"]: e["value"] for e in ic["env"]}
    assert "IRIS_BUNDLE_ID" in env_by_name
    assert "IRIS_WORKDIR_FILES_SRC" in env_by_name
    assert configmap_name is not None
    assert len(extra_volumes) == 1


def test_init_container_for_workdir_file_refs():
    """Blob refs produce an init container with IRIS_WORKDIR_BLOB_REFS env var."""
    req = make_run_req("/my-job/task-0")
    req.entrypoint.workdir_file_refs["_callable.pkl"] = "abcd1234" * 8

    init_containers, _extra_volumes, configmap_name = _build_init_container_spec(
        req,
        "iris-pod-name",
        "myrepo/iris:latest",
        "http://ctrl:8080",
    )

    assert len(init_containers) == 1
    ic = init_containers[0]
    env_by_name = {e["name"]: e["value"] for e in ic["env"]}
    assert env_by_name["IRIS_CONTROLLER_URL"] == "http://ctrl:8080"
    assert "IRIS_WORKDIR_BLOB_REFS" in env_by_name

    refs = json.loads(env_by_name["IRIS_WORKDIR_BLOB_REFS"])
    assert refs == {"_callable.pkl": "abcd1234" * 8}
    assert configmap_name is None


def test_no_init_container_for_blob_refs_without_controller():
    """Blob refs without controller_address are ignored (no way to fetch)."""
    req = make_run_req("/my-job/task-0")
    req.entrypoint.workdir_file_refs["_callable.pkl"] = "abcd1234" * 8

    init_containers, _extra_volumes, configmap_name = _build_init_container_spec(
        req,
        "iris-pod-name",
        "myrepo/iris:latest",
        None,
    )

    assert init_containers == []
    assert configmap_name is None


def test_init_container_workdir_files_and_blob_refs():
    """Both inline files and blob refs produce ConfigMap + blob ref env var."""
    req = make_run_req("/my-job/task-0")
    req.entrypoint.workdir_files["small.txt"] = b"tiny"
    req.entrypoint.workdir_file_refs["big.pkl"] = "deadbeef" * 8

    init_containers, _extra_volumes, configmap_name = _build_init_container_spec(
        req,
        "iris-pod-name",
        "myrepo/iris:latest",
        "http://ctrl:8080",
    )

    assert len(init_containers) == 1
    ic = init_containers[0]
    env_by_name = {e["name"]: e["value"] for e in ic["env"]}
    assert "IRIS_WORKDIR_FILES_SRC" in env_by_name
    assert "IRIS_WORKDIR_BLOB_REFS" in env_by_name
    assert configmap_name is not None


# ---------------------------------------------------------------------------
# Coordinator detection and PDB manifest
# ---------------------------------------------------------------------------


def test_is_coordinator_single_task_no_accelerator():
    """Single-task CPU-only job is a coordinator."""
    req = make_run_req("/coord-job/0")
    req.num_tasks = 1
    assert _is_coordinator_task(req) is True


def test_is_coordinator_default_num_tasks():
    """Default num_tasks (0) is treated as coordinator."""
    req = make_run_req("/coord-job/0")
    assert _is_coordinator_task(req) is True


def test_is_not_coordinator_multi_task():
    """Multi-task jobs are not coordinators."""
    req = make_run_req("/worker-job/0")
    req.num_tasks = 4
    assert _is_coordinator_task(req) is False


def test_is_not_coordinator_with_gpu():
    """GPU jobs are not coordinators."""
    req = make_run_req("/gpu-job/0")
    req.num_tasks = 1
    req.resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="A100", count=4))
    assert _is_coordinator_task(req) is False


def test_build_pdb_manifest_selector_and_cleanup_labels():
    """PDB selector targets task hash; labels include task hash for label-based cleanup."""
    pdb = _build_pdb_manifest("iris-coord-0-abcd1234-0", "iris", "deadbeef12345678")
    assert pdb["spec"]["selector"]["matchLabels"][_LABEL_TASK_HASH] == "deadbeef12345678"
    assert pdb["metadata"]["labels"][_LABEL_TASK_HASH] == "deadbeef12345678"


# ---------------------------------------------------------------------------
# Kueue gang admission (coscheduled jobs)
# ---------------------------------------------------------------------------


def _cosched_req(task_id: str, attempt_id: int = 0, num_tasks: int = 64, group_by: str = "leafgroup", priority=None):
    if priority is None:
        priority = job_pb2.PRIORITY_BAND_UNSPECIFIED
    return make_run_req(
        task_id,
        attempt_id=attempt_id,
        num_tasks=num_tasks,
        coscheduling_group_by=group_by,
        priority=priority,
    )


def test_kueue_labels_for_coscheduled_pod():
    """Coscheduled pod + configured LocalQueue gets the gang label/annotation set."""
    req = _cosched_req("/job/task/0", num_tasks=64, priority=job_pb2.PRIORITY_BAND_BATCH)
    manifest = _build_pod_manifest(req, pod_config(local_queue="iris-lq"))

    labels = manifest["metadata"]["labels"]
    annotations = manifest["metadata"]["annotations"]
    assert labels[_KUEUE_POD_GROUP_NAME] == _pod_group_name(JobName.from_wire("/job/task/0"), 0)
    assert labels[_KUEUE_QUEUE_NAME] == "iris-lq"
    assert annotations[_KUEUE_POD_GROUP_TOTAL] == "64"


def test_kueue_pod_group_pod_index_from_task_ordinal():
    """Each gang pod carries kueue.x-k8s.io/pod-group-pod-index = its task ordinal so Kueue
    TAS can rank-assign the podset; distinct siblings get distinct indices."""
    m0 = _build_pod_manifest(_cosched_req("/run/task/0", attempt_id=0), pod_config(local_queue="iris-lq"))
    m3 = _build_pod_manifest(_cosched_req("/run/task/3", attempt_id=0), pod_config(local_queue="iris-lq"))
    assert m0["metadata"]["labels"][_KUEUE_POD_GROUP_POD_INDEX] == "0"
    assert m3["metadata"]["labels"][_KUEUE_POD_GROUP_POD_INDEX] == "3"


def test_kueue_priority_class_not_stamped_without_config():
    """With no configured priority-class mapping, pods carry no WorkloadPriorityClass label
    (the cluster's Kueue default applies)."""
    req = _cosched_req("/job/task/0", num_tasks=64, priority=job_pb2.PRIORITY_BAND_BATCH)
    manifest = _build_pod_manifest(req, pod_config(local_queue="iris-lq"))
    assert _KUEUE_PRIORITY_CLASS not in manifest["metadata"]["labels"]


def test_kueue_priority_class_stamped_from_config():
    """A configured band->WorkloadPriorityClass mapping stamps the label for that band."""
    req = _cosched_req("/job/task/0", num_tasks=64, priority=job_pb2.PRIORITY_BAND_BATCH)
    manifest = _build_pod_manifest(
        req,
        pod_config(local_queue="iris-lq", kueue_priority_classes={job_pb2.PRIORITY_BAND_BATCH: "iris-batch"}),
    )
    assert manifest["metadata"]["labels"][_KUEUE_PRIORITY_CLASS] == "iris-batch"


def test_kueue_required_topology_for_nvlink_domain():
    """group_by=nvlink.domain -> required (hard) NVLink-domain topology."""
    manifest = _build_pod_manifest(
        _cosched_req("/job/task/0", num_tasks=8, group_by="nvlink.domain"), pod_config(local_queue="iris-lq")
    )
    annotations = manifest["metadata"]["annotations"]
    assert annotations[_KUEUE_REQUIRED_TOPOLOGY] == "ds.coreweave.com/nvlink.domain"
    assert _KUEUE_PREFERRED_TOPOLOGY not in annotations


def test_kueue_required_nvlink_gang_rejects_above_schedulable_slice():
    """A hard nvlink.domain gang larger than a rack's guaranteed-schedulable slice can hang
    whenever the rack is short a node, so it must fail fast (the guard for a programmatic or
    stale client; the CLI routes 17+ NVL72 replicas to the sliced level, never to a hard gang
    this large)."""
    with pytest.raises(ValueError, match="guaranteed-schedulable rack slice"):
        _build_pod_manifest(
            _cosched_req("/job/task/0", num_tasks=SCHEDULABLE_RACK_NODES + 1, group_by="nvlink.domain"),
            pod_config(local_queue="iris-lq"),
        )


def test_kueue_required_nvlink_gang_allows_schedulable_slice():
    """A hard nvlink.domain gang of exactly the guaranteed-schedulable rack slice
    (SCHEDULABLE_RACK_NODES nodes) is the largest hard single-domain gang and is valid."""
    manifest = _build_pod_manifest(
        _cosched_req("/job/task/0", num_tasks=SCHEDULABLE_RACK_NODES, group_by="nvlink.domain"),
        pod_config(local_queue="iris-lq"),
    )
    assert manifest["metadata"]["annotations"][_KUEUE_REQUIRED_TOPOLOGY] == "ds.coreweave.com/nvlink.domain"


def test_kueue_preferred_nvlink_gang_packs_multi_rack():
    """A multi-rack GB200 gang uses the SOFT nvlink.domain.preferred level: it binds the
    nvlink.domain label as a PREFERRED (not required) topology, so Kueue packs the replicas
    into as few whole NVLink domains as possible instead of demanding one (impossible) domain.
    It is admitted for a gang larger than one rack rather than rejected."""
    manifest = _build_pod_manifest(
        _cosched_req("/job/task/0", num_tasks=RACK_SIZE + 1, group_by="nvlink.domain.preferred"),
        pod_config(local_queue="iris-lq"),
    )
    annotations = manifest["metadata"]["annotations"]
    assert annotations[_KUEUE_PREFERRED_TOPOLOGY] == "ds.coreweave.com/nvlink.domain"
    assert _KUEUE_REQUIRED_TOPOLOGY not in annotations


def _sliced_req(
    task_id: str, num_tasks: int, *, gpu_count: int = NVL72_GPUS_PER_NODE, group_by: str = "nvlink.domain.sliced"
):
    """A coscheduled request on the sliced level with a GB200 GPU device (node-saturating by default)."""
    req = _cosched_req(task_id, num_tasks=num_tasks, group_by=group_by)
    req.resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="GB200", count=gpu_count))
    return req


@pytest.mark.parametrize("num_tasks,slice_size", [(24, 12), (32, 16), (48, 16), (20, 10), (64, 16)])
def test_kueue_sliced_nvlink_gang_stamps_balanced_slice_size(num_tasks, slice_size):
    """A multi-rack GB200 gang on the sliced level binds podset-slice-required-topology to
    nvlink.domain with a podset-slice-size that spreads it evenly over the fewest racks (24->12,
    32->16, 48->16), pairs a soft coarse leafgroup preference, and stamps the per-pod index that
    makes slice membership rank-contiguous. It carries neither the whole-podset required nor a
    preferred nvlink.domain request."""
    manifest = _build_pod_manifest(_sliced_req("/job/task/0", num_tasks=num_tasks), pod_config(local_queue="iris-lq"))
    annotations = manifest["metadata"]["annotations"]
    assert annotations[_KUEUE_SLICE_REQUIRED_TOPOLOGY] == "ds.coreweave.com/nvlink.domain"
    assert annotations[_KUEUE_SLICE_SIZE] == str(slice_size)
    assert annotations[_KUEUE_PREFERRED_TOPOLOGY] == "backend.coreweave.cloud/leafgroup"
    assert _KUEUE_REQUIRED_TOPOLOGY not in annotations
    assert manifest["metadata"]["labels"][_KUEUE_POD_GROUP_POD_INDEX] == "0"


def test_kueue_sliced_gang_rejects_uneven_split():
    """A sliced gang that cannot split into equal per-rack slices (17 over ceil(17/16)=2 racks)
    can't place as a balanced layout, so it is rejected at build time."""
    with pytest.raises(ValueError, match="do not divide evenly"):
        _build_pod_manifest(_sliced_req("/job/task/0", num_tasks=17), pod_config(local_queue="iris-lq"))


def test_kueue_sliced_gang_rejects_slice_too_small():
    """A gang whose balanced slices would each be <= half a rack (18 -> two 9-node slices) lets
    two slices share one rack, breaking one slice per rack, so it is rejected."""
    with pytest.raises(ValueError, match="must exceed half a rack"):
        _build_pod_manifest(_sliced_req("/job/task/0", num_tasks=18), pod_config(local_queue="iris-lq"))


def test_kueue_sliced_gang_requires_node_saturating_pods():
    """The one-slice-per-rack guarantee holds only if each pod fills a whole node; a sub-node
    GB200 pod would let two slices share a rack, so the sliced level rejects it."""
    with pytest.raises(ValueError, match="node-saturating"):
        _build_pod_manifest(
            _sliced_req("/job/task/0", num_tasks=32, gpu_count=1),
            pod_config(local_queue="iris-lq"),
        )


def test_kueue_sliced_gang_without_coarse_preferred_omits_preferred_annotation():
    """A sliced binding whose coarse_preferred_label is unset stamps only the slice request, no
    whole-podset preferred topology."""
    manifest = _build_pod_manifest(
        _sliced_req("/job/task/0", num_tasks=32),
        pod_config(
            local_queue="iris-lq",
            kueue_topologies={
                "nvlink.domain.sliced": KueueTopologyBinding(
                    "ds.coreweave.com/nvlink.domain", TopologyMode.SLICE_REQUIRED
                )
            },
        ),
    )
    annotations = manifest["metadata"]["annotations"]
    assert annotations[_KUEUE_SLICE_REQUIRED_TOPOLOGY] == "ds.coreweave.com/nvlink.domain"
    assert annotations[_KUEUE_SLICE_SIZE] == "16"
    assert _KUEUE_PREFERRED_TOPOLOGY not in annotations


def test_kueue_preferred_topology_for_leafgroup():
    """group_by=leafgroup -> preferred (soft) leafgroup topology."""
    manifest = _build_pod_manifest(_cosched_req("/job/task/0", group_by="leafgroup"), pod_config(local_queue="iris-lq"))
    annotations = manifest["metadata"]["annotations"]
    assert annotations[_KUEUE_PREFERRED_TOPOLOGY] == "backend.coreweave.cloud/leafgroup"
    assert _KUEUE_REQUIRED_TOPOLOGY not in annotations


def test_kueue_unmapped_group_by_raises():
    """An unmapped group_by is a misconfiguration: fail fast rather than gang without a
    topology annotation. group_by must name a topology level the cluster provisioned."""
    with pytest.raises(ValueError, match="no topology mapping"):
        _build_pod_manifest(_cosched_req("/job/task/0", group_by="rack"), pod_config(local_queue="iris-lq"))


def test_kueue_siblings_share_pod_group_name():
    """All siblings of one gang (same job, same attempt) carry one pod-group-name."""
    m0 = _build_pod_manifest(_cosched_req("/run/task/0", attempt_id=0), pod_config(local_queue="iris-lq"))
    m1 = _build_pod_manifest(_cosched_req("/run/task/1", attempt_id=0), pod_config(local_queue="iris-lq"))
    assert m0["metadata"]["labels"][_KUEUE_POD_GROUP_NAME] == m1["metadata"]["labels"][_KUEUE_POD_GROUP_NAME]


def test_kueue_generation_bumps_pod_group_name():
    """A new attempt (gang requeue generation) produces a fresh pod-group-name."""
    m0 = _build_pod_manifest(_cosched_req("/run/task/0", attempt_id=0), pod_config(local_queue="iris-lq"))
    m1 = _build_pod_manifest(_cosched_req("/run/task/0", attempt_id=1), pod_config(local_queue="iris-lq"))
    assert m0["metadata"]["labels"][_KUEUE_POD_GROUP_NAME] != m1["metadata"]["labels"][_KUEUE_POD_GROUP_NAME]


def test_pod_group_name_is_valid_label_value():
    """The derived pod-group-name must fit the 63-char k8s label-value limit even for a
    long job path, since Kueue keys the Workload on this label."""
    name = _pod_group_name(JobName.from_wire("/some/long/job/task/0"), 7)
    assert len(name) <= 63


def test_kueue_gang_drops_active_deadline_seconds():
    """A gang omits activeDeadlineSeconds: k8s counts it from creation, so a gang waiting
    SchedulingGated for the autoscaler could burn the deadline before it runs."""
    req = _cosched_req("/job/task/0")
    req.timeout.milliseconds = 3600_000
    manifest = _build_pod_manifest(req, pod_config(local_queue="iris-lq"))
    assert "activeDeadlineSeconds" not in manifest["spec"]


def test_non_coscheduled_pod_keeps_active_deadline_seconds():
    """A non-coscheduled pod keeps activeDeadlineSeconds even though it routes through Kueue:
    single pods admit quickly, and on a K8s-only cluster this is their only timeout
    enforcement (the controller's execution-timeout scan runs only for worker-daemon backends)."""
    req = make_run_req("/job/task/0", num_tasks=4)
    req.timeout.milliseconds = 3600_000
    manifest = _build_pod_manifest(req, pod_config(local_queue="iris-lq"))
    assert manifest["spec"]["activeDeadlineSeconds"] == 3600


def test_kueue_gang_uses_topology_not_affinity():
    """A Kueue-gated gang carries podset topology and never a podAffinity block."""
    req = _cosched_req("/job/task/0", num_tasks=8, group_by="leafgroup")
    manifest = _build_pod_manifest(req, pod_config(local_queue="iris-lq"))
    assert "affinity" not in manifest["spec"]
    assert _KUEUE_PREFERRED_TOPOLOGY in manifest["metadata"]["annotations"]


def test_non_coscheduled_pod_routed_through_kueue_without_gang_metadata():
    """Every pod routes through Kueue when a LocalQueue is set: a non-coscheduled pod
    carries the queue-name label but none of the gang-only pod-group labels or topology
    annotations."""
    manifest = _build_pod_manifest(make_run_req("/job/task/0", num_tasks=4), pod_config(local_queue="iris-lq"))
    labels = manifest["metadata"]["labels"]
    assert labels[_KUEUE_QUEUE_NAME] == "iris-lq"
    assert _KUEUE_POD_GROUP_NAME not in labels
    assert _KUEUE_POD_GROUP_POD_INDEX not in labels
    assert "annotations" not in manifest["metadata"]


def test_single_pod_gpu_job_routed_through_kueue():
    """A single-pod GPU job (not coscheduled) routes through Kueue so its GPU capacity is
    accounted and preemptible: queue-name label and no gang pod-group metadata, but a soft
    finest-level topology request so the topology-aware cw-ib flavor will admit it (a GPU
    workload with no topology request is rejected by TAS)."""
    req = make_run_req("/gpu-job/task/0", num_tasks=1)
    req.resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="H100", count=8))
    manifest = _build_pod_manifest(req, pod_config(local_queue="iris-lq"))
    labels = manifest["metadata"]["labels"]
    annotations = manifest["metadata"]["annotations"]
    assert labels[_KUEUE_QUEUE_NAME] == "iris-lq"
    assert _KUEUE_POD_GROUP_NAME not in labels
    assert annotations[_KUEUE_PREFERRED_TOPOLOGY] == "kubernetes.io/hostname"
    assert _KUEUE_POD_GROUP_TOTAL not in annotations


def test_single_pod_cpu_job_has_no_topology_annotation():
    """A CPU-only pod routes to the non-TAS cw-cpu flavor, so it must NOT carry a topology
    annotation (Kueue would reject a topology request against a non-topology flavor)."""
    manifest = _build_pod_manifest(make_run_req("/cpu-job/task/0", num_tasks=1), pod_config(local_queue="iris-lq"))
    assert "annotations" not in manifest["metadata"]


def test_kueue_topologies_override_config():
    """A configured topologies mapping overrides the CoreWeave defaults for a group_by."""
    manifest = _build_pod_manifest(
        _cosched_req("/job/task/0", group_by="leafgroup"),
        pod_config(
            local_queue="iris-lq",
            kueue_topologies={"leafgroup": KueueTopologyBinding("rack.example.com/pod", TopologyMode.REQUIRED)},
        ),
    )
    annotations = manifest["metadata"]["annotations"]
    assert annotations[_KUEUE_REQUIRED_TOPOLOGY] == "rack.example.com/pod"
    assert _KUEUE_PREFERRED_TOPOLOGY not in annotations
