# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for pod manifest building: naming, env vars, volumes, constraints, init containers."""

import json

import pytest
from iris.cluster.controller.transitions import RunningTaskEntry
from iris.cluster.providers.k8s.tasks import (
    _INFRASTRUCTURE_FAILURE_REASONS,
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
    _pod_name,
    _sanitize_label_value,
    _task_hash,
    _task_update_from_pod,
)
from iris.cluster.providers.k8s.types import parse_k8s_quantity
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

    resources = container["resources"]["limits"]
    assert resources["cpu"] == "1000m"
    assert resources["memory"] == str(4 * 1024**3)


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


def test_build_pod_manifest_gpu():
    req = make_run_req("/test-job/0")
    req.resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="A100", count=4))
    manifest = _build_pod_manifest(req, pod_config())
    limits = manifest["spec"]["containers"][0]["resources"]["limits"]
    assert limits["nvidia.com/gpu"] == "4"


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


def test_is_infrastructure_failure_with_pod_level_reason():
    """Pod-level eviction (no container statuses) is detected as infrastructure failure."""
    pod: dict = {
        "metadata": {"name": "test"},
        "status": {"phase": "Failed", "reason": "Evicted", "containerStatuses": []},
    }
    assert _is_infrastructure_failure(pod)


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
    """GPU pods get NVIDIA GPU toleration."""
    req = make_run_req("/my-job/task-0")
    req.resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="A100", count=4))

    manifest = _build_pod_manifest(req, pod_config())
    tolerations = manifest["spec"].get("tolerations", [])
    toleration_keys = {t.get("key") for t in tolerations}
    assert "nvidia.com/gpu" in toleration_keys
    assert "qos.coreweave.cloud/interruptable" not in toleration_keys


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
    assert not any(t.get("key") == "qos.coreweave.cloud/interruptable" for t in spec["tolerations"])


# ---------------------------------------------------------------------------
# Rack-level colocation (pod affinity for multi-task jobs)
# ---------------------------------------------------------------------------


def test_build_pod_manifest_single_task_no_affinity():
    """Single-task jobs get no podAffinity (no IB colocation needed)."""
    req = make_run_req("/my-job/task-0", attempt_id=1)
    req.num_tasks = 1
    manifest = _build_pod_manifest(
        req, pod_config(default_image="img:latest", colocation_topology_key="coreweave.cloud/spine")
    )
    assert "affinity" not in manifest["spec"]


def test_build_pod_manifest_multi_task_adds_pod_affinity():
    """Multi-task jobs get podAffinity for IB colocation on same spine."""
    req = make_run_req("/my-job/task-0", attempt_id=1)
    req.num_tasks = 2
    manifest = _build_pod_manifest(
        req, pod_config(default_image="img:latest", colocation_topology_key="coreweave.cloud/spine")
    )
    affinity = manifest["spec"]["affinity"]
    pod_affinity = affinity["podAffinity"]
    terms = pod_affinity["preferredDuringSchedulingIgnoredDuringExecution"]
    assert len(terms) == 1
    term = terms[0]
    assert term["weight"] == 100
    assert term["podAffinityTerm"]["topologyKey"] == "coreweave.cloud/spine"
    labels = term["podAffinityTerm"]["labelSelector"]["matchLabels"]
    assert _LABEL_JOB_ID in labels


def test_build_pod_manifest_multi_task_no_topology_key_no_affinity():
    """Empty colocation_topology_key disables affinity even for multi-task jobs."""
    req = make_run_req("/my-job/task-0", attempt_id=1)
    req.num_tasks = 2
    manifest = _build_pod_manifest(req, pod_config(default_image="img:latest", colocation_topology_key=""))
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


def test_build_pod_manifest_includes_standard_volumes():
    """Pod manifest includes all 5 standard volumes plus dshm (6 total)."""
    req = make_run_req("/test-job/0", attempt_id=1)
    manifest = _build_pod_manifest(req, pod_config())

    spec = manifest["spec"]
    container = spec["containers"][0]

    volume_names = {v["name"] for v in spec["volumes"]}
    mount_names = {m["name"] for m in container["volumeMounts"]}
    expected_names = {"workdir", "tmpfs", "uv-cache", "cargo-registry", "cargo-target", "dshm"}
    assert volume_names == expected_names
    assert mount_names == expected_names

    mount_paths = {m["mountPath"] for m in container["volumeMounts"]}
    assert "/app" in mount_paths
    assert "/tmp" in mount_paths
    assert "/uv/cache" in mount_paths
    assert "/dev/shm" in mount_paths

    assert container["workingDir"] == "/app"


def test_build_pod_manifest_shm_size_limit_with_gpu():
    """dshm volume gets sizeLimit=100Gi when GPU resources are requested."""
    req = make_run_req("/test-job/0")
    req.resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="A100", count=4))
    manifest = _build_pod_manifest(req, pod_config())

    dshm_volumes = [v for v in manifest["spec"]["volumes"] if v["name"] == "dshm"]
    assert len(dshm_volumes) == 1
    assert dshm_volumes[0]["emptyDir"]["medium"] == "Memory"
    assert dshm_volumes[0]["emptyDir"]["sizeLimit"] == "100Gi"


def test_build_pod_manifest_shm_no_size_limit_without_gpu():
    """dshm volume has no sizeLimit when no GPU is requested."""
    req = make_run_req("/test-job/0")
    manifest = _build_pod_manifest(req, pod_config())

    dshm_volumes = [v for v in manifest["spec"]["volumes"] if v["name"] == "dshm"]
    assert len(dshm_volumes) == 1
    assert dshm_volumes[0]["emptyDir"]["medium"] == "Memory"
    assert "sizeLimit" not in dshm_volumes[0]["emptyDir"]


def test_build_pod_manifest_shm_size_limit_with_tpu():
    """dshm volume gets sizeLimit=100Gi when TPU resources are requested."""
    req = make_run_req("/test-job/0")
    req.resources.device.tpu.CopyFrom(job_pb2.TpuDevice(variant="v4", count=4))
    manifest = _build_pod_manifest(req, pod_config())

    dshm_volumes = [v for v in manifest["spec"]["volumes"] if v["name"] == "dshm"]
    assert len(dshm_volumes) == 1
    assert dshm_volumes[0]["emptyDir"]["sizeLimit"] == "100Gi"


def test_tpu_adds_sys_resource_capability():
    """TPU pods get SYS_RESOURCE capability for memlock ulimits."""
    req = make_run_req("/test-job/0")
    req.resources.device.tpu.CopyFrom(job_pb2.TpuDevice(variant="v4", count=4))
    manifest = _build_pod_manifest(req, pod_config())

    caps = manifest["spec"]["containers"][0]["securityContext"]["capabilities"]["add"]
    assert "SYS_PTRACE" in caps
    assert "SYS_RESOURCE" in caps


def test_build_volumes_and_mounts_cache_uses_host_path():
    """Cache volumes use hostPath with DirectoryOrCreate under the given cache_dir."""
    volumes, _mounts = _build_volumes_and_mounts("/my-cache", has_accelerator=False)
    cache_volumes = [v for v in volumes if "hostPath" in v]
    assert len(cache_volumes) == 3
    for v in cache_volumes:
        assert v["hostPath"]["path"].startswith("/my-cache/")
        assert v["hostPath"]["type"] == "DirectoryOrCreate"


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
    assert env_by_name["IRIS_TASK_ID"]["value"] == "/test-job/0"
    assert env_by_name["IRIS_NUM_TASKS"]["value"] == "4"
    assert env_by_name["IRIS_BUNDLE_ID"]["value"] == "bundle-abc"
    assert env_by_name["IRIS_CONTROLLER_ADDRESS"]["value"] == "http://ctrl:8080"
    assert env_by_name["IRIS_CONTROLLER_URL"]["value"] == "http://ctrl:8080"
    assert env_by_name["IRIS_BIND_HOST"]["value"] == "0.0.0.0"
    assert env_by_name["IRIS_WORKDIR"]["value"] == "/app"
    assert env_by_name["IRIS_PYTHON"]["value"] == "python"
    assert env_by_name["UV_PYTHON_INSTALL_DIR"]["value"] == "/uv/cache/python"
    assert env_by_name["CARGO_TARGET_DIR"]["value"] == "/root/.cargo/target"


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
    assert env_by_name["IRIS_TASK_ID"] == "/test-job/0"


def test_task_script_includes_setup_commands():
    """Setup commands appear in the task script before the run command."""
    req = make_run_req("/test-job/0")
    req.entrypoint.setup_commands.extend(["pip install foo", "export BAR=1"])
    script = _build_task_script(req)
    lines = script.split("\n")
    assert "set -e" in lines
    assert "pip install foo" in lines
    assert "export BAR=1" in lines
    setup_idx = lines.index("pip install foo")
    exec_idx = next(i for i, l in enumerate(lines) if l.startswith("exec "))
    assert setup_idx < exec_idx


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


def test_build_common_iris_env_no_attempt_suffix_for_first_attempt():
    """IRIS_TASK_ID has no suffix when attempt_id is 0."""
    req = make_run_req("/test-job/0", attempt_id=0)
    env = common_env_from_req(req, controller_address=None)
    assert env["IRIS_TASK_ID"] == "/test-job/0"


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
    assert env_by_name["IRIS_WORKDIR"] == "/app"
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
