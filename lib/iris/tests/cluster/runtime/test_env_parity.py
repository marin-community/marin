# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Conformance tests for build_common_iris_env.

Verifies that both the worker path (task_attempt.build_iris_env) and the k8s
path (_build_pod_manifest) produce the same core set of env vars from identical
RunTaskRequest inputs. This prevents the env-var drift that caused 5 bugs
before the shared function was introduced.
"""

import json

import pytest
from google.protobuf import json_format as jf
from iris.cluster.backends.k8s.tasks import PodConfig, _build_pod_manifest
from iris.cluster.runtime.env import (
    IRIS_SLICE_COUNT,
    IRIS_TASKS_PER_SLICE,
    build_common_iris_env,
    with_slice_topology_env,
)
from iris.rpc import job_pb2


def _make_req(
    task_id: str = "/my-job/task-0",
    attempt_id: int = 0,
    num_tasks: int = 1,
    bundle_id: str = "bundle-abc",
    setup_scripts: list[str] | None = None,
    user_env: dict[str, str] | None = None,
    tpu: bool = False,
    tpu_variant: str = "v4",
    tpu_count: int = 4,
    gpu_count: int = 0,
    ports: list[str] | None = None,
) -> job_pb2.RunTaskRequest:
    req = job_pb2.RunTaskRequest()
    req.task_id = task_id
    req.attempt_id = attempt_id
    req.num_tasks = num_tasks
    req.bundle_id = bundle_id
    req.entrypoint.run_command.argv.extend(["python", "train.py"])
    req.resources.cpu_millicores = 1000
    req.resources.memory_bytes = 4 * 1024**3
    if setup_scripts:
        req.environment.setup_scripts.extend(setup_scripts)
    if user_env:
        for k, v in user_env.items():
            req.environment.env_vars[k] = v
    if tpu:
        req.resources.device.tpu.CopyFrom(job_pb2.TpuDevice(variant=tpu_variant, count=tpu_count))
    if gpu_count:
        req.resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="H100", count=gpu_count))
    if ports:
        req.ports.extend(ports)
    return req


def _common_env(req: job_pb2.RunTaskRequest, controller_address: str | None = None) -> dict[str, str]:
    return build_common_iris_env(
        task_id=req.task_id,
        attempt_id=req.attempt_id,
        num_tasks=req.num_tasks,
        bundle_id=req.bundle_id,
        controller_address=controller_address,
        environment=req.environment,
        constraints=req.constraints,
        ports=req.ports,
        resources=req.resources if req.HasField("resources") else None,
    )


def _k8s_env(req: job_pb2.RunTaskRequest, controller_address: str | None = None) -> dict[str, str]:
    """Extract static env vars from the k8s pod manifest (excludes downward API entries)."""
    config = PodConfig(namespace="test", default_image="img:latest", controller_address=controller_address)
    manifest = _build_pod_manifest(req, config)
    env_list = manifest["spec"]["containers"][0]["env"]
    return {e["name"]: e["value"] for e in env_list if "value" in e}


# ---------------------------------------------------------------------------
# Cross-path parity: k8s static env must be a superset of build_common_iris_env
# ---------------------------------------------------------------------------


_PARITY_CASES = [
    "basic",
    "retry",
    "tpu",
    "tpu_multislice",
    "gpu",
    "setup_scripts",
    "user_env",
    "ports",
    "controller",
]


@pytest.fixture(params=_PARITY_CASES)
def parity_req_and_ctrl(request):
    """Generate (RunTaskRequest, controller_address) pairs for parity tests."""
    case = request.param
    ctrl = None
    if case == "basic":
        req = _make_req()
    elif case == "retry":
        req = _make_req(attempt_id=3)
    elif case == "tpu":
        req = _make_req(tpu=True)
    elif case == "tpu_multislice":
        req = _make_req(tpu=True, tpu_variant="v6e-8", tpu_count=8, num_tasks=2)
    elif case == "gpu":
        req = _make_req(gpu_count=8)
    elif case == "setup_scripts":
        req = _make_req(setup_scripts=["uv sync\n", "echo done\n"])
    elif case == "user_env":
        req = _make_req(user_env={"WANDB_API_KEY": "secret", "MY_FLAG": "1"})
    elif case == "ports":
        req = _make_req(ports=["coordinator", "debug"])
    elif case == "controller":
        req = _make_req()
        ctrl = "http://ctrl:8080"
    else:
        raise ValueError(f"Unknown case: {case}")
    return req, ctrl


def test_k8s_env_contains_all_common_env_keys(parity_req_and_ctrl):
    """Every key produced by build_common_iris_env must appear in the k8s pod manifest."""
    req, ctrl = parity_req_and_ctrl
    common = _common_env(req, ctrl)
    k8s = _k8s_env(req, ctrl)
    missing = set(common) - set(k8s)
    assert not missing, f"Keys in build_common_iris_env but missing from k8s manifest: {missing}"


def test_k8s_env_values_match_common_env(parity_req_and_ctrl):
    """Values for shared keys must be identical between common env and k8s path."""
    req, ctrl = parity_req_and_ctrl
    common = _common_env(req, ctrl)
    k8s = _k8s_env(req, ctrl)
    mismatched = {k: (common[k], k8s[k]) for k in common if k in k8s and common[k] != k8s[k]}
    assert not mismatched, f"Value mismatches between common and k8s: {mismatched}"


# ---------------------------------------------------------------------------
# Unit tests for build_common_iris_env edge cases
# ---------------------------------------------------------------------------


def test_attempt_id_zero_includes_suffix():
    env = _common_env(_make_req(attempt_id=0))
    assert env["IRIS_TASK_ID"] == "/my-job/task-0:0"


def test_attempt_id_nonzero_gets_suffix():
    env = _common_env(_make_req(attempt_id=5))
    assert env["IRIS_TASK_ID"] == "/my-job/task-0:5"


def test_controller_address_omitted_when_none():
    env = _common_env(_make_req(), controller_address=None)
    assert "IRIS_CONTROLLER_ADDRESS" not in env
    assert "IRIS_CONTROLLER_URL" not in env


def test_controller_address_set_when_provided():
    env = _common_env(_make_req(), controller_address="http://ctrl:8080")
    assert env["IRIS_CONTROLLER_ADDRESS"] == "http://ctrl:8080"
    assert env["IRIS_CONTROLLER_URL"] == "http://ctrl:8080"


def test_tpu_device_vars():
    env = _common_env(_make_req(tpu=True))
    assert env["JAX_PLATFORMS"] == "tpu,cpu"
    assert env["PJRT_DEVICE"] == "TPU"
    assert env["JAX_FORCE_TPU_INIT"] == "1"


def test_tpu_single_slice_omits_slice_topology():
    env = _common_env(_make_req(tpu=True, tpu_variant="v6e-4", num_tasks=1))

    assert IRIS_SLICE_COUNT not in env
    assert IRIS_TASKS_PER_SLICE not in env


def test_tpu_multislice_sets_slice_topology_for_single_vm_slices():
    env = _common_env(_make_req(tpu=True, tpu_variant="v6e-8", tpu_count=8, num_tasks=2))

    assert env[IRIS_SLICE_COUNT] == "2"
    assert env[IRIS_TASKS_PER_SLICE] == "1"


def test_tpu_multislice_sets_slice_topology_for_multi_vm_slices():
    env = _common_env(_make_req(tpu=True, tpu_variant="v6e-16", tpu_count=4, num_tasks=8))

    assert env[IRIS_SLICE_COUNT] == "2"
    assert env[IRIS_TASKS_PER_SLICE] == "4"


def test_with_slice_topology_env_publishes_tpu_slice_topology():
    req = _make_req(tpu=True, tpu_variant="v6e-8", tpu_count=8, num_tasks=2)

    environment = with_slice_topology_env(req.environment, req.resources, req.num_tasks)

    assert environment.env_vars[IRIS_SLICE_COUNT] == "2"
    assert environment.env_vars[IRIS_TASKS_PER_SLICE] == "1"


def test_with_slice_topology_env_overwrites_stale_topology():
    req = _make_req(tpu=True, tpu_variant="v6e-8", tpu_count=8, num_tasks=2)
    req.environment.env_vars[IRIS_SLICE_COUNT] = "99"
    req.environment.env_vars[IRIS_TASKS_PER_SLICE] = "99"

    environment = with_slice_topology_env(req.environment, req.resources, req.num_tasks)

    assert environment.env_vars[IRIS_SLICE_COUNT] == "2"
    assert environment.env_vars[IRIS_TASKS_PER_SLICE] == "1"


def test_tpu_multi_vm_single_slice_omits_slice_topology():
    env = _common_env(_make_req(tpu=True, tpu_variant="v6e-16", tpu_count=4, num_tasks=4))

    assert IRIS_SLICE_COUNT not in env
    assert IRIS_TASKS_PER_SLICE not in env


def test_tpu_multislice_rejects_task_count_not_divisible_by_vm_count():
    with pytest.raises(ValueError, match="must be divisible by TPU VM count"):
        _common_env(_make_req(tpu=True, tpu_variant="v6e-16", tpu_count=4, num_tasks=5))


def test_gpu_no_jax_platforms():
    env = _common_env(_make_req(gpu_count=4))
    assert "JAX_PLATFORMS" not in env
    assert "PJRT_DEVICE" not in env


def test_no_device_no_jax_platforms():
    env = _common_env(_make_req())
    assert "JAX_PLATFORMS" not in env


def test_setup_scripts_serialized():
    env = _common_env(_make_req(setup_scripts=["uv sync\n"]))
    assert json.loads(env["IRIS_JOB_SETUP_SCRIPTS"]) == ["uv sync\n"]


def test_user_env_serialized_as_iris_job_env():
    env = _common_env(_make_req(user_env={"FOO": "bar"}))
    assert json.loads(env["IRIS_JOB_ENV"]) == {"FOO": "bar"}


def test_setup_scripts_serialized_when_empty():
    # Always set (even empty) so a child can tell a no-setup parent from a
    # top-level submission with no parent at all.
    env = _common_env(_make_req())
    assert json.loads(env["IRIS_JOB_SETUP_SCRIPTS"]) == []


def test_empty_user_env_omitted():
    env = _common_env(_make_req())
    assert "IRIS_JOB_ENV" not in env


def test_ports_set_to_zero():
    env = _common_env(_make_req(ports=["coordinator", "debug"]))
    assert env["IRIS_PORT_COORDINATOR"] == "0"
    assert env["IRIS_PORT_DEBUG"] == "0"


def test_standard_paths_always_present():
    env = _common_env(_make_req())
    assert env["IRIS_WORKDIR"] == "/app"
    assert env["IRIS_PYTHON"] == "python"
    assert env["IRIS_BIND_HOST"] == "0.0.0.0"
    assert env["UV_PYTHON_INSTALL_DIR"] == "/uv/cache/python"
    assert env["CARGO_TARGET_DIR"] == "/root/.cargo/target"


# ---------------------------------------------------------------------------
# IRIS_TASK_RESOURCES serialization
# ---------------------------------------------------------------------------


def test_task_resources_uses_proto_json():
    """IRIS_TASK_RESOURCES should be valid proto-JSON for ResourceSpecProto."""

    env = _common_env(_make_req())
    raw = env["IRIS_TASK_RESOURCES"]
    proto = job_pb2.ResourceSpecProto()
    jf.Parse(raw, proto)
    assert proto.cpu_millicores == 1000
    assert proto.memory_bytes == 4 * 1024**3


def test_task_resources_includes_gpu_device():
    env = _common_env(_make_req(gpu_count=8))
    proto = job_pb2.ResourceSpecProto()
    jf.Parse(env["IRIS_TASK_RESOURCES"], proto)
    assert proto.device.gpu.count == 8
    assert proto.device.gpu.variant == "H100"


def test_task_resources_includes_tpu_device():
    env = _common_env(_make_req(tpu=True))
    proto = job_pb2.ResourceSpecProto()
    jf.Parse(env["IRIS_TASK_RESOURCES"], proto)
    assert proto.device.tpu.count == 4


def test_task_resources_omits_zero_fields():
    """Zero-valued proto fields should not appear in the JSON, allowing cgroup fallback."""
    env = _common_env(_make_req())
    raw = env["IRIS_TASK_RESOURCES"]
    parsed = json.loads(raw)
    # disk_bytes is 0 by default and should be omitted
    assert "disk_bytes" not in parsed
