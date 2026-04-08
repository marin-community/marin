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

from iris.cluster.providers.k8s.tasks import PodConfig, _build_pod_manifest
from iris.cluster.runtime.env import build_common_iris_env
from iris.rpc import job_pb2


def _make_req(
    task_id: str = "/my-job/task-0",
    attempt_id: int = 0,
    num_tasks: int = 1,
    bundle_id: str = "bundle-abc",
    extras: list[str] | None = None,
    pip_packages: list[str] | None = None,
    user_env: dict[str, str] | None = None,
    tpu: bool = False,
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
    if extras:
        req.environment.extras.extend(extras)
    if pip_packages:
        req.environment.pip_packages.extend(pip_packages)
    if user_env:
        for k, v in user_env.items():
            req.environment.env_vars[k] = v
    if tpu:
        req.resources.device.tpu.CopyFrom(job_pb2.TpuDevice(variant="v4", count=4))
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
    "gpu",
    "extras",
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
    elif case == "gpu":
        req = _make_req(gpu_count=8)
    elif case == "extras":
        req = _make_req(extras=["tpu", "eval"], pip_packages=["torch", "jax"])
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


def test_attempt_id_zero_no_suffix():
    env = _common_env(_make_req(attempt_id=0))
    assert env["IRIS_TASK_ID"] == "/my-job/task-0"


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


def test_gpu_no_jax_platforms():
    env = _common_env(_make_req(gpu_count=4))
    assert "JAX_PLATFORMS" not in env
    assert "PJRT_DEVICE" not in env


def test_no_device_no_jax_platforms():
    env = _common_env(_make_req())
    assert "JAX_PLATFORMS" not in env


def test_extras_serialized():
    env = _common_env(_make_req(extras=["tpu", "eval"]))
    assert json.loads(env["IRIS_JOB_EXTRAS"]) == ["tpu", "eval"]


def test_pip_packages_serialized():
    env = _common_env(_make_req(pip_packages=["torch"]))
    assert json.loads(env["IRIS_JOB_PIP_PACKAGES"]) == ["torch"]


def test_user_env_serialized_as_iris_job_env():
    env = _common_env(_make_req(user_env={"FOO": "bar"}))
    assert json.loads(env["IRIS_JOB_ENV"]) == {"FOO": "bar"}


def test_empty_extras_omitted():
    env = _common_env(_make_req())
    assert "IRIS_JOB_EXTRAS" not in env


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
    from google.protobuf import json_format as jf

    env = _common_env(_make_req())
    raw = env["IRIS_TASK_RESOURCES"]
    proto = job_pb2.ResourceSpecProto()
    jf.Parse(raw, proto)
    assert proto.cpu_millicores == 1000
    assert proto.memory_bytes == 4 * 1024**3


def test_task_resources_includes_gpu_device():
    from google.protobuf import json_format as jf

    env = _common_env(_make_req(gpu_count=8))
    proto = job_pb2.ResourceSpecProto()
    jf.Parse(env["IRIS_TASK_RESOURCES"], proto)
    assert proto.device.gpu.count == 8
    assert proto.device.gpu.variant == "H100"


def test_task_resources_includes_tpu_device():
    from google.protobuf import json_format as jf

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
