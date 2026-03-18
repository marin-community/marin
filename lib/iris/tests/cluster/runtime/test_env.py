# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared env-building helpers in iris.cluster.runtime.env.

Includes conformance tests that verify Docker, process, and K8s runtimes
produce the same Iris env var keys for the same ContainerConfig.
"""

from __future__ import annotations

import json
import subprocess

from iris.cluster.runtime.env import build_common_iris_env, build_device_env_vars
from iris.cluster.runtime.types import ContainerConfig
from iris.rpc import cluster_pb2


def _make_entrypoint(argv: list[str] | None = None) -> cluster_pb2.RuntimeEntrypoint:
    ep = cluster_pb2.RuntimeEntrypoint()
    ep.run_command.CopyFrom(cluster_pb2.CommandEntrypoint(argv=argv or ["python", "-c", "1"]))
    return ep


def _make_tpu_config() -> ContainerConfig:
    device = cluster_pb2.DeviceConfig(tpu=cluster_pb2.TpuDevice(variant="v4", count=4))
    resources = cluster_pb2.ResourceSpecProto(device=device)
    return ContainerConfig(
        image="task:latest",
        entrypoint=_make_entrypoint(),
        env={"USER_VAR": "hello"},
        resources=resources,
    )


# -- build_common_iris_env unit tests --


def test_build_common_iris_env_core_keys():
    env = build_common_iris_env(
        task_id_wire="/job/0:1",
        num_tasks=4,
        bundle_id="abc123",
        worker_id="w-1",
        controller_address="localhost:9000",
        advertise_host="10.0.0.5",
    )
    assert env["IRIS_TASK_ID"] == "/job/0:1"
    assert env["IRIS_NUM_TASKS"] == "4"
    assert env["IRIS_BUNDLE_ID"] == "abc123"
    assert env["IRIS_WORKER_ID"] == "w-1"
    assert env["IRIS_CONTROLLER_ADDRESS"] == "localhost:9000"
    assert env["IRIS_CONTROLLER_URL"] == "localhost:9000"
    assert env["IRIS_ADVERTISE_HOST"] == "10.0.0.5"
    assert env["IRIS_BIND_HOST"] == "0.0.0.0"
    assert env["IRIS_WORKDIR"] == "/app"
    assert env["IRIS_PYTHON"] == "python"


def test_build_common_iris_env_optional_fields_omitted():
    env = build_common_iris_env(
        task_id_wire="/job/0",
        num_tasks=1,
        bundle_id="",
    )
    assert "IRIS_WORKER_ID" not in env
    assert "IRIS_CONTROLLER_ADDRESS" not in env
    assert "IRIS_ADVERTISE_HOST" not in env
    assert "IRIS_JOB_EXTRAS" not in env
    assert "IRIS_JOB_PIP_PACKAGES" not in env
    assert "IRIS_JOB_ENV" not in env
    assert "IRIS_JOB_CONSTRAINTS" not in env


def test_build_common_iris_env_extras_and_pip_packages():
    env = build_common_iris_env(
        task_id_wire="/job/0",
        num_tasks=1,
        bundle_id="",
        extras=["vllm", "flash-attn"],
        pip_packages=["numpy==1.26"],
    )
    assert json.loads(env["IRIS_JOB_EXTRAS"]) == ["vllm", "flash-attn"]
    assert json.loads(env["IRIS_JOB_PIP_PACKAGES"]) == ["numpy==1.26"]


def test_build_common_iris_env_user_env_vars():
    env = build_common_iris_env(
        task_id_wire="/job/0",
        num_tasks=1,
        bundle_id="",
        user_env_vars={"WANDB_PROJECT": "test"},
    )
    assert json.loads(env["IRIS_JOB_ENV"]) == {"WANDB_PROJECT": "test"}


def test_build_common_iris_env_ports():
    env = build_common_iris_env(
        task_id_wire="/job/0",
        num_tasks=1,
        bundle_id="",
        ports={"grpc": 50051, "http": 8080},
    )
    assert env["IRIS_PORT_GRPC"] == "50051"
    assert env["IRIS_PORT_HTTP"] == "8080"


def test_build_common_iris_env_constraints():
    c = cluster_pb2.Constraint(
        key="region",
        op=cluster_pb2.CONSTRAINT_OP_EQ,
        value=cluster_pb2.AttributeValue(string_value="us-central1"),
    )
    env = build_common_iris_env(
        task_id_wire="/job/0",
        num_tasks=1,
        bundle_id="",
        constraints=[c],
    )
    parsed = json.loads(env["IRIS_JOB_CONSTRAINTS"])
    assert len(parsed) == 1
    assert parsed[0]["key"] == "region"


# -- build_device_env_vars tests --


def test_build_device_env_vars_tpu():
    config = _make_tpu_config()
    env = build_device_env_vars(config)
    assert env["JAX_PLATFORMS"] == "tpu,cpu"
    assert env["PJRT_DEVICE"] == "TPU"
    assert env["JAX_FORCE_TPU_INIT"] == "1"


def test_build_device_env_vars_no_resources():
    config = ContainerConfig(
        image="task:latest",
        entrypoint=_make_entrypoint(),
        env={},
    )
    env = build_device_env_vars(config)
    assert env == {}


# -- Cross-runtime conformance tests --


def _k8s_env_keys(monkeypatch, config: ContainerConfig) -> set[str]:
    """Extract the env var keys that the K8s runtime would set on a task Pod."""
    manifests: list[dict] = []

    def fake_run(cmd, input_data=None, capture_output=False, text=False, check=False, timeout=None, **kwargs):
        if "input" in kwargs:
            input_data = kwargs["input"]
        if input_data:
            manifests.append(json.loads(input_data))
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    from iris.cluster.runtime.kubernetes import KubernetesRuntime

    runtime = KubernetesRuntime(namespace="iris")
    handle = runtime.create_container(config)
    handle.run()

    pod = next(m for m in manifests if m.get("kind") == "Pod")
    return {e["name"] for e in pod["spec"]["containers"][0]["env"]}


def _docker_process_env_keys(config: ContainerConfig) -> set[str]:
    """Env var keys that Docker/process runtimes produce."""
    return {*build_device_env_vars(config), *config.env}


def test_k8s_and_docker_produce_same_iris_env_keys(monkeypatch):
    """Both runtimes must inject the same set of env var keys for identical config."""
    config = _make_tpu_config()
    # Populate config.env with what task_attempt.py would put there
    config.env.update(
        build_common_iris_env(
            task_id_wire="/job/0:1",
            num_tasks=1,
            bundle_id="abc",
            advertise_host="10.0.0.1",
        )
    )

    k8s_keys = _k8s_env_keys(monkeypatch, config)
    docker_keys = _docker_process_env_keys(config)

    # K8s replaces IRIS_ADVERTISE_HOST with downward API (still present as key)
    assert "IRIS_ADVERTISE_HOST" in k8s_keys
    assert "IRIS_ADVERTISE_HOST" in docker_keys

    # Core Iris keys must be present in both
    core_keys = {
        "IRIS_TASK_ID",
        "IRIS_NUM_TASKS",
        "IRIS_BUNDLE_ID",
        "IRIS_BIND_HOST",
        "IRIS_WORKDIR",
        "IRIS_PYTHON",
    }
    assert core_keys <= k8s_keys, f"K8s missing: {core_keys - k8s_keys}"
    assert core_keys <= docker_keys, f"Docker missing: {core_keys - docker_keys}"

    # TPU device env vars must be in both
    tpu_keys = {"JAX_PLATFORMS", "PJRT_DEVICE", "JAX_FORCE_TPU_INIT"}
    assert tpu_keys <= k8s_keys, f"K8s missing TPU keys: {tpu_keys - k8s_keys}"
    assert tpu_keys <= docker_keys, f"Docker missing TPU keys: {tpu_keys - docker_keys}"

    # Symmetric difference should only contain backend-specific keys
    # (none expected for now)
    diff = k8s_keys.symmetric_difference(docker_keys)
    assert diff == set(), f"Env key drift between K8s and Docker: {diff}"
