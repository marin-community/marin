# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for Kubernetes provider tests."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from iris.cluster.controller.kubernetes_provider import KubernetesProvider, PodConfig
from iris.cluster.controller.transitions import DirectProviderBatch
from iris.cluster.k8s.kubectl import KubectlLogResult
from iris.rpc import cluster_pb2


@pytest.fixture
def mock_kubectl():
    kubectl = MagicMock()
    kubectl.namespace = "iris"
    kubectl.list_json.return_value = []
    kubectl.stream_logs.return_value = KubectlLogResult(lines=[], byte_offset=0)
    kubectl.logs.return_value = ""
    return kubectl


@pytest.fixture
def provider(mock_kubectl):
    return KubernetesProvider(
        kubectl=mock_kubectl,
        namespace="iris",
        default_image="myrepo/iris:latest",
        cache_dir="/cache",
    )


def pod_config(
    namespace: str = "iris",
    default_image: str = "myrepo/iris:latest",
    **kwargs,
) -> PodConfig:
    return PodConfig(namespace=namespace, default_image=default_image, **kwargs)


def make_run_req(task_id: str, attempt_id: int = 0, cpu_mc: int = 1000) -> cluster_pb2.Worker.RunTaskRequest:
    req = cluster_pb2.Worker.RunTaskRequest()
    req.task_id = task_id
    req.attempt_id = attempt_id
    req.entrypoint.run_command.argv.extend(["python", "train.py"])
    req.environment.env_vars["IRIS_JOB_ID"] = "test-job"
    req.resources.cpu_millicores = cpu_mc
    req.resources.memory_bytes = 4 * 1024**3
    return req


def make_batch(
    tasks_to_run=None,
    tasks_to_kill=None,
    running_tasks=None,
) -> DirectProviderBatch:
    return DirectProviderBatch(
        running_tasks=running_tasks or [],
        tasks_to_run=tasks_to_run or [],
        tasks_to_kill=tasks_to_kill or [],
    )


def make_pod(name: str, phase: str, exit_code: int | None = None, reason: str = "") -> dict:
    pod: dict = {
        "metadata": {"name": name},
        "status": {"phase": phase, "containerStatuses": []},
    }
    if exit_code is not None:
        pod["status"]["containerStatuses"] = [
            {
                "state": {
                    "terminated": {
                        "exitCode": exit_code,
                        "reason": reason,
                    }
                }
            }
        ]
    return pod


def add_eq_constraint(req: cluster_pb2.Worker.RunTaskRequest, key: str, value: str) -> None:
    """Add an EQ string constraint to a RunTaskRequest."""
    c = req.constraints.add()
    c.key = key
    c.op = cluster_pb2.CONSTRAINT_OP_EQ
    c.value.string_value = value


def common_env_from_req(
    req: cluster_pb2.Worker.RunTaskRequest,
    controller_address: str | None = None,
) -> dict[str, str]:
    """Call build_common_iris_env with fields extracted from a RunTaskRequest."""
    from iris.cluster.runtime.env import build_common_iris_env

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


def completed_process(stdout="", stderr="", returncode=0) -> subprocess.CompletedProcess[str]:
    """Build a fake subprocess.CompletedProcess for mocking kubectl."""
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)
