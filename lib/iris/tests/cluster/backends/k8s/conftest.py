# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from finelog.rpc.logging_connect import LogServiceClientSync
from iris.cluster.backends.k8s.tasks import (
    _LABEL_MANAGED,
    _LABEL_RUNTIME,
    _RUNTIME_LABEL_VALUE,
    K8sTaskProvider,
    PodConfig,
)
from iris.cluster.controller.reads import ControlSnapshot
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.types import K8sResource
from iris.cluster.runtime.env import build_common_iris_env
from iris.rpc import job_pb2


class FakeStatsTable:
    """Records every Table.write call so tests can assert on emitted rows."""

    def __init__(self) -> None:
        self.writes: list[list[object]] = []

    def write(self, rows) -> None:
        self.writes.append(list(rows))


@pytest.fixture
def k8s() -> InMemoryK8sService:
    return InMemoryK8sService(namespace="iris")


@pytest.fixture
def log_service(embedded_log_server) -> LogServiceClientSync:
    return LogServiceClientSync(address=embedded_log_server.address)


@pytest.fixture
def task_stats_table() -> FakeStatsTable:
    return FakeStatsTable()


@pytest.fixture
def provider(k8s, task_stats_table):
    p = K8sTaskProvider(
        kubectl=k8s,
        namespace="iris",
        default_image="myrepo/iris:latest",
        cache_dir="/cache",
        task_stats_table=task_stats_table,
        resource_poll_interval=0.05,
        cluster_scan_interval=0.0,
    )
    yield p
    p.close()


@pytest.fixture
def kueue_provider(k8s):
    """K8sTaskProvider with Kueue gang admission enabled (a configured LocalQueue)."""
    p = make_kueue_provider(k8s)
    yield p
    p.close()


def pod_config(
    namespace: str = "iris",
    default_image: str = "myrepo/iris:latest",
    **kwargs,
) -> PodConfig:
    return PodConfig(namespace=namespace, default_image=default_image, **kwargs)


def make_run_req(
    task_id: str,
    attempt_id: int = 0,
    cpu_mc: int = 1000,
    num_tasks: int = 0,
    coscheduling_group_by: str = "",
    priority: int = job_pb2.PRIORITY_BAND_UNSPECIFIED,
) -> job_pb2.RunTaskRequest:
    req = job_pb2.RunTaskRequest()
    req.task_id = task_id
    req.attempt_id = attempt_id
    req.num_tasks = num_tasks
    req.entrypoint.run_command.argv.extend(["python", "train.py"])
    req.environment.env_vars["IRIS_JOB_ID"] = "test-job"
    req.resources.cpu_millicores = cpu_mc
    req.resources.memory_bytes = 4 * 1024**3
    if coscheduling_group_by:
        req.coscheduling.group_by = coscheduling_group_by
    req.priority = priority
    return req


def make_kueue_provider(k8s, *, local_queue: str = "iris-lq", **kwargs) -> K8sTaskProvider:
    """K8sTaskProvider with Kueue gang admission enabled (a configured LocalQueue)."""
    kwargs.setdefault("cluster_scan_interval", 0.0)
    return K8sTaskProvider(
        kubectl=k8s,
        namespace="iris",
        default_image="myrepo/iris:latest",
        cache_dir="/cache",
        local_queue=local_queue,
        **kwargs,
    )


def make_batch(
    tasks_to_run=None,
    running_tasks=None,
) -> ControlSnapshot:
    return ControlSnapshot(
        worker_addresses={},
        reconcile_rows=[],
        timeout_rows=[],
        running_tasks=running_tasks or [],
        tasks_to_run=tasks_to_run or [],
    )


def make_pod(name: str, phase: str, exit_code: int | None = None, reason: str = "", message: str = "") -> dict:
    pod: dict = {
        "metadata": {"name": name},
        "status": {"phase": phase, "containerStatuses": []},
    }
    if exit_code is not None:
        terminated: dict = {"exitCode": exit_code, "reason": reason}
        if message:
            terminated["message"] = message
        pod["status"]["containerStatuses"] = [{"state": {"terminated": terminated}}]
    return pod


def populate_pod(
    k8s: InMemoryK8sService,
    name: str,
    phase: str,
    exit_code: int | None = None,
    reason: str = "",
    labels: dict[str, str] | None = None,
) -> None:
    """Insert a pod manifest into InMemoryK8sService with correct Iris labels."""
    base_labels = {
        _LABEL_MANAGED: "true",
        _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE,
    }
    if labels:
        base_labels.update(labels)
    pod = make_pod(name, phase, exit_code=exit_code, reason=reason)
    pod["kind"] = "Pod"
    pod["metadata"]["labels"] = base_labels
    k8s.seed_resource(K8sResource.PODS, name, pod)


def populate_node(
    k8s: InMemoryK8sService,
    name: str,
    cpu: str = "4",
    memory: str = "8Gi",
    taints: list[dict] | None = None,
) -> None:
    """Insert a Node manifest into InMemoryK8sService."""
    node = {
        "kind": "Node",
        "metadata": {"name": name},
        "spec": {"taints": taints or []},
        "status": {"allocatable": {"cpu": cpu, "memory": memory}},
    }
    k8s.seed_resource(K8sResource.NODES, name, node)


def add_eq_constraint(req: job_pb2.RunTaskRequest, key: str, value: str) -> None:
    """Add an EQ string constraint to a RunTaskRequest."""
    c = req.constraints.add()
    c.key = key
    c.op = job_pb2.CONSTRAINT_OP_EQ
    c.value.string_value = value


def common_env_from_req(
    req: job_pb2.RunTaskRequest,
    controller_address: str | None = None,
) -> dict[str, str]:
    """Call build_common_iris_env with fields extracted from a RunTaskRequest."""
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
