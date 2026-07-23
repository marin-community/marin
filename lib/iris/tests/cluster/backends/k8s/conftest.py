# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iris.cluster.backends.k8s.tasks import (
    _KUEUE_POD_GROUP_NAME,
    _LABEL_MANAGED,
    _LABEL_RUNTIME,
    _RUNTIME_LABEL_VALUE,
    _TASK_CONTAINER_NAME,
    K8sTaskProvider,
    PodConfig,
)
from iris.cluster.controller.reads import ControlSnapshot
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.types import K8sResource
from iris.cluster.runtime.env import build_common_iris_env
from iris.rpc import job_pb2
from iris.test_util import FakeStatsTable


@pytest.fixture
def k8s() -> InMemoryK8sService:
    return InMemoryK8sService(namespace="iris")


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
        # Kueue is mandatory on the K8s backend, so every provider carries a LocalQueue.
        local_queue="iris-lq",
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
    local_queue: str = "iris-lq",
    **kwargs,
) -> PodConfig:
    # Kueue is mandatory on the K8s backend, so a LocalQueue is always configured.
    return PodConfig(namespace=namespace, default_image=default_image, local_queue=local_queue, **kwargs)


def make_run_req(
    task_id: str,
    attempt_id: int = 0,
    cpu_mc: int = 1000,
    num_tasks: int = 0,
    coscheduling_group_by: str = "",
    priority: int = job_pb2.PRIORITY_BAND_UNSPECIFIED,
    attempt_uid: str = "",
) -> job_pb2.RunTaskRequest:
    req = job_pb2.RunTaskRequest()
    req.task_id = task_id
    req.attempt_id = attempt_id
    req.attempt_uid = attempt_uid
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


# A Kueue admission message for an over-large GPU request (cpu=160 on 128-vCPU
# H100 nodes under InfiniBand TAS): the whole pod cannot fit one node, so every
# node is excluded and the workload never reserves quota. The motivating incident
# for the status_message / iris.task_event diagnostics.
KUEUE_UNADMITTED_MSG = (
    "couldn't assign flavors to pod set main: topology \"infiniband\" doesn't allow to "
    'fit any of 1 pod(s). Total nodes: 32; excluded: resource "cpu": 32'
)


def gated_pod(name: str = "iris-job-0-0", pod_group: str = "wl-abc") -> dict:
    """A Pending pod blocked on a Kueue scheduling gate (no container has started)."""
    return {
        "metadata": {"name": name, "labels": {_KUEUE_POD_GROUP_NAME: pod_group}},
        "status": {
            "phase": "Pending",
            "containerStatuses": [],
            "conditions": [
                {
                    "type": "PodScheduled",
                    "status": "False",
                    "reason": "SchedulingGated",
                    "message": "Scheduling is blocked due to non-empty scheduling gates",
                }
            ],
        },
    }


def unadmitted_workload(name: str = "wl-abc", msg: str = KUEUE_UNADMITTED_MSG) -> dict:
    """A Workload Kueue has evaluated and declined: QuotaReserved=False with a reason."""
    return {
        "metadata": {"name": name},
        "spec": {"queueName": "cw-use02a-lq"},
        "status": {"conditions": [{"type": "QuotaReserved", "status": "False", "reason": "Pending", "message": msg}]},
    }


def unevaluated_workload(name: str = "wl-abc") -> dict:
    """A Workload Kueue has not yet ruled on — no QuotaReserved condition."""
    return {"metadata": {"name": name}, "spec": {"queueName": "cw-use02a-lq"}, "status": {}}


def imagepull_pod(name: str = "iris-job-0-0") -> dict:
    """A Pending pod whose task container is stuck in ImagePullBackOff."""
    return {
        "metadata": {"name": name},
        "status": {
            "phase": "Pending",
            "containerStatuses": [
                {
                    "name": _TASK_CONTAINER_NAME,
                    "state": {
                        "waiting": {"reason": "ImagePullBackOff", "message": 'Back-off pulling image "ghcr.io/nope"'}
                    },
                }
            ],
        },
    }


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
