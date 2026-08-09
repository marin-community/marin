# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest
from iris.cluster.backends.k8s.tasks import K8sTaskProvider, PodConfig
from iris.cluster.controller.persistence.reads import ControlSnapshot
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.task_state import RunningTaskEntry
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.types import K8sResource
from iris.cluster.resources.attempt import AttemptLaunch, AttemptLaunchTemplate
from iris.cluster.resources.job import ContainerProfile, CoschedulingConfig, PriorityBand
from iris.cluster.runtime.env import build_common_iris_env
from iris.cluster.types import AttemptUid, JobName
from iris.rpc import job_pb2
from iris.rpc.legacy_job_codec import (
    constraint_from_proto,
    environment_from_proto,
    resource_spec_from_proto,
    runtime_entrypoint_from_proto,
)
from iris.test_util import FakeStatsTable
from iris.time_proto import duration_from_proto

KUEUE_POD_GROUP_NAME = "kueue.x-k8s.io/pod-group-name"
LABEL_MANAGED = "iris.managed"
LABEL_RUNTIME = "iris.runtime"
RUNTIME_LABEL_VALUE = "iris-kubernetes"
TASK_CONTAINER_NAME = "task"


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
        pods=pod_config(),
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
    priority: int = job_pb2.PRIORITY_BAND_INHERIT,
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
    return K8sTaskProvider(kubectl=k8s, pods=pod_config(local_queue=local_queue), **kwargs)


def launch_from_request(request: job_pb2.RunTaskRequest) -> AttemptLaunch:
    """Decode the legacy worker launch wire used by these adapter fixtures."""
    return AttemptLaunch(
        task_id=JobName.from_wire(request.task_id),
        attempt_id=request.attempt_id,
        attempt_uid=AttemptUid(request.attempt_uid),
        template=AttemptLaunchTemplate(
            num_tasks=request.num_tasks,
            entrypoint=runtime_entrypoint_from_proto(request.entrypoint),
            environment=environment_from_proto(request.environment),
            bundle_id=request.bundle_id,
            resources=resource_spec_from_proto(request.resources),
            timeout=duration_from_proto(request.timeout) if request.HasField("timeout") else None,
            ports=tuple(request.ports),
            constraints=tuple(constraint_from_proto(value) for value in request.constraints),
            task_image=request.task_image,
            coscheduling=(
                CoschedulingConfig(request.coscheduling.group_by) if request.HasField("coscheduling") else None
            ),
            priority_band=PriorityBand(request.priority),
            container_profile=ContainerProfile(request.container_profile),
        ),
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
        tasks_to_run=[launch_from_request(task) for task in tasks_to_run or []],
    )


def observe_pod_update(pod: dict, workload: dict | None = None) -> TaskUpdate:
    """Reconcile a seeded pod through the public Kubernetes provider boundary."""
    k8s = InMemoryK8sService(namespace="iris")
    provider = K8sTaskProvider(kubectl=k8s, pods=pod_config(), cluster_scan_interval=0.0)
    entry = RunningTaskEntry(task_id=JobName.from_wire("/job/0"), attempt_id=0)
    try:
        provider.sync(make_batch(tasks_to_run=[make_run_req("/job/0")]))
        applied = k8s.list_json(K8sResource.PODS)[0]
        observed = deepcopy(pod)
        observed["kind"] = "Pod"
        observed_metadata = observed.setdefault("metadata", {})
        observed_metadata["name"] = applied["metadata"]["name"]
        observed_metadata["labels"] = {
            **applied["metadata"]["labels"],
            **observed_metadata.get("labels", {}),
        }
        k8s.seed_resource(K8sResource.PODS, observed_metadata["name"], observed)
        if workload is not None:
            workload_name = workload.get("metadata", {}).get("name", "workload")
            k8s.seed_resource(K8sResource.WORKLOADS, workload_name, deepcopy(workload))
        updates = provider.sync(make_batch(running_tasks=[entry]))
        assert len(updates) == 1
        return updates[0]
    finally:
        provider.close()


# A Kueue admission message for an over-large GPU request (cpu=160 on 128-vCPU
# H100 nodes under InfiniBand TAS): the whole pod cannot fit one node, so every
# node is excluded and the workload never reserves quota. The motivating incident
# for the status_message / iris.task_event diagnostics.
KUEUE_UNADMITTED_MSG = (
    "couldn't assign flavors to pod set main: topology \"infiniband\" doesn't allow to "
    'fit any of 1 pod(s). Total nodes: 32; excluded: resource "cpu": 32'
)
SINGLETON_POD_UID = "pod-uid"


def gated_pod(name: str = "iris-job-0-0", pod_group: str = "wl-abc") -> dict:
    """A Pending pod blocked on a Kueue scheduling gate (no container has started)."""
    return {
        "metadata": {"name": name, "labels": {KUEUE_POD_GROUP_NAME: pod_group}},
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


def singleton_gated_pod(name: str = "iris-job-0-0", uid: str = SINGLETON_POD_UID) -> dict:
    """A Kueue-managed singleton Pod, which has no pod-group label."""
    pod = gated_pod(name=name)
    pod["metadata"] = {
        "name": name,
        "uid": uid,
        "labels": {"kueue.x-k8s.io/queue-name": "cw-use02a-lq"},
    }
    return pod


def unadmitted_workload(name: str = "wl-abc", msg: str = KUEUE_UNADMITTED_MSG) -> dict:
    """A Workload Kueue has evaluated and declined: QuotaReserved=False with a reason."""
    return {
        "metadata": {"name": name},
        "spec": {"queueName": "cw-use02a-lq"},
        "status": {"conditions": [{"type": "QuotaReserved", "status": "False", "reason": "Pending", "message": msg}]},
    }


def singleton_unadmitted_workload(
    pod_name: str = "iris-job-0-0",
    pod_uid: str = SINGLETON_POD_UID,
    msg: str = KUEUE_UNADMITTED_MSG,
) -> dict:
    """The auto-generated Workload owned by one Kueue-managed Pod."""
    workload = unadmitted_workload(name=f"pod-{pod_name}-abcde", msg=msg)
    workload["metadata"].update(
        {
            "labels": {"kueue.x-k8s.io/job-uid": pod_uid},
            "ownerReferences": [{"apiVersion": "v1", "kind": "Pod", "name": pod_name, "uid": pod_uid}],
        }
    )
    return workload


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
                    "name": TASK_CONTAINER_NAME,
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
        LABEL_MANAGED: "true",
        LABEL_RUNTIME: RUNTIME_LABEL_VALUE,
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
        environment=environment_from_proto(req.environment),
        constraints=tuple(constraint_from_proto(value) for value in req.constraints),
        ports=req.ports,
        resources=resource_spec_from_proto(req.resources) if req.HasField("resources") else None,
    )
