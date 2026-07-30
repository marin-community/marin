# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import click
import pytest
from iris.cluster.config import (
    CoreweavePlatformConfig,
    IrisClusterConfig,
    KubernetesProviderConfig,
    PlatformConfig,
)
from iris.rpc import job_pb2

from scripts.iris.dev_gpu import (
    CoreweaveTarget,
    DevGpuState,
    PodRef,
    Priority,
    parse_running_pod,
    require_coreweave_platform,
    resolve_coscheduling,
    session_summary,
    wait_for_running_tasks,
)

JOB_ID = "/dev/dev-gpu-devbox"


def make_state(node_count: int) -> DevGpuState:
    return DevGpuState(
        session_name="devbox",
        config_file="/abs/coreweave.yaml",
        job_id=JOB_ID,
        gpus_per_node=4,
        gpu_variant="GB200",
        priority=Priority.INTERACTIVE,
        target=CoreweaveTarget(namespace="iris", kubeconfig_path="/k/cfg"),
        pods=[PodRef(namespace="iris", pod_name=f"dev-gpu-pod-{i}", container="task") for i in range(node_count)],
    )


def test_state_round_trip():
    # The session file is a persisted contract: `status` and `release` read it back.
    assert DevGpuState.from_json(make_state(2).to_json()) == make_state(2)


def test_require_coreweave_namespace_comes_from_kubernetes_provider():
    # Regression: pods are created/listed in kubernetes_provider.namespace, NOT
    # platform.coreweave.namespace (independent config fields that can diverge).
    c = IrisClusterConfig(
        platform=PlatformConfig(coreweave=CoreweavePlatformConfig(namespace="platform-ns")),
        kubernetes_provider=KubernetesProviderConfig(namespace="pods-live-here"),
    )
    assert require_coreweave_platform(c).namespace == "pods-live-here"


@pytest.mark.parametrize(
    "pods, expected",
    [
        # picks the Running pod, ignoring Pending
        (
            {
                "items": [
                    {"metadata": {"name": "b"}, "status": {"phase": "Pending"}},
                    {"metadata": {"name": "a"}, "status": {"phase": "Running"}},
                ]
            },
            "a",
        ),
        # deterministic tie-break: lexicographically-first among multiple Running
        (
            {
                "items": [
                    {"metadata": {"name": "z"}, "status": {"phase": "Running"}},
                    {"metadata": {"name": "a"}, "status": {"phase": "Running"}},
                ]
            },
            "a",
        ),
        # nothing Running -> None (so the caller keeps polling)
        ({"items": [{"metadata": {"name": "a"}, "status": {"phase": "Pending"}}]}, None),
    ],
)
def test_parse_running_pod(pods, expected):
    assert parse_running_pod(pods) == expected


@pytest.mark.parametrize(
    "variant, gpus_per_node, node_count, expected",
    [
        # A single node needs no gang at all.
        ("GB200", 4, 1, None),
        ("H100", 8, 1, None),
        # The point of a two-node GB200 session: both trays on one rack's NVLink fabric.
        ("GB200", 4, 2, "nvlink.domain"),
        # H100 nodes carry no nvlink.domain label, so the gang gets soft IB colocation.
        ("H100", 8, 2, "leafgroup"),
    ],
)
def test_resolve_coscheduling(variant, gpus_per_node, node_count, expected):
    level = resolve_coscheduling(variant, gpus_per_node, node_count)
    assert (level.group_by if level else None) == expected


def test_multi_node_rejects_fractional_pods():
    # Two 2-GPU pods could both land on one 4-GPU tray, which is not a two-node session.
    with pytest.raises(click.ClickException, match="whole nodes"):
        resolve_coscheduling("GB200", 2, 2)


def test_session_summary_lists_every_node():
    summary = session_summary(make_state(2))
    assert "Nodes: 2 x 4 GB200" in summary
    assert "Node 0: dev-gpu-pod-0" in summary
    assert "Node 1: dev-gpu-pod-1" in summary


class FakeTask:
    def __init__(self, task_index: int):
        self.task_id = f"{JOB_ID}/{task_index}"
        self.task_index = task_index

    def status(self):
        return job_pb2.TaskStatus(state=job_pb2.TASK_STATE_RUNNING)


class FakeJob:
    """A running job whose tasks come back in an arbitrary order."""

    def __init__(self, replicas: int):
        self.tasks_returned = [FakeTask(index) for index in reversed(range(replicas))]

    def state_only(self):
        return job_pb2.JOB_STATE_RUNNING

    def tasks(self):
        return self.tasks_returned


def test_running_tasks_come_back_in_task_index_order():
    # `connect --node 1` means "the second node", so pod order must follow the task
    # index rather than whatever order the controller happens to list tasks in.
    task_ids = wait_for_running_tasks(FakeJob(2), node_count=2, timeout=1)
    assert task_ids == [f"{JOB_ID}/0", f"{JOB_ID}/1"]
