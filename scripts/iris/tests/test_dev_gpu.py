# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from contextlib import contextmanager

import click
import pytest
from click.testing import CliRunner
from iris.cluster.config import (
    CoreweavePlatformConfig,
    IrisClusterConfig,
    KubernetesProviderConfig,
    PlatformConfig,
)
from iris.rpc import job_pb2

from scripts.iris import dev_gpu
from scripts.iris.dev_gpu import (
    CoreweaveTarget,
    DevGpuNode,
    DevGpuState,
    PodRef,
    Priority,
    parse_running_pod,
    pod_label_selector,
    require_coreweave_platform,
    resolve_coscheduling,
    select_node,
    session_summary,
)

JOB_ID = "/dev/dev-gpu-devbox"


def make_node(index: int) -> DevGpuNode:
    return DevGpuNode(
        task_id=f"{JOB_ID}/{index}",
        pod=PodRef(namespace="iris", pod_name=f"dev-gpu-pod-{index}", container="task"),
    )


def make_state(node_count: int, *, gpu_variant: str = "GB200", gpus_per_node: int = 4) -> DevGpuState:
    return DevGpuState(
        session_name="devbox",
        config_file="/abs/coreweave.yaml",
        job_id=JOB_ID,
        gpus_per_node=gpus_per_node,
        gpu_variant=gpu_variant,
        priority=Priority.INTERACTIVE,
        target=CoreweaveTarget(namespace="iris", kubeconfig_path="/k/cfg"),
        nodes=[make_node(index) for index in range(node_count)],
    )


class FakeTask:
    def __init__(self, task_id: str, task_index: int):
        self.task_id = task_id
        self.task_index = task_index

    def status(self):
        return job_pb2.TaskStatus(state=job_pb2.TASK_STATE_RUNNING)


class FakeJob:
    def __init__(self, job_id: str, replicas: int):
        self.job_id = job_id
        # Reverse order: the tool must sort by task index, not trust list order,
        # or `connect --node 1` would land on an arbitrary pod.
        self._tasks = [FakeTask(f"{job_id}/{index}", index) for index in reversed(range(replicas))]

    def state_only(self):
        return job_pb2.JOB_STATE_RUNNING

    def tasks(self):
        return self._tasks


class FakeClient:
    def __init__(self):
        self.submitted: dict = {}
        self.terminated: list[str] = []

    def submit(self, **kwargs):
        self.submitted = kwargs
        return FakeJob(JOB_ID, kwargs["replicas"])

    def terminate(self, job_name):
        self.terminated.append(str(job_name))

    def job_state(self, job_name):
        return job_pb2.JOB_STATE_RUNNING


def coreweave_config() -> IrisClusterConfig:
    return IrisClusterConfig(
        platform=PlatformConfig(coreweave=CoreweavePlatformConfig(kubeconfig_path="/k/cfg")),
        kubernetes_provider=KubernetesProviderConfig(namespace="iris"),
    )


@pytest.fixture
def fake_cluster(monkeypatch, tmp_path):
    """Run dev_gpu's CLI against a fake controller and a fake kubectl.

    One pod per task, named after the task's index, so a test can assert which pod a
    session recorded for which node.
    """
    client = FakeClient()

    @contextmanager
    def fake_controller_client(config_file):
        yield client

    def fake_kubectl(cmd, **kwargs):
        selector = cmd[cmd.index("-l") + 1]
        index = next(i for i in range(8) if pod_label_selector(f"{JOB_ID}/{i}") == selector)
        pods = {"items": [{"metadata": {"name": f"dev-gpu-pod-{index}"}, "status": {"phase": "Running"}}]}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(pods), stderr="")

    monkeypatch.setattr(dev_gpu, "load_config", lambda config_file: coreweave_config())
    monkeypatch.setattr(dev_gpu, "controller_client", fake_controller_client)
    monkeypatch.setattr(dev_gpu.subprocess, "run", fake_kubectl)
    return client


def invoke(state_dir, *args):
    return CliRunner().invoke(
        dev_gpu.cli,
        ["--config", "/abs/coreweave.yaml", "--name", "devbox", *args],
        obj=dev_gpu.Context(state_dir=state_dir),
    )


def test_state_round_trip():
    # The session file is a persisted contract: `status` and `release` read it back.
    state = make_state(2)
    assert DevGpuState.from_json(state.to_json()) == state


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


def test_connect_rejects_out_of_range_node():
    with pytest.raises(click.ClickException, match="has 2 node"):
        select_node(make_state(2), 2)


def test_session_summary_lists_every_pod():
    summary = session_summary(make_state(2))
    assert "Nodes: 2 x 4 GB200" in summary
    assert "Pod[0]: dev-gpu-pod-0" in summary
    assert "Pod[1]: dev-gpu-pod-1" in summary


@pytest.mark.parametrize(
    "variant, node_count, expected_coscheduling",
    [("H100", 1, None), ("GB200", 2, "nvlink.domain")],
)
def test_allocate_records_every_node_then_releases(
    fake_cluster, monkeypatch, tmp_path, variant, node_count, expected_coscheduling
):
    """allocate holds the gang, records one pod per node, and releases on Ctrl-C."""
    state_file = tmp_path / "devbox.json"
    held: list[DevGpuState] = []

    def interrupt_the_hold(seconds):
        held.append(DevGpuState.from_json(state_file.read_text()))
        raise KeyboardInterrupt

    monkeypatch.setattr(dev_gpu.time, "sleep", interrupt_the_hold)

    result = invoke(tmp_path, "allocate", "--gpu-variant", variant, "--nodes", str(node_count))
    assert result.exit_code == 0, result.output

    assert fake_cluster.submitted["replicas"] == node_count
    coscheduling = fake_cluster.submitted["coscheduling"]
    assert (coscheduling.group_by if coscheduling else None) == expected_coscheduling

    (state,) = held
    assert [node.pod.pod_name for node in state.nodes] == [f"dev-gpu-pod-{i}" for i in range(node_count)]
    assert state.gpus_per_node == dev_gpu.GPUS_PER_NODE[variant]
    for index, node in enumerate(state.nodes):
        assert node.task_id == f"{JOB_ID}/{index}"

    # Releasing must tear down the whole gang and leave no stale session behind.
    assert fake_cluster.terminated == [JOB_ID]
    assert not state_file.exists()


def test_connect_targets_the_requested_node(fake_cluster, monkeypatch, tmp_path):
    dev_gpu.save_state(tmp_path / "devbox.json", make_state(2))
    connect_commands: list[list[str]] = []
    monkeypatch.setattr(dev_gpu, "run_logged", lambda cmd, **kwargs: connect_commands.append(cmd))

    result = invoke(tmp_path, "connect", "--node", "1")
    assert result.exit_code == 0, result.output

    (cmd,) = connect_commands
    assert "dev-gpu-pod-1" in cmd
    assert "dev-gpu-pod-0" not in cmd
