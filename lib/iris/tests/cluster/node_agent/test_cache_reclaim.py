# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from iris.cluster.node_agent.cache_reclaim import reclaim_cache
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.types import K8sResource
from rigging.timing import Duration, Timestamp


class RecordingK8sService(InMemoryK8sService):
    def __init__(self, *, task_after_taint: bool = False):
        super().__init__(namespace="iris")
        self.node_taint_states: list[bool] = []
        self.task_after_taint = task_after_taint

    def add_node_taint(self, node_name: str, *, key: str, value: str, effect: str) -> None:
        self.node_taint_states.append(True)
        super().add_node_taint(node_name, key=key, value=value, effect=effect)
        if self.task_after_taint:
            self.seed_resource(
                K8sResource.PODS,
                "task-pod",
                {
                    "metadata": {
                        "name": "task-pod",
                        "labels": {"iris.managed": "true", "iris.runtime": "iris-kubernetes"},
                    },
                    "spec": {"nodeName": node_name},
                    "status": {"phase": "Pending"},
                },
            )

    def remove_node_taint(self, node_name: str, *, key: str, value: str, effect: str) -> None:
        self.node_taint_states.append(False)
        super().remove_node_taint(node_name, key=key, value=value, effect=effect)


def _k8s_with_node(*, task_after_taint: bool = False) -> RecordingK8sService:
    k8s = RecordingK8sService(task_after_taint=task_after_taint)
    k8s.seed_resource(
        K8sResource.NODES,
        "node-a",
        {
            "apiVersion": "v1",
            "kind": "Node",
            "metadata": {"name": "node-a"},
            "spec": {"taints": [{"key": "nvidia.com/gpu", "effect": "NoSchedule"}]},
        },
    )
    return k8s


def test_reclaim_cache_removes_stale_entries_after_node_becomes_idle(tmp_path):
    cache_dir = tmp_path / "iris-cache"
    cache_namespace = cache_dir / "cache"
    stale = cache_namespace / "old-model"
    fresh = cache_namespace / "current-model"
    stale.mkdir(parents=True)
    fresh.mkdir()
    (stale / "weights").write_bytes(b"stale")
    (fresh / "weights").write_bytes(b"fresh")
    os.utime(stale / "weights", (100.0, 100.0))
    os.utime(stale, (100.0, 100.0))
    os.utime(fresh, (100.0, 100.0))
    os.utime(fresh / "weights", (950.0, 950.0))

    k8s = _k8s_with_node()
    reclaimed = reclaim_cache(
        cache_dir,
        max_age=Duration.from_seconds(500),
        kubectl=k8s,
        node_name="node-a",
        now=Timestamp.from_seconds(1_000),
    )

    assert reclaimed == 1
    assert cache_namespace.is_dir()
    assert not stale.exists()
    assert (fresh / "weights").read_bytes() == b"fresh"
    assert k8s.node_taint_states == [True, False]
    assert k8s.get_json(K8sResource.NODES, "node-a")["spec"]["taints"] == [
        {"key": "nvidia.com/gpu", "effect": "NoSchedule"}
    ]


@pytest.mark.parametrize("phase", ["Pending", "Running"])
def test_reclaim_cache_with_active_task_preserves_stale_entries(tmp_path, phase):
    cache_dir = tmp_path / "iris-cache"
    stale = cache_dir / "uv-cache" / "old-wheel"
    stale.mkdir(parents=True)
    os.utime(stale, (100.0, 100.0))
    k8s = _k8s_with_node()
    k8s.seed_resource(
        K8sResource.PODS,
        "task-pod",
        {
            "metadata": {
                "name": "task-pod",
                "labels": {"iris.managed": "true", "iris.runtime": "iris-kubernetes"},
            },
            "spec": {"nodeName": "node-a"},
            "status": {"phase": phase},
        },
    )

    reclaimed = reclaim_cache(
        cache_dir,
        max_age=Duration.from_seconds(500),
        kubectl=k8s,
        node_name="node-a",
        now=Timestamp.from_seconds(1_000),
    )

    assert reclaimed == 0
    assert stale.is_dir()
    assert k8s.node_taint_states == []


def test_reclaim_cache_rechecks_for_tasks_after_blocking_admission(tmp_path):
    stale = tmp_path / "iris-cache" / "uv-cache" / "old-wheel"
    stale.mkdir(parents=True)
    os.utime(stale, (100.0, 100.0))
    k8s = _k8s_with_node(task_after_taint=True)

    reclaimed = reclaim_cache(
        tmp_path / "iris-cache",
        max_age=Duration.from_seconds(500),
        kubectl=k8s,
        node_name="node-a",
        now=Timestamp.from_seconds(1_000),
    )

    assert reclaimed == 0
    assert stale.is_dir()
    assert k8s.node_taint_states == [True, False]
