# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from iris.cluster.node_agent.cache_reclaim import reclaim_cache
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.types import K8sResource
from rigging.timing import Duration


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

    reclaimed = reclaim_cache(
        cache_dir,
        max_age=Duration.from_seconds(500),
        kubectl=InMemoryK8sService(namespace="iris"),
        node_name="node-a",
        now=1_000.0,
    )

    assert reclaimed == 1
    assert cache_namespace.is_dir()
    assert not stale.exists()
    assert (fresh / "weights").read_bytes() == b"fresh"


@pytest.mark.parametrize("phase", ["Pending", "Running"])
def test_reclaim_cache_with_active_task_preserves_stale_entries(tmp_path, phase):
    cache_dir = tmp_path / "iris-cache"
    stale = cache_dir / "uv-cache" / "old-wheel"
    stale.mkdir(parents=True)
    os.utime(stale, (100.0, 100.0))
    k8s = InMemoryK8sService(namespace="iris")
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
        now=1_000.0,
    )

    assert reclaimed == 0
    assert stale.is_dir()
