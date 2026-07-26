# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from iris.cli import cluster as cluster_cli
from iris.cluster.config import KUBERNETES_WORKER_RUNTIME


def test_build_cluster_images_for_kubernetes_supports_task_node_architectures(monkeypatch):
    builds: list[tuple[str, str]] = []

    def record_build(**kwargs):
        builds.append((kwargs["image_type"], kwargs["platform"]))

    monkeypatch.setattr(cluster_cli, "build_image", record_build)
    config = SimpleNamespace(
        controller=SimpleNamespace(image="ghcr.io/marin-community/iris-controller:tree"),
        defaults=SimpleNamespace(
            worker=SimpleNamespace(
                runtime=KUBERNETES_WORKER_RUNTIME,
                docker_image="ghcr.io/marin-community/iris-worker:tree",
                default_task_image="ghcr.io/marin-community/iris-task:tree",
            )
        ),
    )

    built = cluster_cli._build_cluster_images(config, "tree")

    assert built == {
        "controller": "ghcr.io/marin-community/iris-controller:tree",
        "task": "ghcr.io/marin-community/iris-task:tree",
    }
    assert builds == [
        ("controller", "linux/amd64,linux/arm64"),
        ("task", "linux/amd64,linux/arm64"),
    ]
