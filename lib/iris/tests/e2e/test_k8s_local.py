# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the Iris K8s service layer using a local kind cluster.

These tests validate that CloudK8sService can talk to a real Kubernetes
API server. They exercise the service protocol — not full Iris E2E jobs.

Run with: uv run pytest lib/iris/tests/e2e/test_k8s_local.py -m kind -v
"""

import time

import pytest
from iris.cluster.providers.k8s.service import CloudK8sService

from .kind_fixtures import KindCluster

pytestmark = pytest.mark.kind

NAMESPACE = "default"


@pytest.fixture
def k8s_service(kind_cluster: KindCluster) -> CloudK8sService:
    return CloudK8sService(
        namespace=NAMESPACE,
        kubeconfig_path=str(kind_cluster.kubeconfig_path),
    )


def test_kind_cluster_is_reachable(k8s_service: CloudK8sService) -> None:
    """Basic connectivity: list nodes in the kind cluster."""
    nodes = k8s_service.list_json("nodes", cluster_scoped=True)
    assert len(nodes) >= 1
    node_name = nodes[0]["metadata"]["name"]
    assert node_name  # non-empty string


def test_k8s_service_can_list_pods(k8s_service: CloudK8sService) -> None:
    """Instantiate CloudK8sService pointed at kind, list pods without error."""
    pods = k8s_service.list_json("pods")
    # default namespace may have zero pods — the important thing is no error
    assert isinstance(pods, list)


def test_k8s_service_pod_lifecycle(k8s_service: CloudK8sService) -> None:
    """Create a pod, wait for Running, delete it, verify deletion."""
    pod_name = "iris-kind-smoke-test"
    manifest = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": pod_name, "namespace": NAMESPACE},
        "spec": {
            "containers": [
                {
                    "name": "sleeper",
                    "image": "busybox:1.36",
                    "command": ["sleep", "30"],
                }
            ],
            "restartPolicy": "Never",
        },
    }

    k8s_service.apply_json(manifest)
    try:
        # Poll until Running or Succeeded (image pull may take a moment)
        deadline = time.monotonic() + 60
        phase = None
        while time.monotonic() < deadline:
            pod = k8s_service.get_json("pod", pod_name)
            assert pod is not None, f"Pod {pod_name} disappeared unexpectedly"
            phase = pod.get("status", {}).get("phase")
            if phase in ("Running", "Succeeded"):
                break
            time.sleep(1)
        assert phase in ("Running", "Succeeded"), f"Pod never reached Running, stuck in {phase}"
    finally:
        k8s_service.delete("pod", pod_name, wait=False)

    # Verify the pod is gone (with a short polling window for async deletion)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if k8s_service.get_json("pod", pod_name) is None:
            break
        time.sleep(1)
    assert k8s_service.get_json("pod", pod_name) is None, f"Pod {pod_name} was not deleted"
