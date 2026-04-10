# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the K8sService contract using InMemoryK8sService.

Covers validation, scheduling, resource tracking, and protocol behavior
that matters to K8sTaskProvider and CoreWeave controller consumers.
"""

from __future__ import annotations

import pytest

from iris.cluster.providers.k8s.fake import FakeNodeResources, InMemoryK8sService
from iris.cluster.providers.k8s.types import K8sResource, KubectlError


def _get_pod_phase(svc: InMemoryK8sService, name: str) -> str:
    """Get a pod's phase, asserting it exists."""
    result = svc.get_json(K8sResource.PODS, name)
    assert result is not None, f"Pod {name!r} not found"
    return result["status"]["phase"]


def _pod_manifest(
    name: str,
    labels: dict[str, str] | None = None,
    node_pool: str | None = None,
    resources: dict | None = None,
    node_selector: dict[str, str] | None = None,
    tolerations: list[dict] | None = None,
) -> dict:
    """Build a minimal Pod manifest for testing."""
    spec: dict = {"containers": [{"name": "main", "image": "busybox"}]}
    if node_pool:
        spec.setdefault("nodeSelector", {})["cloud.google.com/gke-nodepool"] = node_pool
    if node_selector:
        spec.setdefault("nodeSelector", {}).update(node_selector)
    if tolerations:
        spec["tolerations"] = tolerations
    if resources:
        spec["containers"][0]["resources"] = resources
    manifest: dict = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": name},
        "spec": spec,
    }
    if labels:
        manifest["metadata"]["labels"] = labels
    return manifest


def _deployment_manifest(name: str, node_pool: str | None = None) -> dict:
    pod_spec: dict = {"containers": [{"name": "app", "image": "nginx"}]}
    if node_pool:
        pod_spec["nodeSelector"] = {"cloud.google.com/gke-nodepool": node_pool}
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {"name": name},
        "spec": {"template": {"spec": pod_spec}},
    }


@pytest.fixture
def svc() -> InMemoryK8sService:
    return InMemoryK8sService(namespace="test-ns", available_node_pools=["cpu-pool", "gpu-pool"])


@pytest.fixture
def sched_svc() -> InMemoryK8sService:
    """InMemoryK8sService with no pools for scheduling tests."""
    return InMemoryK8sService(namespace="test-ns")


# ========================================================================
# Manifest validation
# ========================================================================


def test_missing_kind(svc: InMemoryK8sService):
    with pytest.raises(KubectlError, match="missing 'kind'"):
        svc.apply_json({"metadata": {"name": "x"}})


def test_missing_name(svc: InMemoryK8sService):
    with pytest.raises(KubectlError, match=r"missing 'metadata\.name'"):
        svc.apply_json({"kind": "Pod", "metadata": {}})


def test_invalid_node_pool(svc: InMemoryK8sService):
    manifest = _pod_manifest("p1", node_pool="nonexistent-pool")
    with pytest.raises(KubectlError, match="not found"):
        svc.apply_json(manifest)


def test_unknown_resource_type(svc: InMemoryK8sService):
    manifest = _pod_manifest("p1", resources={"requests": {"bogus/resource": "1"}})
    with pytest.raises(KubectlError, match="Unknown resource type"):
        svc.apply_json(manifest)


def test_unknown_manifest_kind(svc: InMemoryK8sService):
    """apply_json rejects manifests whose kind is not in _KIND_TO_PLURAL."""
    manifest = {"apiVersion": "v1", "kind": "FancyWidget", "metadata": {"name": "w1"}}
    with pytest.raises(KubectlError, match=r"Unknown manifest kind.*FancyWidget"):
        svc.apply_json(manifest)


def test_deployment_node_pool_validation(svc: InMemoryK8sService):
    """Node pool validation works for nested pod specs (Deployments, Jobs)."""
    manifest = _deployment_manifest("d1", node_pool="nonexistent-pool")
    with pytest.raises(KubectlError, match="not found"):
        svc.apply_json(manifest)


def test_no_node_pool_constraint_skips_validation(svc: InMemoryK8sService):
    """Manifests without node pool selectors pass even with available_node_pools set."""
    manifest = _pod_manifest("p1")
    svc.apply_json(manifest)
    assert svc.get_json(K8sResource.PODS, "p1") is not None


def test_no_available_pools_skips_pool_validation():
    """When available_node_pools is None, any pool selector is accepted."""
    svc = InMemoryK8sService(available_node_pools=None)
    manifest = _pod_manifest("p1", node_pool="any-pool")
    svc.apply_json(manifest)
    assert svc.get_json(K8sResource.PODS, "p1") is not None


# ========================================================================
# Failure injection
# ========================================================================


def test_inject_failure_apply(svc: InMemoryK8sService):
    svc.inject_failure("apply_json", KubectlError("scheduling failed"))
    with pytest.raises(KubectlError, match="scheduling failed"):
        svc.apply_json(_pod_manifest("p1"))

    # One-shot: next call succeeds
    svc.apply_json(_pod_manifest("p1"))
    assert svc.get_json(K8sResource.PODS, "p1") is not None


def test_inject_failure_get(svc: InMemoryK8sService):
    svc.inject_failure("get_json", KubectlError("timeout"))
    with pytest.raises(KubectlError, match="timeout"):
        svc.get_json(K8sResource.PODS, "p1")


def test_inject_failure_list(svc: InMemoryK8sService):
    svc.inject_failure("list_json", KubectlError("api error"))
    with pytest.raises(KubectlError, match="api error"):
        svc.list_json(K8sResource.PODS)


def test_inject_failure_delete(svc: InMemoryK8sService):
    svc.inject_failure("delete", KubectlError("forbidden"))
    with pytest.raises(KubectlError, match="forbidden"):
        svc.delete(K8sResource.PODS, "p1")


# ========================================================================
# List filtering by label
# ========================================================================


def test_list_json_label_filter(svc: InMemoryK8sService):
    svc.apply_json(_pod_manifest("p1", labels={"app": "web", "env": "prod"}))
    svc.apply_json(_pod_manifest("p2", labels={"app": "web", "env": "staging"}))
    svc.apply_json(_pod_manifest("p3", labels={"app": "worker"}))

    results = svc.list_json(K8sResource.PODS, labels={"app": "web"})
    assert len(results) == 2

    results = svc.list_json(K8sResource.PODS, labels={"app": "web", "env": "prod"})
    assert len(results) == 1
    assert results[0]["metadata"]["name"] == "p1"

    results = svc.list_json(K8sResource.PODS, labels={"app": "nonexistent"})
    assert len(results) == 0


# ========================================================================
# Delete-then-describe behavior
# ========================================================================


def test_apply_delete_then_get_returns_none(svc: InMemoryK8sService):
    svc.apply_json(_pod_manifest("myapp"))
    svc.delete(K8sResource.PODS, "myapp")
    assert svc.get_json(K8sResource.PODS, "myapp") is None


def test_delete_nonexistent_is_idempotent(svc: InMemoryK8sService):
    """Deleting a nonexistent resource does not raise."""
    svc.delete(K8sResource.PODS, "nope")


def test_apply_overwrites(svc: InMemoryK8sService):
    m1 = _pod_manifest("p1")
    m2 = _pod_manifest("p1", labels={"version": "v2"})
    svc.apply_json(m1)
    svc.apply_json(m2)
    result = svc.get_json(K8sResource.PODS, "p1")
    assert result is not None
    assert result["metadata"].get("labels", {}).get("version") == "v2"


# ========================================================================
# Logs and events (used by K8sTaskProvider)
# ========================================================================


def test_logs_with_tail(svc: InMemoryK8sService):
    svc.set_logs("mypod", "line1\nline2\nline3")
    assert svc.logs("mypod", tail=2) == "line2\nline3"
    assert svc.logs("mypod") == "line1\nline2\nline3"


def test_logs_missing_pod(svc: InMemoryK8sService):
    assert svc.logs("nope") == ""


def test_events_field_selector(svc: InMemoryK8sService):
    svc.add_event({"involvedObject": {"name": "pod-a"}, "reason": "Scheduled"})
    svc.add_event({"involvedObject": {"name": "pod-b"}, "reason": "Failed"})

    all_events = svc.get_events()
    assert len(all_events) == 2

    filtered = svc.get_events(field_selector="involvedObject.name=pod-a")
    assert len(filtered) == 1
    assert filtered[0]["reason"] == "Scheduled"


def test_stream_logs_incremental(svc: InMemoryK8sService):
    svc.set_logs("p1", "hello world")
    result = svc.stream_logs("p1")

    # No new content → empty
    assert svc.stream_logs("p1", since_time=result.last_timestamp).lines == []

    # Appended content → only the delta
    svc.set_logs("p1", "hello world\nnew line")
    result2 = svc.stream_logs("p1", since_time=result.last_timestamp)
    assert [l.data for l in result2.lines] == ["new line"]


# ========================================================================
# Port forwarding
# ========================================================================


def test_port_forward_yields_url(svc: InMemoryK8sService):
    with svc.port_forward("my-svc", 8080) as url:
        assert url.startswith("http://127.0.0.1:")


def test_port_forward_with_explicit_port(svc: InMemoryK8sService):
    with svc.port_forward("my-svc", 8080, local_port=12345) as url:
        assert url == "http://127.0.0.1:12345"


def test_port_forward_failure_injection(svc: InMemoryK8sService):
    svc.inject_failure("port_forward", KubectlError("tunnel failed", 1))
    with pytest.raises(KubectlError):
        with svc.port_forward("my-svc", 8080):
            pass


# ========================================================================
# Scheduling: pod placement, resource tracking, taints/tolerations
# ========================================================================


def test_pod_scheduled_on_matching_node(sched_svc: InMemoryK8sService):
    sched_svc.add_node_pool(
        "gpu-pool",
        labels={"accelerator": "nvidia-a100"},
        resources=FakeNodeResources(gpu_count=4, cpu_millicores=8000, memory_bytes=64 * 1024**3),
    )
    pod = _pod_manifest(
        "gpu-pod",
        node_selector={"accelerator": "nvidia-a100", "cloud.google.com/gke-nodepool": "gpu-pool"},
        resources={"requests": {"nvidia.com/gpu": "2"}},
    )
    sched_svc.apply_json(pod)
    result = sched_svc.get_json(K8sResource.PODS, "gpu-pod")
    assert result is not None
    assert result["status"]["phase"] == "Running"
    assert result["spec"]["nodeName"] == "gpu-pool-0"


def test_pod_pending_no_matching_node(sched_svc: InMemoryK8sService):
    sched_svc.add_node_pool("cpu-pool")
    pod = _pod_manifest(
        "gpu-pod",
        node_selector={"accelerator": "nvidia-a100"},
        resources={"requests": {"nvidia.com/gpu": "1"}},
    )
    sched_svc.apply_json(pod)
    result = sched_svc.get_json(K8sResource.PODS, "gpu-pod")
    assert result is not None
    assert result["status"]["phase"] == "Pending"


def test_pod_pending_insufficient_gpu(sched_svc: InMemoryK8sService):
    sched_svc.add_node_pool(
        "gpu-pool",
        resources=FakeNodeResources(gpu_count=4),
    )
    pod = _pod_manifest(
        "big-pod",
        node_selector={"cloud.google.com/gke-nodepool": "gpu-pool"},
        resources={"requests": {"nvidia.com/gpu": "8"}},
    )
    sched_svc.apply_json(pod)
    result = sched_svc.get_json(K8sResource.PODS, "big-pod")
    assert result is not None
    assert result["status"]["phase"] == "Pending"


def test_toleration_required_for_tainted_node(sched_svc: InMemoryK8sService):
    sched_svc.add_node_pool(
        "tainted-pool",
        taints=[{"key": "dedicated", "value": "gpu", "effect": "NoSchedule"}],
    )

    pod_no_tol = _pod_manifest(
        "no-tol",
        node_selector={"cloud.google.com/gke-nodepool": "tainted-pool"},
    )
    sched_svc.apply_json(pod_no_tol)
    assert _get_pod_phase(sched_svc, "no-tol") == "Pending"

    pod_with_tol = _pod_manifest(
        "with-tol",
        node_selector={"cloud.google.com/gke-nodepool": "tainted-pool"},
        tolerations=[{"key": "dedicated", "value": "gpu", "effect": "NoSchedule"}],
    )
    sched_svc.apply_json(pod_with_tol)
    assert _get_pod_phase(sched_svc, "with-tol") == "Running"


def test_resource_commitment_tracking(sched_svc: InMemoryK8sService):
    sched_svc.add_node_pool(
        "gpu-pool",
        resources=FakeNodeResources(gpu_count=4),
    )

    pod1 = _pod_manifest(
        "pod1",
        node_selector={"cloud.google.com/gke-nodepool": "gpu-pool"},
        resources={"requests": {"nvidia.com/gpu": "2"}},
    )
    sched_svc.apply_json(pod1)
    assert _get_pod_phase(sched_svc, "pod1") == "Running"

    pod2 = _pod_manifest(
        "pod2",
        node_selector={"cloud.google.com/gke-nodepool": "gpu-pool"},
        resources={"requests": {"nvidia.com/gpu": "2"}},
    )
    sched_svc.apply_json(pod2)
    assert _get_pod_phase(sched_svc, "pod2") == "Running"

    # No capacity left
    pod3 = _pod_manifest(
        "pod3",
        node_selector={"cloud.google.com/gke-nodepool": "gpu-pool"},
        resources={"requests": {"nvidia.com/gpu": "2"}},
    )
    sched_svc.apply_json(pod3)
    assert _get_pod_phase(sched_svc, "pod3") == "Pending"


def test_delete_pod_releases_resources(sched_svc: InMemoryK8sService):
    sched_svc.add_node_pool(
        "gpu-pool",
        resources=FakeNodeResources(gpu_count=2),
    )

    pod1 = _pod_manifest(
        "pod1",
        node_selector={"cloud.google.com/gke-nodepool": "gpu-pool"},
        resources={"requests": {"nvidia.com/gpu": "2"}},
    )
    sched_svc.apply_json(pod1)
    assert _get_pod_phase(sched_svc, "pod1") == "Running"

    pod2 = _pod_manifest(
        "pod2",
        node_selector={"cloud.google.com/gke-nodepool": "gpu-pool"},
        resources={"requests": {"nvidia.com/gpu": "2"}},
    )
    sched_svc.apply_json(pod2)
    assert _get_pod_phase(sched_svc, "pod2") == "Pending"

    sched_svc.delete(K8sResource.PODS, "pod1")

    pod3 = _pod_manifest(
        "pod3",
        node_selector={"cloud.google.com/gke-nodepool": "gpu-pool"},
        resources={"requests": {"nvidia.com/gpu": "2"}},
    )
    sched_svc.apply_json(pod3)
    assert _get_pod_phase(sched_svc, "pod3") == "Running"


def test_failed_scheduling_event_generated(sched_svc: InMemoryK8sService):
    sched_svc.add_node_pool("cpu-pool")
    pod = _pod_manifest(
        "unschedulable",
        node_selector={"accelerator": "nonexistent"},
    )
    sched_svc.apply_json(pod)
    assert _get_pod_phase(sched_svc, "unschedulable") == "Pending"

    events = sched_svc.get_events(field_selector="involvedObject.name=unschedulable")
    assert len(events) == 1
    assert events[0]["reason"] == "FailedScheduling"


def test_list_nodes_returns_node_dicts(sched_svc: InMemoryK8sService):
    sched_svc.add_node_pool(
        "my-pool",
        resources=FakeNodeResources(gpu_count=8, cpu_millicores=16000),
    )
    nodes = sched_svc.list_json(K8sResource.NODES)
    assert len(nodes) == 1
    node = nodes[0]
    assert node["metadata"]["name"] == "my-pool-0"
    assert node["status"]["allocatable"]["cpu"] == "16000m"
    assert node["status"]["allocatable"]["nvidia.com/gpu"] == "8"


# ========================================================================
# Exec (used by profiling)
# ========================================================================


def test_exec_existing_pod(svc: InMemoryK8sService):
    svc.apply_json(_pod_manifest("p1"))
    result = svc.exec("p1", ["echo", "hello"])
    assert result.returncode == 0


def test_exec_missing_pod(svc: InMemoryK8sService):
    result = svc.exec("nope", ["echo", "hello"])
    assert result.returncode == 1
