# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for K8sServiceImpl — in-memory K8sService for DRY_RUN/LOCAL modes."""

from __future__ import annotations

import pytest

from unittest.mock import MagicMock

from iris.cluster.k8s.k8s_service import K8sService
from iris.cluster.k8s.k8s_service_impl import FakeNodeResources, K8sServiceImpl
from iris.cluster.k8s.k8s_types import KubectlError
from iris.cluster.service_mode import ServiceMode


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
def svc() -> K8sServiceImpl:
    return K8sServiceImpl(namespace="test-ns", available_node_pools=["cpu-pool", "gpu-pool"])


# -- Protocol conformance --


def test_implements_protocol():
    impl = K8sServiceImpl()
    assert isinstance(impl, K8sService)


# -- Manifest validation --


def test_missing_kind(svc: K8sServiceImpl):
    with pytest.raises(KubectlError, match="missing 'kind'"):
        svc.apply_json({"metadata": {"name": "x"}})


def test_missing_name(svc: K8sServiceImpl):
    with pytest.raises(KubectlError, match=r"missing 'metadata\.name'"):
        svc.apply_json({"kind": "Pod", "metadata": {}})


def test_invalid_node_pool(svc: K8sServiceImpl):
    manifest = _pod_manifest("p1", node_pool="nonexistent-pool")
    with pytest.raises(KubectlError, match="not found"):
        svc.apply_json(manifest)


def test_valid_node_pool(svc: K8sServiceImpl):
    manifest = _pod_manifest("p1", node_pool="gpu-pool")
    svc.apply_json(manifest)
    assert svc.get_json("pod", "p1") == manifest


def test_unknown_resource_type(svc: K8sServiceImpl):
    manifest = _pod_manifest("p1", resources={"requests": {"bogus/resource": "1"}})
    with pytest.raises(KubectlError, match="Unknown resource type"):
        svc.apply_json(manifest)


def test_valid_resource_types(svc: K8sServiceImpl):
    manifest = _pod_manifest(
        "p1",
        resources={
            "requests": {"cpu": "100m", "memory": "256Mi", "nvidia.com/gpu": "1"},
            "limits": {"cpu": "200m", "memory": "512Mi"},
        },
    )
    svc.apply_json(manifest)
    assert svc.get_json("pod", "p1") is not None


def test_deployment_node_pool_validation(svc: K8sServiceImpl):
    """Node pool validation works for nested pod specs (Deployments, Jobs)."""
    manifest = _deployment_manifest("d1", node_pool="nonexistent-pool")
    with pytest.raises(KubectlError, match="not found"):
        svc.apply_json(manifest)


def test_no_node_pool_constraint_skips_validation(svc: K8sServiceImpl):
    """Manifests without node pool selectors pass even with available_node_pools set."""
    manifest = _pod_manifest("p1")
    svc.apply_json(manifest)
    assert svc.get_json("pod", "p1") is not None


def test_no_available_pools_skips_pool_validation():
    """When available_node_pools is None, any pool selector is accepted."""
    svc = K8sServiceImpl(available_node_pools=None)
    manifest = _pod_manifest("p1", node_pool="any-pool")
    svc.apply_json(manifest)
    assert svc.get_json("pod", "p1") is not None


# -- State tracking: apply → get → list → delete --


def test_apply_get_delete_cycle(svc: K8sServiceImpl):
    manifest = _pod_manifest("myapp")
    svc.apply_json(manifest)
    assert svc.get_json("pod", "myapp") == manifest

    svc.delete("pod", "myapp")
    assert svc.get_json("pod", "myapp") is None


def test_get_nonexistent(svc: K8sServiceImpl):
    assert svc.get_json("pod", "nope") is None


def test_delete_nonexistent_is_idempotent(svc: K8sServiceImpl):
    svc.delete("pod", "nope")  # should not raise


def test_apply_overwrites(svc: K8sServiceImpl):
    m1 = _pod_manifest("p1")
    m2 = _pod_manifest("p1", labels={"version": "v2"})
    svc.apply_json(m1)
    svc.apply_json(m2)
    result = svc.get_json("pod", "p1")
    assert result is not None
    assert result["metadata"].get("labels", {}).get("version") == "v2"


def test_list_json_by_resource_type(svc: K8sServiceImpl):
    svc.apply_json(_pod_manifest("p1"))
    svc.apply_json(_pod_manifest("p2"))
    svc.apply_json(_deployment_manifest("d1"))

    pods = svc.list_json("pod")
    assert len(pods) == 2
    assert {m["metadata"]["name"] for m in pods} == {"p1", "p2"}

    deployments = svc.list_json("deployment")
    assert len(deployments) == 1


# -- Label filtering --


def test_list_json_label_filter(svc: K8sServiceImpl):
    svc.apply_json(_pod_manifest("p1", labels={"app": "web", "env": "prod"}))
    svc.apply_json(_pod_manifest("p2", labels={"app": "web", "env": "staging"}))
    svc.apply_json(_pod_manifest("p3", labels={"app": "worker"}))

    results = svc.list_json("pod", labels={"app": "web"})
    assert len(results) == 2

    results = svc.list_json("pod", labels={"app": "web", "env": "prod"})
    assert len(results) == 1
    assert results[0]["metadata"]["name"] == "p1"

    results = svc.list_json("pod", labels={"app": "nonexistent"})
    assert len(results) == 0


# -- Failure injection --


def test_inject_failure_apply(svc: K8sServiceImpl):
    svc.inject_failure("apply_json", KubectlError("scheduling failed"))
    with pytest.raises(KubectlError, match="scheduling failed"):
        svc.apply_json(_pod_manifest("p1"))

    # One-shot: next call succeeds
    svc.apply_json(_pod_manifest("p1"))
    assert svc.get_json("pod", "p1") is not None


def test_inject_failure_get(svc: K8sServiceImpl):
    svc.inject_failure("get_json", KubectlError("timeout"))
    with pytest.raises(KubectlError, match="timeout"):
        svc.get_json("pod", "p1")


def test_inject_failure_list(svc: K8sServiceImpl):
    svc.inject_failure("list_json", KubectlError("api error"))
    with pytest.raises(KubectlError, match="api error"):
        svc.list_json("pod")


def test_inject_failure_delete(svc: K8sServiceImpl):
    svc.inject_failure("delete", KubectlError("forbidden"))
    with pytest.raises(KubectlError, match="forbidden"):
        svc.delete("pod", "p1")


def test_clear_failure(svc: K8sServiceImpl):
    svc.inject_failure("apply_json", KubectlError("fail"))
    svc.clear_failure("apply_json")
    svc.apply_json(_pod_manifest("p1"))  # should not raise


# -- Node pool manipulation --


def test_remove_node_pool_causes_scheduling_failure(svc: K8sServiceImpl):
    manifest = _pod_manifest("p1", node_pool="gpu-pool")
    svc.apply_json(manifest)  # succeeds initially

    svc.remove_node_pool("gpu-pool")
    with pytest.raises(KubectlError, match="not found"):
        svc.apply_json(_pod_manifest("p2", node_pool="gpu-pool"))


def test_add_node_pool(svc: K8sServiceImpl):
    manifest = _pod_manifest("p1", node_pool="new-pool")
    with pytest.raises(KubectlError, match="not found"):
        svc.apply_json(manifest)

    svc.add_node_pool("new-pool")
    svc.apply_json(manifest)
    assert svc.get_json("pod", "p1") is not None


def test_add_node_pool_when_none():
    """add_node_pool initializes the set when it was None."""
    svc = K8sServiceImpl(available_node_pools=None)
    svc.add_node_pool("pool-a")
    # Now validation is enabled
    with pytest.raises(KubectlError, match="not found"):
        svc.apply_json(_pod_manifest("p1", node_pool="pool-b"))


# -- Logs and events --


def test_logs(svc: K8sServiceImpl):
    svc.set_logs("mypod", "line1\nline2\nline3")
    assert svc.logs("mypod", tail=2) == "line2\nline3"
    assert svc.logs("mypod") == "line1\nline2\nline3"


def test_logs_missing_pod(svc: K8sServiceImpl):
    assert svc.logs("nope") == ""


def test_events(svc: K8sServiceImpl):
    svc.add_event({"involvedObject": {"name": "pod-a"}, "reason": "Scheduled"})
    svc.add_event({"involvedObject": {"name": "pod-b"}, "reason": "Failed"})

    all_events = svc.get_events()
    assert len(all_events) == 2

    filtered = svc.get_events(field_selector="involvedObject.name=pod-a")
    assert len(filtered) == 1
    assert filtered[0]["reason"] == "Scheduled"


# -- Misc protocol methods --


def test_top_pod_existing(svc: K8sServiceImpl):
    svc.apply_json(_pod_manifest("p1"))
    result = svc.top_pod("p1")
    assert result is not None
    cpu, mem = result
    assert cpu > 0
    assert mem > 0


def test_top_pod_missing(svc: K8sServiceImpl):
    assert svc.top_pod("nope") is None


def test_exec_existing_pod(svc: K8sServiceImpl):
    svc.apply_json(_pod_manifest("p1"))
    result = svc.exec("p1", ["echo", "hello"])
    assert result.returncode == 0


def test_exec_missing_pod(svc: K8sServiceImpl):
    result = svc.exec("nope", ["echo", "hello"])
    assert result.returncode == 1


def test_namespace(svc: K8sServiceImpl):
    assert svc.namespace == "test-ns"


def test_default_namespace():
    svc = K8sServiceImpl()
    assert svc.namespace == "iris"


def test_stream_logs(svc: K8sServiceImpl):
    svc.set_logs("p1", "hello world")
    result = svc.stream_logs("p1")
    assert result.byte_offset > 0

    # No new content
    result2 = svc.stream_logs("p1", byte_offset=result.byte_offset)
    assert result2.byte_offset == result.byte_offset
    assert result2.lines == []


def test_case_insensitive_resource_type(svc: K8sServiceImpl):
    """Resource type lookup is case-insensitive."""
    svc.apply_json(_pod_manifest("p1"))
    assert svc.get_json("Pod", "p1") is not None
    assert svc.get_json("POD", "p1") is not None
    assert len(svc.list_json("Pod")) == 1


# ---------------------------------------------------------------------------
# Scheduling tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sched_svc() -> K8sServiceImpl:
    """K8sServiceImpl with a GPU node pool for scheduling tests."""
    svc = K8sServiceImpl(namespace="test-ns")
    return svc


def test_pod_scheduled_on_matching_node(sched_svc: K8sServiceImpl):
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
    result = sched_svc.get_json("pod", "gpu-pod")
    assert result is not None
    assert result["status"]["phase"] == "Running"
    assert result["spec"]["nodeName"] == "gpu-pool-0"


def test_pod_pending_no_matching_node(sched_svc: K8sServiceImpl):
    sched_svc.add_node_pool("cpu-pool")
    pod = _pod_manifest(
        "gpu-pod",
        node_selector={"accelerator": "nvidia-a100"},
        resources={"requests": {"nvidia.com/gpu": "1"}},
    )
    sched_svc.apply_json(pod)
    result = sched_svc.get_json("pod", "gpu-pod")
    assert result is not None
    assert result["status"]["phase"] == "Pending"


def test_pod_pending_insufficient_gpu(sched_svc: K8sServiceImpl):
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
    result = sched_svc.get_json("pod", "big-pod")
    assert result is not None
    assert result["status"]["phase"] == "Pending"


def test_toleration_required_for_tainted_node(sched_svc: K8sServiceImpl):
    sched_svc.add_node_pool(
        "tainted-pool",
        taints=[{"key": "dedicated", "value": "gpu", "effect": "NoSchedule"}],
    )

    # Without toleration → Pending
    pod_no_tol = _pod_manifest(
        "no-tol",
        node_selector={"cloud.google.com/gke-nodepool": "tainted-pool"},
    )
    sched_svc.apply_json(pod_no_tol)
    assert sched_svc.get_json("pod", "no-tol")["status"]["phase"] == "Pending"

    # With toleration → Running
    pod_with_tol = _pod_manifest(
        "with-tol",
        node_selector={"cloud.google.com/gke-nodepool": "tainted-pool"},
        tolerations=[{"key": "dedicated", "value": "gpu", "effect": "NoSchedule"}],
    )
    sched_svc.apply_json(pod_with_tol)
    assert sched_svc.get_json("pod", "with-tol")["status"]["phase"] == "Running"


def test_resource_commitment_tracking(sched_svc: K8sServiceImpl):
    sched_svc.add_node_pool(
        "gpu-pool",
        resources=FakeNodeResources(gpu_count=4),
    )

    # First 2-GPU pod → Running
    pod1 = _pod_manifest(
        "pod1",
        node_selector={"cloud.google.com/gke-nodepool": "gpu-pool"},
        resources={"requests": {"nvidia.com/gpu": "2"}},
    )
    sched_svc.apply_json(pod1)
    assert sched_svc.get_json("pod", "pod1")["status"]["phase"] == "Running"

    # Second 2-GPU pod → Running (4 total committed)
    pod2 = _pod_manifest(
        "pod2",
        node_selector={"cloud.google.com/gke-nodepool": "gpu-pool"},
        resources={"requests": {"nvidia.com/gpu": "2"}},
    )
    sched_svc.apply_json(pod2)
    assert sched_svc.get_json("pod", "pod2")["status"]["phase"] == "Running"

    # Third 2-GPU pod → Pending (no capacity)
    pod3 = _pod_manifest(
        "pod3",
        node_selector={"cloud.google.com/gke-nodepool": "gpu-pool"},
        resources={"requests": {"nvidia.com/gpu": "2"}},
    )
    sched_svc.apply_json(pod3)
    assert sched_svc.get_json("pod", "pod3")["status"]["phase"] == "Pending"


def test_list_nodes_returns_node_dicts(sched_svc: K8sServiceImpl):
    sched_svc.add_node_pool(
        "my-pool",
        resources=FakeNodeResources(gpu_count=8, cpu_millicores=16000),
    )
    nodes = sched_svc.list_json("nodes", cluster_scoped=True)
    assert len(nodes) == 1
    node = nodes[0]
    assert node["metadata"]["name"] == "my-pool-0"
    assert node["status"]["allocatable"]["cpu"] == "16000m"
    assert node["status"]["allocatable"]["nvidia.com/gpu"] == "8"


def test_add_node_pool_with_attributes(sched_svc: K8sServiceImpl):
    sched_svc.add_node_pool(
        "big-pool",
        node_count=3,
        labels={"zone": "us-central1-a"},
        resources=FakeNodeResources(cpu_millicores=8000),
    )
    nodes = sched_svc.list_json("nodes", cluster_scoped=True)
    assert len(nodes) == 3
    for node in nodes:
        assert node["metadata"]["labels"]["zone"] == "us-central1-a"
        assert node["metadata"]["labels"]["cloud.google.com/gke-nodepool"] == "big-pool"


def test_transition_pod_helper(sched_svc: K8sServiceImpl):
    sched_svc.add_node_pool("pool")
    pod = _pod_manifest("p1", node_selector={"cloud.google.com/gke-nodepool": "pool"})
    sched_svc.apply_json(pod)
    assert sched_svc.get_json("pod", "p1")["status"]["phase"] == "Running"

    sched_svc.transition_pod("p1", "Failed", exit_code=1, reason="OOMKilled")
    result = sched_svc.get_json("pod", "p1")
    assert result["status"]["phase"] == "Failed"
    assert result["status"]["containerStatuses"][0]["state"]["terminated"]["exitCode"] == 1
    assert result["status"]["containerStatuses"][0]["state"]["terminated"]["reason"] == "OOMKilled"


def test_delete_pod_releases_resources(sched_svc: K8sServiceImpl):
    sched_svc.add_node_pool(
        "gpu-pool",
        resources=FakeNodeResources(gpu_count=2),
    )

    # Schedule a 2-GPU pod
    pod1 = _pod_manifest(
        "pod1",
        node_selector={"cloud.google.com/gke-nodepool": "gpu-pool"},
        resources={"requests": {"nvidia.com/gpu": "2"}},
    )
    sched_svc.apply_json(pod1)
    assert sched_svc.get_json("pod", "pod1")["status"]["phase"] == "Running"

    # Second pod can't fit
    pod2 = _pod_manifest(
        "pod2",
        node_selector={"cloud.google.com/gke-nodepool": "gpu-pool"},
        resources={"requests": {"nvidia.com/gpu": "2"}},
    )
    sched_svc.apply_json(pod2)
    assert sched_svc.get_json("pod", "pod2")["status"]["phase"] == "Pending"

    # Delete first pod, freeing resources
    sched_svc.delete("pod", "pod1")

    # Now a new pod can schedule
    pod3 = _pod_manifest(
        "pod3",
        node_selector={"cloud.google.com/gke-nodepool": "gpu-pool"},
        resources={"requests": {"nvidia.com/gpu": "2"}},
    )
    sched_svc.apply_json(pod3)
    assert sched_svc.get_json("pod", "pod3")["status"]["phase"] == "Running"


def test_failed_scheduling_event_generated(sched_svc: K8sServiceImpl):
    sched_svc.add_node_pool("cpu-pool")
    pod = _pod_manifest(
        "unschedulable",
        node_selector={"accelerator": "nonexistent"},
    )
    sched_svc.apply_json(pod)
    assert sched_svc.get_json("pod", "unschedulable")["status"]["phase"] == "Pending"

    events = sched_svc.get_events(field_selector="involvedObject.name=unschedulable")
    assert len(events) == 1
    assert events[0]["reason"] == "FailedScheduling"


# ---------------------------------------------------------------------------
# port_forward tests
# ---------------------------------------------------------------------------


def test_port_forward_yields_url(svc: K8sServiceImpl):
    with svc.port_forward("my-svc", 8080) as url:
        assert url.startswith("http://127.0.0.1:")
        assert "19999" in url


def test_port_forward_with_explicit_port(svc: K8sServiceImpl):
    with svc.port_forward("my-svc", 8080, local_port=12345) as url:
        assert url == "http://127.0.0.1:12345"


def test_port_forward_failure_injection(svc: K8sServiceImpl):
    svc.inject_failure("port_forward", KubectlError("tunnel failed", 1))
    with pytest.raises(KubectlError):
        with svc.port_forward("my-svc", 8080):
            pass


# ---------------------------------------------------------------------------
# CLOUD mode tests
# ---------------------------------------------------------------------------


def test_cloud_mode_construction():
    svc = K8sServiceImpl(namespace="test", mode=ServiceMode.CLOUD)
    assert svc.mode == ServiceMode.CLOUD
    assert svc.namespace == "test"


def test_cloud_mode_satisfies_k8s_service_protocol():
    svc = K8sServiceImpl(namespace="test", mode=ServiceMode.CLOUD)
    assert isinstance(svc, K8sService)


def test_cloud_mode_delegates_to_kubectl():
    svc = K8sServiceImpl(namespace="test", mode=ServiceMode.CLOUD)
    mock_kubectl = MagicMock()
    svc._kubectl = mock_kubectl

    manifest = {"kind": "Pod", "metadata": {"name": "test"}}
    svc.apply_json(manifest)
    mock_kubectl.apply_json.assert_called_once_with(manifest)

    svc.get_json("pod", "test")
    mock_kubectl.get_json.assert_called_once_with("pod", "test", cluster_scoped=False)

    svc.delete("pod", "test")
    mock_kubectl.delete.assert_called_once_with("pod", "test", cluster_scoped=False, force=False, wait=True)
