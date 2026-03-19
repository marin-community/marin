# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for K8sServiceImpl — in-memory K8sService for DRY_RUN/LOCAL modes."""

from __future__ import annotations

import pytest

from iris.cluster.k8s.k8s_service import K8sService
from iris.cluster.k8s.k8s_service_impl import K8sServiceImpl
from iris.cluster.k8s.kubectl import KubectlError


def _pod_manifest(
    name: str,
    labels: dict[str, str] | None = None,
    node_pool: str | None = None,
    resources: dict | None = None,
) -> dict:
    """Build a minimal Pod manifest for testing."""
    spec: dict = {"containers": [{"name": "main", "image": "busybox"}]}
    if node_pool:
        spec["nodeSelector"] = {"cloud.google.com/gke-nodepool": node_pool}
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
