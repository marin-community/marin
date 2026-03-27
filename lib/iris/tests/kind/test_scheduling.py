# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Kind-based integration tests for K8s scheduling correctness.

These tests apply pod manifests built by _build_pod_manifest() to a real
Kind cluster and assert that the K8s scheduler handles topology constraints,
node selectors, taints, and resource limits correctly.

Requires: Docker daemon, `kind` binary, `kubectl` binary.
Run with: uv run pytest -m kind lib/iris/tests/kind/
"""

from __future__ import annotations

import pytest

from iris.cluster.k8s.provider import PodConfig, _build_pod_manifest
from iris.rpc import cluster_pb2

pytestmark = [pytest.mark.kind, pytest.mark.slow]


def _make_run_req(
    task_id: str,
    attempt_id: int = 0,
    num_tasks: int = 1,
    cpu_mc: int = 100,
    memory_bytes: int = 64 * 1024 * 1024,
) -> cluster_pb2.Worker.RunTaskRequest:
    """Build a minimal RunTaskRequest for scheduling tests."""
    req = cluster_pb2.Worker.RunTaskRequest()
    req.task_id = task_id
    req.attempt_id = attempt_id
    req.num_tasks = num_tasks
    req.entrypoint.run_command.argv.extend(["sleep", "3600"])
    req.environment.env_vars["IRIS_JOB_ID"] = "test-job"
    req.resources.cpu_millicores = cpu_mc
    req.resources.memory_bytes = memory_bytes
    return req


def _add_constraint(req: cluster_pb2.Worker.RunTaskRequest, key: str, value: str) -> None:
    c = req.constraints.add()
    c.key = key
    c.op = cluster_pb2.CONSTRAINT_OP_EQ
    c.value.string_value = value


def test_pod_scheduled_on_labeled_node(kind_cluster):
    """Pod with nodeSelector matching a node label gets scheduled."""
    nodes = kind_cluster.get_node_names()
    assert nodes, "Kind cluster has no nodes"
    node = nodes[0]

    # Label the node with a pool label that iris uses for nodeSelector.
    kind_cluster.label_node(node, {"iris.pool": "cpu-pool"})

    req = _make_run_req("/sched-test/task-0")
    _add_constraint(req, "pool", "cpu-pool")

    config = PodConfig(namespace=kind_cluster.namespace, default_image="busybox:1.36")
    manifest = _build_pod_manifest(req, config)

    kind_cluster.apply_manifest(manifest)
    pod_name = manifest["metadata"]["name"]

    phase = kind_cluster.wait_for_pod_phase(pod_name, {"Running", "Succeeded"})
    assert phase in {"Running", "Succeeded"}


def test_pod_unschedulable_with_wrong_node_selector(kind_cluster):
    """Pod with a nodeSelector that no node satisfies stays Pending/Unschedulable."""
    req = _make_run_req("/sched-noselector/task-0")
    _add_constraint(req, "pool", "nonexistent-pool-xyz")

    config = PodConfig(namespace=kind_cluster.namespace, default_image="busybox:1.36")
    manifest = _build_pod_manifest(req, config)

    kind_cluster.apply_manifest(manifest)
    pod_name = manifest["metadata"]["name"]

    kind_cluster.wait_for_unschedulable(pod_name)
    assert kind_cluster.get_pod_phase(pod_name) == "Pending"


def test_pod_unschedulable_with_taint_no_toleration(kind_cluster):
    """Pod without matching toleration cannot schedule on a tainted node."""
    nodes = kind_cluster.get_node_names()
    assert nodes, "Kind cluster has no nodes"
    node = nodes[0]

    # Taint all nodes with a NoSchedule taint. The test pod has no toleration for it.
    kind_cluster.taint_node(node, "iris-test/block=true:NoSchedule")

    req = _make_run_req("/sched-taint/task-0")
    config = PodConfig(namespace=kind_cluster.namespace, default_image="busybox:1.36")
    manifest = _build_pod_manifest(req, config)

    kind_cluster.apply_manifest(manifest)
    pod_name = manifest["metadata"]["name"]

    kind_cluster.wait_for_unschedulable(pod_name)
    assert kind_cluster.get_pod_phase(pod_name) == "Pending"


def test_pod_scheduled_with_toleration(kind_cluster):
    """Pod with a matching toleration can schedule on a tainted node."""
    nodes = kind_cluster.get_node_names()
    assert nodes, "Kind cluster has no nodes"
    node = nodes[0]

    kind_cluster.taint_node(node, "nvidia.com/gpu=present:NoSchedule")

    req = _make_run_req("/sched-tolerate/task-0")
    # Request GPU resources so the provider adds the nvidia toleration.
    req.resources.device.gpu.count = 1
    req.resources.device.gpu.type = "nvidia-test"

    config = PodConfig(namespace=kind_cluster.namespace, default_image="busybox:1.36")
    manifest = _build_pod_manifest(req, config)

    # Remove the nvidia.com/gpu resource limit since Kind nodes don't have GPUs.
    # We only care that the toleration allows scheduling.
    container = manifest["spec"]["containers"][0]
    if "resources" in container and "limits" in container["resources"]:
        container["resources"]["limits"].pop("nvidia.com/gpu", None)
        container["resources"]["limits"].pop("rdma/ib", None)
        if not container["resources"]["limits"]:
            del container["resources"]["limits"]
        if not container["resources"]:
            del container["resources"]

    kind_cluster.apply_manifest(manifest)
    pod_name = manifest["metadata"]["name"]

    phase = kind_cluster.wait_for_pod_phase(pod_name, {"Running", "Succeeded"})
    assert phase in {"Running", "Succeeded"}


def test_pod_pending_on_resource_exhaustion(kind_cluster):
    """Pod requesting more CPU than available stays Pending."""
    req = _make_run_req("/sched-exhaust/task-0", cpu_mc=999_000, memory_bytes=64 * 1024 * 1024)

    config = PodConfig(namespace=kind_cluster.namespace, default_image="busybox:1.36")
    manifest = _build_pod_manifest(req, config)

    kind_cluster.apply_manifest(manifest)
    pod_name = manifest["metadata"]["name"]

    kind_cluster.wait_for_unschedulable(pod_name)
    assert kind_cluster.get_pod_phase(pod_name) == "Pending"


def test_colocation_affinity_with_valid_topology_key(kind_cluster):
    """Multi-task job with valid topology key produces pods with podAffinity set."""
    nodes = kind_cluster.get_node_names()
    assert nodes, "Kind cluster has no nodes"
    node = nodes[0]

    # Label node with the topology key so affinity can be satisfied.
    topology_key = "kubernetes.io/hostname"
    kind_cluster.label_node(node, {topology_key: node})

    config = PodConfig(
        namespace=kind_cluster.namespace,
        default_image="busybox:1.36",
        colocation_topology_key=topology_key,
    )

    # Create two sibling tasks (num_tasks=2) in the same job.
    req0 = _make_run_req("/sched-coloc/task-0", num_tasks=2)
    req1 = _make_run_req("/sched-coloc/task-1", attempt_id=0, num_tasks=2)

    manifest0 = _build_pod_manifest(req0, config)
    manifest1 = _build_pod_manifest(req1, config)

    # Verify affinity is present in the manifest.
    assert "affinity" in manifest0["spec"], "Expected podAffinity for multi-task job"
    assert "affinity" in manifest1["spec"]

    kind_cluster.apply_manifest(manifest0)
    kind_cluster.apply_manifest(manifest1)

    pod0 = manifest0["metadata"]["name"]
    pod1 = manifest1["metadata"]["name"]

    phase0 = kind_cluster.wait_for_pod_phase(pod0, {"Running", "Succeeded"})
    phase1 = kind_cluster.wait_for_pod_phase(pod1, {"Running", "Succeeded"})
    assert phase0 in {"Running", "Succeeded"}
    assert phase1 in {"Running", "Succeeded"}


def test_single_task_job_no_affinity(kind_cluster):
    """Single-task job (num_tasks=1) should not have podAffinity set."""
    config = PodConfig(
        namespace=kind_cluster.namespace,
        default_image="busybox:1.36",
        colocation_topology_key="kubernetes.io/hostname",
    )
    req = _make_run_req("/sched-single/task-0", num_tasks=1)
    manifest = _build_pod_manifest(req, config)

    assert "affinity" not in manifest["spec"], "Single-task job should not have affinity"

    kind_cluster.apply_manifest(manifest)
    pod_name = manifest["metadata"]["name"]

    phase = kind_cluster.wait_for_pod_phase(pod_name, {"Running", "Succeeded"})
    assert phase in {"Running", "Succeeded"}
