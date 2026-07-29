# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Federation availability inferred from the cached kubectl cluster sync: GPU
free/total counting on :class:`ClusterState` and its attribution to a backend's
advertised device variant in ``K8sTaskProvider.resource_capacity``."""

from iris.cluster.backends.k8s.tasks import ClusterState, K8sTaskProvider
from iris.cluster.controller.backend import DeviceCapacity
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.platforms.k8s.types import IRIS_PRIORITY_CLASS_BATCH, IRIS_PRIORITY_CLASS_INTERACTIVE
from iris.cluster.types import WellKnownAttribute
from iris.rpc import job_pb2

_GPU = "nvidia.com/gpu"

_PRIORITY_CLASSES = {
    job_pb2.PRIORITY_BAND_INTERACTIVE: IRIS_PRIORITY_CLASS_INTERACTIVE,
    job_pb2.PRIORITY_BAND_BATCH: IRIS_PRIORITY_CLASS_BATCH,
}


def _node(name: str, gpus: int) -> dict:
    return {"metadata": {"name": name}, "status": {"allocatable": {_GPU: str(gpus)}}}


def _pod(
    name: str,
    gpus: int,
    *,
    phase: str = "Running",
    node: str = "n1",
    priority_class: str = IRIS_PRIORITY_CLASS_INTERACTIVE,
    gated: bool = False,
) -> dict:
    """A managed pod. ``gated`` models one Kueue has not admitted (queued, holds nothing)."""
    spec: dict = {"containers": [{"resources": {"requests": {_GPU: str(gpus)}}}]}
    if node:
        spec["nodeName"] = node
    if priority_class:
        spec["priorityClassName"] = priority_class
    if gated:
        spec["schedulingGates"] = [{"name": "kueue.x-k8s.io/admission"}]
    return {"metadata": {"name": name}, "status": {"phase": phase}, "spec": spec}


def _state(nodes: list[dict], pods: list[dict]) -> ClusterState:
    state = ClusterState()
    state.update(pods=pods, nodes=nodes, workloads=[], node_pools=[])
    return state


def test_gpu_capacity_is_allocatable_minus_admitted_requests():
    state = _state([_node("n1", 8), _node("n2", 8)], [_pod("a", 8), _pod("b", 2, node="n2")])
    capacity = state.gpu_capacity(_PRIORITY_CLASSES)
    assert (capacity.free, capacity.total) == (6, 16)  # 16 allocatable - 10 held


def test_gpu_capacity_ignores_terminal_pods():
    # Succeeded/Failed pods have released their GPUs even if still listed.
    state = _state(
        [_node("n1", 8)],
        [_pod("done", 8, phase="Succeeded"), _pod("dead", 4, phase="Failed"), _pod("live", 2)],
    )
    capacity = state.gpu_capacity(_PRIORITY_CLASSES)
    assert (capacity.free, capacity.total) == (6, 8)


def test_gpu_capacity_ignores_pods_kueue_has_not_admitted():
    # A gated pod is still in the peer's queue and holds no GPU: it must not suppress
    # the free count, which is what kept federated jobs out of a peer with a long queue.
    state = _state([_node("n1", 8)], [_pod("queued", 8, phase="Pending", node="", gated=True), _pod("live", 2)])
    capacity = state.gpu_capacity(_PRIORITY_CLASSES)
    assert (capacity.free, capacity.total) == (6, 8)
    assert capacity.held_by_band == {job_pb2.PRIORITY_BAND_INTERACTIVE: 2}


def test_gpu_capacity_counts_admitted_pods_before_they_bind():
    # Kueue released the gate, so its quota is reserved even while the pod waits for a
    # node. Reporting those GPUs free would attract handoffs the peer cannot admit.
    state = _state([_node("n1", 8)], [_pod("admitted", 8, phase="Pending", node="")])
    capacity = state.gpu_capacity(_PRIORITY_CLASSES)
    assert (capacity.free, capacity.total) == (0, 8)
    assert capacity.held_by_band == {job_pb2.PRIORITY_BAND_INTERACTIVE: 8}


def test_gpu_capacity_splits_held_gpus_by_priority_band():
    state = _state(
        [_node("n1", 8), _node("n2", 8)],
        [
            _pod("batch", 6, priority_class=IRIS_PRIORITY_CLASS_BATCH),
            _pod("interactive", 8, node="n2"),
            _pod("unknown-class", 2, priority_class="third-party"),
        ],
    )
    capacity = state.gpu_capacity(_PRIORITY_CLASSES)
    # All 16 GPUs are held; the pod on a class Iris does not own is held but unattributed.
    assert (capacity.free, capacity.total) == (0, 16)
    assert capacity.held_by_band == {
        job_pb2.PRIORITY_BAND_BATCH: 6,
        job_pb2.PRIORITY_BAND_INTERACTIVE: 8,
    }


def test_gpu_capacity_never_negative_when_oversubscribed():
    # Admitted requests can exceed allocatable transiently; the free hint floors at 0
    # while the total still reports allocatable.
    state = _state([_node("n1", 8)], [_pod("a", 8), _pod("b", 8)])
    assert state.gpu_capacity(_PRIORITY_CLASSES).free == 0


def test_gpu_capacity_zero_without_gpu_nodes():
    state = _state([{"metadata": {"name": "cpu"}, "status": {"allocatable": {"cpu": "16"}}}], [])
    assert state.gpu_capacity(_PRIORITY_CLASSES) == DeviceCapacity(free=0, total=0)


def _provider(advertised: dict[str, set[str]]) -> K8sTaskProvider:
    provider = K8sTaskProvider(
        kubectl=InMemoryK8sService(namespace="test-ns"),
        namespace="test-ns",
        default_image="img",
        advertised=advertised,
    )
    provider._cluster_state.update(pods=[_pod("a", 2)], nodes=[_node("n1", 8)], workloads=[], node_pools=[])
    return provider


def test_resource_capacity_attributes_gpus_to_the_sole_variant():
    provider = _provider({WellKnownAttribute.DEVICE_VARIANT: {"H100"}})
    assert provider.resource_capacity() == {  # lowercased, 8 - 2
        "h100": DeviceCapacity(free=6, total=8, held_by_band={job_pb2.PRIORITY_BAND_INTERACTIVE: 2})
    }


def test_resource_capacity_is_unset_when_the_variant_is_ambiguous():
    # Two variants: free GPUs cannot be attributed to one, so fall back to shape-only.
    assert _provider({WellKnownAttribute.DEVICE_VARIANT: {"h100", "a100"}}).resource_capacity() is None


def test_resource_capacity_is_unset_without_an_advertised_variant():
    assert _provider({}).resource_capacity() is None
