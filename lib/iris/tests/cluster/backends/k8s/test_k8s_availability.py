# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Federation availability inferred from the cached kubectl cluster sync: GPU
free/total counting on :class:`ClusterState` and its attribution to a backend's
advertised device variant in ``K8sTaskProvider.resource_capacity``."""

from iris.cluster.backends.k8s.tasks import ClusterState, K8sTaskProvider
from iris.cluster.controller.backend import DeviceCapacity
from iris.cluster.platforms.k8s.fake import InMemoryK8sService
from iris.cluster.types import WellKnownAttribute

_GPU = "nvidia.com/gpu"


def _node(name: str, gpus: int) -> dict:
    return {"metadata": {"name": name}, "status": {"allocatable": {_GPU: str(gpus)}}}


def _pod(name: str, gpus: int, *, phase: str = "Running") -> dict:
    return {
        "metadata": {"name": name},
        "status": {"phase": phase},
        "spec": {"containers": [{"resources": {"requests": {_GPU: str(gpus)}}}]},
    }


def _state(nodes: list[dict], pods: list[dict]) -> ClusterState:
    state = ClusterState()
    state.update(pods=pods, nodes=nodes, workloads=[], node_pools=[])
    return state


def test_gpu_capacity_is_allocatable_minus_running_requests():
    state = _state([_node("n1", 8), _node("n2", 8)], [_pod("a", 8), _pod("b", 2)])
    assert state.gpu_capacity() == DeviceCapacity(free=6, total=16)  # 16 allocatable - 10 requested


def test_gpu_capacity_ignores_terminal_pods():
    # Succeeded/Failed pods have released their GPUs even if still listed.
    state = _state(
        [_node("n1", 8)],
        [_pod("done", 8, phase="Succeeded"), _pod("dead", 4, phase="Failed"), _pod("live", 2)],
    )
    assert state.gpu_capacity() == DeviceCapacity(free=6, total=8)


def test_gpu_capacity_never_negative_when_oversubscribed():
    # Requests can exceed allocatable transiently (pending pods); the free hint
    # floors at 0 while the total still reports allocatable.
    state = _state([_node("n1", 8)], [_pod("a", 8), _pod("b", 8)])
    assert state.gpu_capacity() == DeviceCapacity(free=0, total=8)


def test_gpu_capacity_zero_without_gpu_nodes():
    state = _state([{"metadata": {"name": "cpu"}, "status": {"allocatable": {"cpu": "16"}}}], [])
    assert state.gpu_capacity() == DeviceCapacity(free=0, total=0)


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
    assert provider.resource_capacity() == {"h100": DeviceCapacity(free=6, total=8)}  # lowercased, 8 - 2


def test_resource_capacity_is_unset_when_the_variant_is_ambiguous():
    # Two variants: free GPUs cannot be attributed to one, so fall back to shape-only.
    assert _provider({WellKnownAttribute.DEVICE_VARIANT: {"h100", "a100"}}).resource_capacity() is None


def test_resource_capacity_is_unset_without_an_advertised_variant():
    assert _provider({}).resource_capacity() is None
