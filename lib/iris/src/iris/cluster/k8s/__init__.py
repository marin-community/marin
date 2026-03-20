# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared Kubernetes helpers for Iris cluster components."""

from iris.cluster.k8s.k8s_service import K8sService
from iris.cluster.k8s.k8s_service_impl import DryRunK8sService, K8sServiceImpl, LocalK8sService
from iris.cluster.k8s.k8s_types import KubectlError
from iris.cluster.service_mode import ServiceMode


def create_k8s_service(
    mode: ServiceMode,
    namespace: str = "iris",
    *,
    kubeconfig_path: str | None = None,
    timeout: float = 60.0,
    available_node_pools: list[str] | None = None,
) -> K8sService:
    """Create a K8sService for the given mode."""
    if mode == ServiceMode.CLOUD:
        from iris.cluster.k8s.kubectl import Kubectl

        return Kubectl(namespace=namespace, kubeconfig_path=kubeconfig_path, timeout=timeout)
    if mode == ServiceMode.LOCAL:
        return LocalK8sService(namespace=namespace, available_node_pools=available_node_pools)
    return DryRunK8sService(namespace=namespace, available_node_pools=available_node_pools)
