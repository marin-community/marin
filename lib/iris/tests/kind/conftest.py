# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for Kind-based Kubernetes integration tests.

Manages a Kind cluster (one per session) and per-test namespaces.
Requires: Docker daemon, `kind` binary, `kubectl` binary.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass

import pytest

from iris.cluster.k8s.kubectl import Kubectl

logger = logging.getLogger(__name__)

# Cluster name shared across all tests in the session.
_KIND_CLUSTER_NAME = "iris-test"

# How long to wait for the Kind cluster to become ready (seconds).
_CLUSTER_CREATE_TIMEOUT = 120

# How long to wait for a pod phase transition when polling (seconds).
_POD_WAIT_TIMEOUT = 30

# Polling interval for pod phase checks (seconds).
_POD_POLL_INTERVAL = 0.5


def _has_kind() -> bool:
    return shutil.which("kind") is not None


def _has_docker() -> bool:
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _kind_cluster_exists(name: str) -> bool:
    result = subprocess.run(
        ["kind", "get", "clusters"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return name in result.stdout.splitlines()


@dataclass
class KindCluster:
    """Handle to a running Kind cluster with a per-test namespace."""

    kubeconfig: str
    namespace: str
    kubectl: Kubectl

    def apply_manifest(self, manifest: dict) -> None:
        """Apply a K8s manifest dict via kubectl."""
        self.kubectl.apply_json(manifest)

    def get_pod_phase(self, pod_name: str) -> str | None:
        """Get the phase of a pod, or None if not found."""
        pod = self.kubectl.get_json("pod", pod_name)
        if pod is None:
            return None
        return pod.get("status", {}).get("phase", "Unknown")

    def wait_for_pod_phase(
        self,
        pod_name: str,
        target_phases: set[str],
        timeout: float = _POD_WAIT_TIMEOUT,
    ) -> str:
        """Poll until the pod reaches one of the target phases.

        Returns the phase reached. Raises TimeoutError if not reached within timeout.
        """
        deadline = time.monotonic() + timeout
        last_phase = None
        while time.monotonic() < deadline:
            phase = self.get_pod_phase(pod_name)
            if phase in target_phases:
                return phase
            last_phase = phase
            time.sleep(_POD_POLL_INTERVAL)
        raise TimeoutError(
            f"Pod {pod_name} did not reach {target_phases} within {timeout}s " f"(last phase: {last_phase})"
        )

    def pod_is_unschedulable(self, pod_name: str) -> bool:
        """Check if a pod has an Unschedulable condition."""
        pod = self.kubectl.get_json("pod", pod_name)
        if pod is None:
            return False
        conditions = pod.get("status", {}).get("conditions", [])
        for cond in conditions:
            if cond.get("reason") == "Unschedulable":
                return True
        return False

    def wait_for_unschedulable(
        self,
        pod_name: str,
        timeout: float = _POD_WAIT_TIMEOUT,
    ) -> None:
        """Poll until the pod has an Unschedulable condition.

        Raises TimeoutError if not seen within timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.pod_is_unschedulable(pod_name):
                return
            time.sleep(_POD_POLL_INTERVAL)
        raise TimeoutError(f"Pod {pod_name} was not marked Unschedulable within {timeout}s")

    def label_node(self, node_name: str, labels: dict[str, str]) -> None:
        """Add labels to a node."""
        label_args = [f"{k}={v}" for k, v in labels.items()]
        result = self.kubectl.run(
            ["label", "node", node_name, "--overwrite", *label_args],
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to label node {node_name}: {result.stderr}")

    def taint_node(self, node_name: str, taint: str) -> None:
        """Add a taint to a node. Format: key=value:Effect"""
        result = self.kubectl.run(["taint", "node", node_name, taint, "--overwrite"])
        if result.returncode != 0:
            raise RuntimeError(f"Failed to taint node {node_name}: {result.stderr}")

    def get_node_names(self) -> list[str]:
        """Get names of all nodes in the cluster."""
        nodes = self.kubectl.list_json("nodes", cluster_scoped=True)
        return [n["metadata"]["name"] for n in nodes]

    def delete_pod(self, pod_name: str) -> None:
        """Delete a pod."""
        self.kubectl.delete("pod", pod_name, force=True)


@pytest.fixture(scope="session")
def kind_kubeconfig(tmp_path_factory):
    """Create a Kind cluster for the test session and return the kubeconfig path.

    Tears down the cluster after all tests complete.
    """
    if not _has_kind():
        pytest.skip("kind binary not found")
    if not _has_docker():
        pytest.skip("Docker daemon not available")

    kubeconfig = str(tmp_path_factory.mktemp("kind") / "kubeconfig")

    # Create cluster if it doesn't already exist (allows re-running tests).
    if not _kind_cluster_exists(_KIND_CLUSTER_NAME):
        logger.info("Creating Kind cluster %r", _KIND_CLUSTER_NAME)
        # Use a config with a single control-plane node (doubles as worker).
        config = {
            "kind": "Cluster",
            "apiVersion": "kind.x-k8s.io/v1alpha4",
            "nodes": [
                {"role": "control-plane"},
            ],
        }
        config_json = json.dumps(config)
        result = subprocess.run(
            [
                "kind",
                "create",
                "cluster",
                "--name",
                _KIND_CLUSTER_NAME,
                "--config",
                "-",
                "--kubeconfig",
                kubeconfig,
                "--wait",
                f"{_CLUSTER_CREATE_TIMEOUT}s",
            ],
            input=config_json,
            capture_output=True,
            text=True,
            timeout=_CLUSTER_CREATE_TIMEOUT + 30,
        )
        if result.returncode != 0:
            pytest.fail(f"Kind cluster creation failed: {result.stderr}")
    else:
        logger.info("Reusing existing Kind cluster %r", _KIND_CLUSTER_NAME)
        result = subprocess.run(
            [
                "kind",
                "export",
                "kubeconfig",
                "--name",
                _KIND_CLUSTER_NAME,
                "--kubeconfig",
                kubeconfig,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            pytest.fail(f"Kind kubeconfig export failed: {result.stderr}")

    yield kubeconfig

    # Tear down cluster after session.
    logger.info("Deleting Kind cluster %r", _KIND_CLUSTER_NAME)
    subprocess.run(
        ["kind", "delete", "cluster", "--name", _KIND_CLUSTER_NAME],
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture
def kind_cluster(kind_kubeconfig) -> KindCluster:
    """Per-test fixture: creates a unique namespace and yields a KindCluster handle.

    The namespace is deleted after the test to avoid cross-test pollution.
    """
    namespace = f"iris-test-{uuid.uuid4().hex[:8]}"
    kubectl = Kubectl(namespace=namespace, kubeconfig_path=kind_kubeconfig)

    # Create namespace.
    ns_manifest = {
        "apiVersion": "v1",
        "kind": "Namespace",
        "metadata": {"name": namespace},
    }
    kubectl.apply_json(ns_manifest)

    cluster = KindCluster(
        kubeconfig=kind_kubeconfig,
        namespace=namespace,
        kubectl=kubectl,
    )

    yield cluster

    # Cleanup namespace.
    kubectl.delete("namespace", namespace, cluster_scoped=True, wait=False)
