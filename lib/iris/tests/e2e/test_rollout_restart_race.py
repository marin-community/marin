# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Rollout-restart race condition in ``_wait_for_deployment_ready``.

``cluster start`` calls ``kubectl rollout restart`` then waits only for
``availableReplicas >= 1``.  When a Deployment already exists this returns
while the old pod is still terminating — a port-forward through the Service
can land on the dying pod and get connection-refused.

Two tests:
- ``test_availability_check_passes_during_rollout``: proves the bug exists
  (availableReplicas check passes while >1 pods are alive).
- ``test_rollout_status_waits_for_completion``: proves ``kubectl rollout
  status`` (the fix) blocks until exactly 1 pod remains.
"""

from __future__ import annotations

import json
import subprocess
import time
import uuid

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.slow]

DEPLOY = {
    "apiVersion": "apps/v1",
    "kind": "Deployment",
    "metadata": {"name": "race-server", "labels": {"app": "race-server"}},
    "spec": {
        "replicas": 1,
        "selector": {"matchLabels": {"app": "race-server"}},
        "strategy": {"type": "RollingUpdate", "rollingUpdate": {"maxSurge": 1, "maxUnavailable": 0}},
        "template": {
            "metadata": {"labels": {"app": "race-server"}},
            "spec": {
                "terminationGracePeriodSeconds": 30,
                "containers": [
                    {
                        "name": "server",
                        "image": "nginx:alpine",
                        "ports": [{"containerPort": 80}],
                        "readinessProbe": {
                            "httpGet": {"path": "/", "port": 80},
                            "initialDelaySeconds": 2,
                            "periodSeconds": 2,
                        },
                    }
                ],
            },
        },
    },
}


def _kubectl(*args: str, namespace: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["kubectl", "-n", namespace, *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _apply(manifest: dict, namespace: str) -> None:
    manifest = {**manifest, "metadata": {**manifest["metadata"], "namespace": namespace}}
    subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=json.dumps(manifest).encode(),
        capture_output=True,
        check=True,
        timeout=30,
    )


def _wait_available_replicas(namespace: str, timeout: float = 120) -> None:
    """Same logic as ``_wait_for_deployment_ready`` in coreweave.py."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = _kubectl("get", "deployment", "race-server", "-o", "json", namespace=namespace)
        if result.returncode == 0:
            available = json.loads(result.stdout).get("status", {}).get("availableReplicas", 0)
            if available and available >= 1:
                return
        time.sleep(1)
    raise TimeoutError("Deployment did not become available")


def _pod_count(namespace: str) -> int:
    result = _kubectl("get", "pods", "-l", "app=race-server", "--no-headers", namespace=namespace)
    return len([line for line in result.stdout.strip().splitlines() if line])


def _running_pod_count(namespace: str) -> int:
    result = _kubectl("get", "pods", "-l", "app=race-server", "--no-headers", namespace=namespace)
    return len([line for line in result.stdout.strip().splitlines() if line and "Terminating" not in line])


@pytest.mark.timeout(180)
def test_availability_check_passes_during_rollout(k8s_cluster):
    """availableReplicas >= 1 returns true while old pod is still alive."""
    namespace = f"race-{uuid.uuid4().hex[:8]}"
    subprocess.run(["kubectl", "create", "namespace", namespace], check=True, capture_output=True, timeout=30)

    try:
        _apply(DEPLOY, namespace)
        _wait_available_replicas(namespace)
        assert _pod_count(namespace) == 1

        _kubectl("rollout", "restart", "deployment/race-server", namespace=namespace)
        _wait_available_replicas(namespace)

        assert _pod_count(namespace) > 1, (
            "Expected >1 pods after availability check (old pod still terminating). "
            "If this fails, increase terminationGracePeriodSeconds to widen the race window."
        )
    finally:
        subprocess.run(
            ["kubectl", "delete", "namespace", namespace, "--ignore-not-found"],
            capture_output=True,
            timeout=60,
        )


@pytest.mark.timeout(180)
def test_rollout_status_waits_for_completion(k8s_cluster):
    """``kubectl rollout status`` blocks until only the new pod serves traffic."""
    namespace = f"race-{uuid.uuid4().hex[:8]}"
    subprocess.run(["kubectl", "create", "namespace", namespace], check=True, capture_output=True, timeout=30)

    try:
        _apply(DEPLOY, namespace)
        _wait_available_replicas(namespace)

        _kubectl("rollout", "restart", "deployment/race-server", namespace=namespace)
        _wait_available_replicas(namespace)
        subprocess.run(
            ["kubectl", "-n", namespace, "rollout", "status", "deployment/race-server", "--timeout=120s"],
            check=True,
            capture_output=True,
            timeout=130,
        )

        # Old pod may still be in Terminating state (graceful shutdown), but K8s
        # removes it from Service endpoints so port-forwards won't route to it.
        assert _running_pod_count(namespace) == 1
    finally:
        subprocess.run(
            ["kubectl", "delete", "namespace", namespace, "--ignore-not-found"],
            capture_output=True,
            timeout=60,
        )
