# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU pod resource allocation coverage using fake-k8s and Kind.

This module keeps one deterministic in-memory regression test and one optional
Kind-backed integration test. The Kind test is skipped when the local machine
does not have the required tooling.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock

import pytest

from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint as FrayEntrypoint
from fray.types import GpuConfig, JobRequest, ResourceConfig
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.bundle import BundleStore
from iris.cluster.controller.controller import Controller, ControllerConfig
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.stores import ControllerStore
from iris.cluster.controller.transitions import ControllerTransitions
from iris.log_server.server import LogServiceImpl
from iris.cluster.providers.k8s.fake import FakeNodeResources, InMemoryK8sService
from iris.cluster.providers.k8s.service import CloudK8sService
from iris.cluster.providers.k8s.tasks import K8sTaskProvider, _LABEL_MANAGED, _LABEL_RUNTIME, _RUNTIME_LABEL_VALUE
from iris.cluster.providers.k8s.types import K8sResource
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName, ResourceSpec, TaskAttempt
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from rigging.timing import Duration

# ---------------------------------------------------------------------------
# Inlined helpers from lib/iris/tests/cluster/ to avoid cross-package imports
# (the iris test helpers aren't importable from top-level tests/).
# ---------------------------------------------------------------------------


class _HarnessController:
    """Minimal controller mock satisfying the ControllerProtocol surface."""

    def __init__(self) -> None:
        self.wake = Mock()
        self.kill_tasks_on_workers = Mock()
        self.create_scheduling_context = Mock(return_value=Mock())
        self.get_job_scheduling_diagnostics = Mock(return_value=None)
        self.autoscaler = None
        self.provider: object = Mock()
        self.has_direct_provider = False


@dataclass
class ServiceTestHarness:
    """Controller service backed by either GCP or K8s, without booting a cluster."""

    service: ControllerServiceImpl
    state: ControllerTransitions
    db: ControllerDB
    provider_type: str

    k8s: InMemoryK8sService | None = None
    k8s_provider: K8sTaskProvider | None = None

    def sync_k8s(self) -> None:
        assert self.k8s_provider is not None, "sync_k8s requires K8s harness"
        with self.state._store.transaction() as cur:
            batch = self.state.drain_for_direct_provider(cur)
        result = self.k8s_provider.sync(batch)
        with self.state._store.transaction() as cur:
            self.state.apply_direct_provider_updates(cur, result.updates)


def _make_test_entrypoint() -> job_pb2.RuntimeEntrypoint:
    entrypoint = job_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    return entrypoint


pytestmark = pytest.mark.e2e

KIND_CLUSTER = "iris-gpu-test"
KIND_CONTEXT = f"kind-{KIND_CLUSTER}"
NAMESPACE = "iris-canary"
CONTROLLER_PORT = 10555
POLL_TIMEOUT = 30.0

HAS_KIND = shutil.which("kind") is not None
HAS_KUBECTL = shutil.which("kubectl") is not None
HAS_DOCKER = (
    shutil.which("docker") is not None
    and subprocess.run(
        ["docker", "info"],
        capture_output=True,
    ).returncode
    == 0
)
# Kind cluster creation fails on many CI runners (cgroup/networking issues)
# even when the binaries exist. Require explicit opt-in via env var.
KIND_ENABLED = os.environ.get("IRIS_KIND_TESTS", "") == "1"

skip_no_kind = pytest.mark.skipif(
    not (KIND_ENABLED and HAS_KIND and HAS_KUBECTL and HAS_DOCKER),
    reason="Set IRIS_KIND_TESTS=1 with kind, kubectl, and a running Docker daemon",
)


def _gpu_resources() -> job_pb2.ResourceSpecProto:
    resources = job_pb2.ResourceSpecProto(
        cpu_millicores=32_000,
        memory_bytes=256 * 1024**3,
        disk_bytes=256 * 1024**3,
    )
    resources.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="H100", count=8))
    return resources


def _cpu_resources() -> job_pb2.ResourceSpecProto:
    return job_pb2.ResourceSpecProto(
        cpu_millicores=1000,
        memory_bytes=16 * 1024**3,
        disk_bytes=16 * 1024**3,
    )


def _get_iris_pods(k8s: InMemoryK8sService) -> list[dict]:
    return k8s.list_json(K8sResource.PODS, labels={_LABEL_MANAGED: "true", _LABEL_RUNTIME: _RUNTIME_LABEL_VALUE})


def _make_coreweave_harness(tmp_path: Path) -> ServiceTestHarness:
    db = ControllerDB(db_dir=tmp_path / "cw_db")
    log_service = LogServiceImpl(log_dir=tmp_path / "cw_logs")
    store = ControllerStore(db)
    state = ControllerTransitions(store=store)

    k8s = InMemoryK8sService()
    k8s.add_node_pool(
        "cpu-erapids",
        node_count=1,
        labels={"iris.pool": "cpu-erapids"},
        resources=FakeNodeResources(
            cpu_millicores=64_000,
            memory_bytes=256 * 1024**3,
            ephemeral_storage_bytes=256 * 1024**3,
        ),
    )
    k8s.add_node_pool(
        "h100-8x",
        node_count=1,
        labels={"iris.pool": "h100-8x"},
        taints=[
            {"key": "nvidia.com/gpu", "effect": "NoSchedule", "operator": "Exists"},
        ],
        resources=FakeNodeResources(
            cpu_millicores=128_000,
            memory_bytes=2048 * 1024**3,
            gpu_count=8,
            ephemeral_storage_bytes=512 * 1024**3,
        ),
    )

    k8s_provider = K8sTaskProvider(
        kubectl=k8s,
        namespace=NAMESPACE,
        default_image="ghcr.io/marin-community/iris-task:latest",
        host_network=True,
        controller_address="http://iris-controller-svc.iris-canary.svc.cluster.local:10000",
    )

    ctrl = _HarnessController()
    ctrl.has_direct_provider = True
    ctrl.provider = k8s_provider

    service = ControllerServiceImpl(
        state,
        store,
        controller=ctrl,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "cw_bundles")),
        log_service=log_service,
    )

    return ServiceTestHarness(
        service=service,
        state=state,
        db=db,
        provider_type="k8s",
        k8s=k8s,
        k8s_provider=k8s_provider,
    )


def _run(cmd: list[str], *, check: bool = True, stdin_data: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=check, input=stdin_data)


def _kubectl(*args: str, stdin_data: str | None = None) -> subprocess.CompletedProcess[str]:
    return _run(["kubectl", "--context", KIND_CONTEXT, *args], stdin_data=stdin_data)


def _kind_exists() -> bool:
    result = _run(["kind", "get", "clusters"], check=False)
    return KIND_CLUSTER in result.stdout.split()


def _reset_kind_cluster() -> None:
    if _kind_exists():
        _run(["kind", "delete", "cluster", "--name", KIND_CLUSTER])

    _run(["kind", "create", "cluster", "--name", KIND_CLUSTER])
    _kubectl("create", "namespace", NAMESPACE)

    for manifest in [
        {"apiVersion": "v1", "kind": "ServiceAccount", "metadata": {"name": "default", "namespace": NAMESPACE}},
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRole",
            "metadata": {"name": f"iris-controller-{NAMESPACE}"},
            "rules": [
                {
                    "apiGroups": [""],
                    "verbs": ["get", "list", "watch", "create", "update", "patch", "delete"],
                    "resources": ["pods", "pods/log", "pods/exec", "configmaps", "nodes", "events"],
                },
                {"apiGroups": ["metrics.k8s.io"], "resources": ["pods"], "verbs": ["get", "list"]},
            ],
        },
        {
            "apiVersion": "rbac.authorization.k8s.io/v1",
            "kind": "ClusterRoleBinding",
            "metadata": {"name": f"iris-controller-{NAMESPACE}"},
            "roleRef": {
                "apiGroup": "rbac.authorization.k8s.io",
                "kind": "ClusterRole",
                "name": f"iris-controller-{NAMESPACE}",
            },
            "subjects": [{"kind": "ServiceAccount", "name": "default", "namespace": NAMESPACE}],
        },
    ]:
        _kubectl("apply", "-f", "-", stdin_data=json.dumps(manifest))


@contextmanager
def _boot_controller(tmp_path: Path):
    kubeconfig = os.environ.get("KUBECONFIG", str(Path.home() / ".kube" / "config"))

    provider = K8sTaskProvider(
        kubectl=CloudK8sService(namespace=NAMESPACE, kubeconfig_path=kubeconfig),
        namespace=NAMESPACE,
        default_image="python:3.12-slim",
        host_network=False,
        controller_address=f"http://127.0.0.1:{CONTROLLER_PORT}",
    )

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    db = ControllerDB(db_dir=tmp_path / "db")
    controller = Controller(
        config=ControllerConfig(
            host="127.0.0.1",
            port=CONTROLLER_PORT,
            remote_state_dir=f"file://{state_dir}",
            heartbeat_interval=Duration.from_seconds(1),
            heartbeat_failure_threshold=10,
            local_state_dir=tmp_path / "local",
        ),
        provider=provider,
        db=db,
    )
    controller.start()
    url = f"http://127.0.0.1:{CONTROLLER_PORT}"

    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        controller.stop()
        db.close()
        raise AssertionError("controller did not become healthy")

    try:
        yield url
    finally:
        controller.stop()
        db.close()


def _wait_for_kind_pods(expected_pod_count: int) -> list[dict]:
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        result = _kubectl("-n", NAMESPACE, "get", "pods", "-o", "json", "-l", "iris.managed=true")
        pods = json.loads(result.stdout).get("items", [])
        if len(pods) >= expected_pod_count:
            return pods
        time.sleep(1)
    raise AssertionError(f"timed out waiting for {expected_pod_count} iris pods in Kind")


@pytest.fixture
def kind_cluster():
    _reset_kind_cluster()
    try:
        yield
    finally:
        _run(["kind", "delete", "cluster", "--name", KIND_CLUSTER], check=False)


def test_gpu_pod_attributes_with_in_memory_k8s(tmp_path: Path) -> None:
    harness = _make_coreweave_harness(tmp_path)
    try:
        launcher_id = JobName.root("runner", "canary-launcher")
        launcher_request = controller_pb2.Controller.LaunchJobRequest(
            name=launcher_id.to_wire(),
            entrypoint=_make_test_entrypoint(),
            resources=_cpu_resources(),
            environment=job_pb2.EnvironmentConfig(),
            replicas=1,
        )
        harness.service.launch_job(launcher_request, None)
        harness.sync_k8s()

        child_request = controller_pb2.Controller.LaunchJobRequest(
            name=launcher_id.child("grug-train").to_wire(),
            entrypoint=_make_test_entrypoint(),
            resources=_gpu_resources(),
            environment=job_pb2.EnvironmentConfig(),
            replicas=1,
        )
        harness.service.launch_job(child_request, None)
        harness.sync_k8s()

        pods = _get_iris_pods(harness.k8s)
        assert len(pods) == 2, f"expected launcher and child pods, got {len(pods)}"

        gpu_pods = [
            pod
            for pod in pods
            if "nvidia.com/gpu" in pod["spec"]["containers"][0].get("resources", {}).get("limits", {})
        ]
        cpu_pods = [
            pod
            for pod in pods
            if "nvidia.com/gpu" not in pod["spec"]["containers"][0].get("resources", {}).get("limits", {})
        ]

        assert len(gpu_pods) == 1
        assert len(cpu_pods) == 1

        gpu_limits = gpu_pods[0]["spec"]["containers"][0]["resources"]["limits"]
        gpu_toleration_keys = {t.get("key") for t in gpu_pods[0]["spec"].get("tolerations", [])}

        assert gpu_limits["nvidia.com/gpu"] == "8"
        assert gpu_limits["rdma/ib"] == "8"
        assert "nvidia.com/gpu" in gpu_toleration_keys
        assert "qos.coreweave.cloud/interruptable" not in gpu_toleration_keys
        assert "h100-8x" in gpu_pods[0]["spec"].get("nodeName", "")
        assert "cpu-erapids" in cpu_pods[0]["spec"].get("nodeName", "")
    finally:
        harness.db.close()


@skip_no_kind
def test_gpu_pod_attributes_with_kind(tmp_path: Path, kind_cluster) -> None:
    with _boot_controller(tmp_path) as url:
        iris_client = IrisClient.remote(url)
        fray_client = FrayIrisClient.from_iris_client(iris_client)

        iris_client.submit(
            entrypoint=Entrypoint.from_command("sleep", "3600"),
            name="launcher",
            resources=ResourceSpec(cpu=1, memory="16g", disk="16g"),
            environment=EnvironmentSpec(),
        )

        launcher_ctx = IrisContext(
            job_id=JobName.root("runner", "canary-launcher"),
            task_attempt=TaskAttempt(
                task_id=JobName.root("runner", "canary-launcher").task(0),
                attempt_id=0,
            ),
            client=iris_client,
        )

        with iris_ctx_scope(launcher_ctx):

            def _fake_train(config: str) -> None:
                del config

            fray_client.submit(
                JobRequest(
                    name="grug-train-canary",
                    entrypoint=FrayEntrypoint.from_callable(_fake_train, args=["fake-config"]),
                    resources=ResourceConfig(
                        cpu=32,
                        ram="256g",
                        disk="256g",
                        device=GpuConfig(variant="H100", count=8),
                    ),
                )
            )

        pods = _wait_for_kind_pods(expected_pod_count=2)
        gpu_pods = [
            pod
            for pod in pods
            if "nvidia.com/gpu" in pod["spec"]["containers"][0].get("resources", {}).get("limits", {})
        ]
        cpu_pods = [
            pod
            for pod in pods
            if "nvidia.com/gpu" not in pod["spec"]["containers"][0].get("resources", {}).get("limits", {})
        ]

        assert len(gpu_pods) == 1, f"expected one GPU pod, saw {len(gpu_pods)}"
        assert len(cpu_pods) == 1, f"expected one CPU pod, saw {len(cpu_pods)}"

        gpu_limits = gpu_pods[0]["spec"]["containers"][0]["resources"]["limits"]
        gpu_toleration_keys = {t.get("key") for t in gpu_pods[0]["spec"].get("tolerations", [])}
        cpu_toleration_keys = {t.get("key") for t in cpu_pods[0]["spec"].get("tolerations", [])}

        assert gpu_limits["nvidia.com/gpu"] == "8"
        assert "nvidia.com/gpu" in gpu_toleration_keys
        assert "qos.coreweave.cloud/interruptable" not in gpu_toleration_keys
        assert "nvidia.com/gpu" not in cpu_toleration_keys
