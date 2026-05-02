# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for cluster-level tests.

Provides constraint builders, resource spec fixtures, and the
ServiceTestHarness used across scheduling, controller, and integration tests.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from finelog.server import LogServiceImpl
from iris.cluster.bundle import BundleStore
from iris.cluster.constraints import Constraint, ConstraintOp, WellKnownAttribute
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import (
    TASK_DETAIL_PROJECTION,
    WORKER_DETAIL_PROJECTION,
)
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.stores import ControllerStore
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    HeartbeatApplyRequest,
    TaskUpdate,
)
from iris.cluster.providers.k8s.fake import FakeNodeResources, InMemoryK8sService
from iris.cluster.providers.k8s.tasks import K8sTaskProvider
from iris.cluster.providers.k8s.types import K8sResource
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Timestamp

# ---------------------------------------------------------------------------
# Constraint builders
# ---------------------------------------------------------------------------


def eq_constraint(key: str, value: str) -> Constraint:
    """Build an EQ constraint for the given key and string value."""
    return Constraint.create(key=key, op=ConstraintOp.EQ, value=value)


def in_constraint(key: str, values: list[str]) -> Constraint:
    """Build an IN constraint for the given key and string values."""
    return Constraint.create(key=key, op=ConstraintOp.IN, values=values)


# ---------------------------------------------------------------------------
# Resource spec fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cpu_resource_spec() -> job_pb2.ResourceSpecProto:
    """Standard CPU resource spec for scheduling tests."""
    return job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=4 * 1024**3)


@pytest.fixture
def gpu_resource_spec() -> job_pb2.ResourceSpecProto:
    """GPU resource spec with device type constraint."""
    spec = job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=4 * 1024**3)
    spec.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="h100", count=1))
    return spec


# ---------------------------------------------------------------------------
# Worker attribute helpers
# ---------------------------------------------------------------------------


def make_worker_attrs(
    region: str = "us-central1",
    device_type: str = "cpu",
    device_variant: str = "",
    preemptible: str | None = None,
    zone: str | None = None,
    tpu_name: str | None = None,
    tpu_worker_id: int | None = None,
    **extras: str,
) -> dict[str, job_pb2.AttributeValue]:
    """Build a worker attributes dict for scheduling tests.

    Returns a dict suitable for setting on WorkerMetadata.attributes.
    """
    attrs: dict[str, job_pb2.AttributeValue] = {
        WellKnownAttribute.DEVICE_TYPE: job_pb2.AttributeValue(string_value=device_type),
    }
    if region:
        attrs[WellKnownAttribute.REGION] = job_pb2.AttributeValue(string_value=region)
    if device_variant:
        attrs[WellKnownAttribute.DEVICE_VARIANT] = job_pb2.AttributeValue(string_value=device_variant)
    if preemptible is not None:
        attrs[WellKnownAttribute.PREEMPTIBLE] = job_pb2.AttributeValue(string_value=preemptible)
    if zone is not None:
        attrs[WellKnownAttribute.ZONE] = job_pb2.AttributeValue(string_value=zone)
    if tpu_name is not None:
        attrs[WellKnownAttribute.TPU_NAME] = job_pb2.AttributeValue(string_value=tpu_name)
    if tpu_worker_id is not None:
        attrs[WellKnownAttribute.TPU_WORKER_ID] = job_pb2.AttributeValue(int_value=tpu_worker_id)
    for key, val in extras.items():
        attrs[key] = job_pb2.AttributeValue(string_value=val)
    return attrs


# ---------------------------------------------------------------------------
# ServiceTestHarness — parameterized GCP / K8s controller service harness
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
    """Controller service backed by either GCP or K8s, without booting a cluster.

    Provides a generic task-driving interface for tests that just care about
    outcomes (submit -> succeed/fail), plus provider-specific extensions for
    tests that need deeper control.
    """

    service: ControllerServiceImpl
    state: ControllerTransitions
    db: ControllerDB
    provider_type: str  # "gcp" or "k8s"

    # Provider-specific — one set will be None depending on provider_type.
    k8s: InMemoryK8sService | None = None
    k8s_provider: K8sTaskProvider | None = None

    # ── Generic task interface ──────────────────────────────────

    def submit(
        self,
        name: str,
        *,
        user: str = "test-user",
        replicas: int = 1,
        max_retries_failure: int = 0,
        resources: job_pb2.ResourceSpecProto | None = None,
    ) -> JobName:
        """Submit a job via the RPC layer. Returns job_id."""
        from tests.cluster.controller.conftest import make_test_entrypoint

        job_id = JobName.root(user, name)
        request = controller_pb2.Controller.LaunchJobRequest(
            name=job_id.to_wire(),
            entrypoint=make_test_entrypoint(),
            resources=resources or job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            environment=job_pb2.EnvironmentConfig(),
            replicas=replicas,
            max_retries_failure=max_retries_failure,
        )
        self.service.launch_job(request, None)
        return job_id

    def drive_task_state(self, task_id: JobName, new_state: int) -> None:
        """Advance a task to the given state, abstracting over provider.

        For GCP: simulates a worker heartbeat reporting the new state.
        For K8s: transitions the pod and runs a sync cycle.
        """
        if self.provider_type == "k8s":
            self._drive_k8s(task_id, new_state)
        else:
            self._drive_gcp(task_id, new_state)

    def drive_job_to_completion(
        self,
        job_id: JobName,
        state: int = job_pb2.TASK_STATE_SUCCEEDED,
    ) -> None:
        """Drive all tasks in a job to the given terminal state."""
        for task in self._query_tasks(job_id):
            self.drive_task_state(task.task_id, state)

    def get_job_status(self, job_id: JobName) -> job_pb2.JobStatus:
        """Query job status via the RPC layer."""
        req = controller_pb2.Controller.GetJobStatusRequest(job_id=job_id.to_wire())
        return self.service.get_job_status(req, None).job

    # ── K8s-specific ────────────────────────────────────────────

    def add_k8s_node_pool(
        self,
        name: str,
        *,
        node_count: int = 1,
        labels: dict[str, str] | None = None,
        taints: list[dict[str, str]] | None = None,
        resources: FakeNodeResources | None = None,
    ) -> None:
        """Add a K8s node pool (K8s harness only)."""
        assert self.k8s is not None, "add_k8s_node_pool requires K8s harness"
        self.k8s.add_node_pool(
            name,
            node_count=node_count,
            labels=labels or {},
            taints=taints or [],
            resources=resources or FakeNodeResources(),
        )

    def sync_k8s(self) -> None:
        """Run one K8s direct provider sync cycle."""
        assert self.k8s_provider is not None, "sync_k8s requires K8s harness"
        with self.state._store.transaction() as cur:
            batch = self.state.drain_for_direct_provider(cur)
        result = self.k8s_provider.sync(batch)
        with self.state._store.transaction() as cur:
            self.state.apply_direct_provider_updates(cur, result.updates)

    # ── GCP-specific ────────────────────────────────────────────

    def register_gcp_worker(
        self,
        worker_id: str,
        *,
        device_type: str = "cpu",
        preemptible: bool = False,
        region: str = "us-central1",
    ) -> WorkerId:
        """Register a fake GCP worker with scheduling attributes."""
        assert self.provider_type == "gcp", "register_gcp_worker requires GCP harness"
        wid = WorkerId(worker_id)
        metadata = job_pb2.WorkerMetadata(
            hostname=worker_id,
            ip_address="127.0.0.1",
            cpu_count=8,
            memory_bytes=16 * 1024**3,
            disk_bytes=100 * 1024**3,
        )
        metadata.attributes["device-type"].string_value = device_type
        metadata.attributes["preemptible"].string_value = str(preemptible).lower()
        metadata.attributes["region"].string_value = region
        with self.state._store.transaction() as cur:
            self.state.register_or_refresh_worker(cur, wid, f"{worker_id}:8080", metadata, Timestamp.now())
        return wid

    # ── Private drivers ─────────────────────────────────────────

    def _query_tasks(self, job_id: JobName):
        with self.db.snapshot() as q:
            return TASK_DETAIL_PROJECTION.decode(q.fetchall("SELECT * FROM tasks WHERE job_id = ?", (job_id.to_wire(),)))

    def _query_task(self, task_id: JobName):
        with self.db.snapshot() as q:
            return TASK_DETAIL_PROJECTION.decode_one(
                q.fetchall("SELECT * FROM tasks WHERE task_id = ? LIMIT 1", (task_id.to_wire(),)),
            )

    def _drive_k8s(self, task_id: JobName, new_state: int) -> None:
        """K8s: drain to create pod, transition pod, sync to apply."""
        assert self.k8s is not None and self.k8s_provider is not None

        # First sync: promotes PENDING -> ASSIGNED, creates pod
        self.sync_k8s()

        pod_name = self._find_pod_for_task(task_id)
        if pod_name is None:
            raise ValueError(f"No pod found for task {task_id}")

        if new_state == job_pb2.TASK_STATE_RUNNING:
            self.k8s.transition_pod(pod_name, "Running")
        elif new_state == job_pb2.TASK_STATE_SUCCEEDED:
            self.k8s.transition_pod(pod_name, "Running")
            self.sync_k8s()
            self.k8s.transition_pod(pod_name, "Succeeded")
        elif new_state == job_pb2.TASK_STATE_FAILED:
            self.k8s.transition_pod(pod_name, "Running")
            self.sync_k8s()
            self.k8s.transition_pod(pod_name, "Failed", exit_code=1, reason="Error")
        elif new_state == job_pb2.TASK_STATE_WORKER_FAILED:
            self.k8s.transition_pod(pod_name, "Running")
            self.sync_k8s()
            self.k8s.transition_pod(pod_name, "Failed", exit_code=137, reason="OOMKilled")

        self.sync_k8s()

    def _find_pod_for_task(self, task_id: JobName) -> str | None:
        """Find the pod for a task's latest attempt by scanning K8s pod labels.

        When a task retries, multiple pods share the same task_hash. We pick
        the pod with the highest attempt_id so callers drive the current attempt.
        """
        assert self.k8s is not None
        expected_hash = hashlib.sha256(task_id.to_wire().encode()).hexdigest()[:16]
        best_name: str | None = None
        best_attempt = -1
        for pod in self.k8s.list_json(K8sResource.PODS):
            labels = pod.get("metadata", {}).get("labels", {})
            if labels.get("iris.task_hash") == expected_hash:
                attempt = int(labels.get("iris.attempt_id", "0"))
                if attempt > best_attempt:
                    best_attempt = attempt
                    best_name = pod["metadata"]["name"]
        return best_name

    def _current_attempt_info(self, task_id: JobName) -> tuple[WorkerId | None, int]:
        """Read current worker_id and attempt_id from the task_attempts table.

        SELECT * FROM tasks doesn't join with task_attempts, so
        TaskDetailRow.current_worker_id may be None when read via _query_task. We read
        the attempt row directly instead.
        """
        with self.db.snapshot() as q:
            rows = q.raw(
                "SELECT worker_id, attempt_id FROM task_attempts " "WHERE task_id = ? ORDER BY attempt_id DESC LIMIT 1",
                (task_id.to_wire(),),
            )
        if not rows:
            return None, 0
        return WorkerId(rows[0].worker_id), int(rows[0].attempt_id)

    def _drive_gcp(self, task_id: JobName, new_state: int) -> None:
        """GCP: find the assigned worker and simulate a heartbeat update."""
        task = self._query_task(task_id)
        if task is None:
            raise ValueError(f"Task {task_id} not found")

        # If still PENDING, assign to an available worker.
        if task.state == job_pb2.TASK_STATE_PENDING:
            with self.db.snapshot() as q:
                workers = WORKER_DETAIL_PROJECTION.decode(q.fetchall("SELECT * FROM workers"))
            if not workers:
                raise ValueError("No GCP workers registered -- call register_gcp_worker first")
            with self.state._store.transaction() as cur:
                self.state.queue_assignments(
                    cur, [Assignment(task_id=task_id, worker_id=WorkerId(workers[0].worker_id))]
                )

        worker_id, attempt_id = self._current_attempt_info(task_id)
        if worker_id is None:
            raise ValueError(f"Task {task_id} has no assigned worker")

        # For terminal states, go through RUNNING first.
        if (
            new_state
            in (
                job_pb2.TASK_STATE_SUCCEEDED,
                job_pb2.TASK_STATE_FAILED,
                job_pb2.TASK_STATE_WORKER_FAILED,
            )
            and task.state != job_pb2.TASK_STATE_RUNNING
        ):
            with self.state._store.transaction() as cur:
                self.state.apply_task_updates(
                    cur,
                    HeartbeatApplyRequest(
                        worker_id=worker_id,
                        worker_resource_snapshot=None,
                        updates=[
                            TaskUpdate(
                                task_id=task_id,
                                attempt_id=attempt_id,
                                new_state=job_pb2.TASK_STATE_RUNNING,
                            )
                        ],
                    ),
                )

        with self.state._store.transaction() as cur:
            self.state.apply_task_updates(
                cur,
                HeartbeatApplyRequest(
                    worker_id=worker_id,
                    worker_resource_snapshot=None,
                    updates=[
                        TaskUpdate(
                            task_id=task_id,
                            attempt_id=attempt_id,
                            new_state=new_state,
                        )
                    ],
                ),
            )


# ---------------------------------------------------------------------------
# Harness factory functions
# ---------------------------------------------------------------------------


def _make_k8s_harness(tmp_path) -> ServiceTestHarness:
    db = ControllerDB(db_dir=tmp_path / "k8s_db")
    store = ControllerStore(db)
    state = ControllerTransitions(store=store)

    k8s = InMemoryK8sService()
    k8s.add_node_pool(
        "default-cpu",
        node_count=4,
        resources=FakeNodeResources(cpu_millicores=8000, memory_bytes=32 * 1024**3),
    )

    k8s_provider = K8sTaskProvider(
        kubectl=k8s,
        namespace="default",
        default_image="iris:test",
        controller_address="http://localhost:0",
    )

    ctrl = _HarnessController()
    ctrl.has_direct_provider = True
    ctrl.provider = k8s_provider

    service = ControllerServiceImpl(
        state,
        store,
        controller=ctrl,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "k8s_bundles")),
        log_service=LogServiceImpl(),
    )

    return ServiceTestHarness(
        service=service,
        state=state,
        db=db,
        provider_type="k8s",
        k8s=k8s,
        k8s_provider=k8s_provider,
    )


def _make_gcp_harness(tmp_path) -> ServiceTestHarness:
    db = ControllerDB(db_dir=tmp_path / "gcp_db")
    store = ControllerStore(db)
    state = ControllerTransitions(store=store)

    ctrl = _HarnessController()
    ctrl.has_direct_provider = False

    service = ControllerServiceImpl(
        state,
        store,
        controller=ctrl,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "gcp_bundles")),
        log_service=LogServiceImpl(),
    )

    return ServiceTestHarness(
        service=service,
        state=state,
        db=db,
        provider_type="gcp",
    )


@pytest.fixture(params=["gcp", "k8s"])
def harness(request, tmp_path) -> ServiceTestHarness:
    """ControllerServiceImpl backed by either GCP or K8s provider.

    Tests using this fixture run twice -- once with each provider -- to ensure
    both code paths are exercised.
    """
    if request.param == "k8s":
        h = _make_k8s_harness(tmp_path)
    else:
        h = _make_gcp_harness(tmp_path)
    yield h
    h.db.close()
