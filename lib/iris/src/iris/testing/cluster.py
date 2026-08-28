# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared cluster test harnesses and factories."""

from dataclasses import dataclass

from finelog.client import LogClient

from iris.cluster.backends.k8s.tasks import K8sTaskProvider, PodConfig
from iris.cluster.bundle import BundleStore
from iris.cluster.constraints import Constraint, ConstraintOp
from iris.cluster.controller.backend import BackendCapability
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.controller.transition_reader import DbTransitionReader
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.platforms.k8s.fake import FakeNodeResources, InMemoryK8sService
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from iris.testing.controller import MockController, make_test_entrypoint
from iris.testing.controller_state import ControllerTestState
from iris.testing.k8s import k8s_backend_descriptor

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
# Resource spec builders
# ---------------------------------------------------------------------------


def make_cpu_resource_spec() -> job_pb2.ResourceSpecProto:
    """Standard CPU resource spec for scheduling tests."""
    return job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=4 * 1024**3)


def make_gpu_resource_spec() -> job_pb2.ResourceSpecProto:
    """GPU resource spec with device type constraint."""
    spec = job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=4 * 1024**3)
    spec.device.gpu.CopyFrom(job_pb2.GpuDevice(variant="h100", count=1))
    return spec


# ---------------------------------------------------------------------------
# ServiceTestHarness — parameterized GCP / K8s controller service harness
# ---------------------------------------------------------------------------


@dataclass
class ServiceTestHarness:
    """Controller service backed by either GCP or K8s, without booting a cluster.

    ``state`` carries the projections the service reads through; the factories
    build it so ``db.caches`` is populated.
    """

    service: ControllerServiceImpl
    state: ControllerTestState
    db: ControllerDB

    def submit(
        self,
        name: str,
        *,
        user: str = "test-user",
        replicas: int = 1,
        max_retries_failure: int = 0,
        max_task_failures: int = 0,
        resources: job_pb2.ResourceSpecProto | None = None,
    ) -> JobName:
        """Submit a job via the RPC layer. Returns job_id."""
        job_id = JobName.root(user, name)
        request = controller_pb2.Controller.LaunchJobRequest(
            name=job_id.to_wire(),
            entrypoint=make_test_entrypoint(),
            resources=resources or job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            environment=job_pb2.EnvironmentConfig(),
            replicas=replicas,
            max_retries_failure=max_retries_failure,
            max_task_failures=max_task_failures,
        )
        self.service.launch_job(request, None)
        return job_id


# ---------------------------------------------------------------------------
# Harness factory functions
# ---------------------------------------------------------------------------


def _make_k8s_harness(tmp_path, log_address: str) -> ServiceTestHarness:
    db = ControllerDB(db_dir=tmp_path / "k8s_db")
    health = WorkerHealthTracker()
    state = ControllerTestState(db, health=health)

    k8s = InMemoryK8sService()
    k8s.add_node_pool(
        "default-cpu",
        node_count=4,
        resources=FakeNodeResources(cpu_millicores=8000, memory_bytes=32 * 1024**3),
    )

    k8s_provider = K8sTaskProvider(
        descriptor=k8s_backend_descriptor(),
        kubectl=k8s,
        pods=PodConfig(
            namespace="default",
            default_image="iris:test",
            controller_address="http://localhost:0",
            # Kueue is mandatory on the K8s backend, so every provider carries a LocalQueue.
            local_queue="iris-lq",
        ),
        cluster_scan_interval=0.0,
        transition_reader=DbTransitionReader(db),
    )

    ctrl = MockController()
    ctrl.capabilities = frozenset({BackendCapability.CLUSTER_VIEW})
    ctrl.provider = k8s_provider

    service = ControllerServiceImpl(
        controller=ctrl,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "k8s_bundles")),
        log_client=LogClient.connect(log_address),
        db=state._db,
        endpoint_service=EndpointServiceImpl(db=db),
    )

    return ServiceTestHarness(service=service, state=state, db=db)


def _make_gcp_harness(tmp_path, log_address: str) -> ServiceTestHarness:
    db = ControllerDB(db_dir=tmp_path / "gcp_db")
    health = WorkerHealthTracker()
    state = ControllerTestState(db, health=health)

    ctrl = MockController()
    ctrl.capabilities = frozenset({BackendCapability.WORKER_DAEMON, BackendCapability.IRIS_AUTOSCALER})
    # Share the harness tracker so the service registers into and reads liveness
    # through the same object this harness's ControllerTestState exposes.
    ctrl.provider.health = health

    service = ControllerServiceImpl(
        controller=ctrl,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "gcp_bundles")),
        log_client=LogClient.connect(log_address),
        db=state._db,
        endpoint_service=EndpointServiceImpl(db=db),
    )

    return ServiceTestHarness(service=service, state=state, db=db)


def make_service_test_harness(provider_type: str, tmp_path, log_address: str) -> ServiceTestHarness:
    """Build a controller service harness for the requested provider."""
    if provider_type == "k8s":
        return _make_k8s_harness(tmp_path, log_address)
    if provider_type == "gcp":
        return _make_gcp_harness(tmp_path, log_address)
    raise ValueError(f"Unknown provider type: {provider_type}")
