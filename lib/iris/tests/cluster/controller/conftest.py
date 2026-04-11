# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for controller unit tests."""

import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import replace as _replace
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from iris.cluster.bundle import BundleStore
from iris.cluster.constraints import (
    Constraint,
    ConstraintOp,
    DeviceType,
    PlacementRequirements,
    WellKnownAttribute,
    constraints_from_resources,
    device_variant_constraint,
    merge_constraints,
    preemptible_constraint,
    region_constraint,
    zone_constraint,
)
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.models import DemandEntry
from iris.cluster.controller.db import (
    ACTIVE_TASK_STATES,
    TERMINAL_TASK_STATES,
    ControllerDB,
    _decode_attribute_rows,
    job_is_finished,
    task_row_can_be_scheduled,
    task_row_is_finished,
)
from iris.cluster.controller.schema import (
    ATTEMPT_PROJECTION,
    JOB_CONFIG_JOIN,
    JOB_DETAIL_PROJECTION,
    JOB_SCHEDULING_PROJECTION,
    TASK_DETAIL_PROJECTION,
    WORKER_ROW_PROJECTION,
    JobDetailRow,
    TaskDetailRow,
    WorkerRow,
    tasks_with_attempts,
)
from iris.cluster.controller.provider import ProviderUnsupportedError
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.cluster.controller.service import ControllerServiceImpl
from iris.log_server.server import LogServiceImpl
from iris.cluster.controller.transitions import (
    Assignment,
    ControllerTransitions,
    DispatchBatch,
    HEARTBEAT_FAILURE_THRESHOLD,
    HeartbeatApplyRequest,
    TaskUpdate,
)
from iris.cluster.providers.gcp.fake import InMemoryGcpService
from iris.cluster.providers.gcp.workers import GcpWorkerProvider
from iris.cluster.providers.types import CloudSliceState
from iris.cluster.service_mode import ServiceMode
from iris.cluster.types import JobName, WorkerId
from iris.rpc import config_pb2
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from iris.time_proto import duration_to_proto
from rigging.timing import Duration, Timestamp
from tests.cluster.providers.conftest import make_mock_platform

check_task_can_be_scheduled = task_row_can_be_scheduled
check_task_is_finished = task_row_is_finished


def check_job_is_finished(j: JobDetailRow) -> bool:
    """Whether a job row is in a terminal state."""
    return job_is_finished(j.state)


class FakeProvider:
    """Minimal TaskProvider for tests that only exercise transitions, not RPCs."""

    def sync(
        self,
        batches: list[DispatchBatch],
    ) -> list[tuple[DispatchBatch, HeartbeatApplyRequest | None, str | None]]:
        return [(b, None, "no stub") for b in batches]

    def get_process_status(
        self,
        worker_id: WorkerId,
        address: str | None,
        request: job_pb2.GetProcessStatusRequest,
    ) -> job_pb2.GetProcessStatusResponse:
        raise ProviderUnsupportedError("fake")

    def on_worker_failed(self, worker_id: WorkerId, address: str | None) -> None:
        pass

    def profile_task(
        self,
        address: str,
        request: job_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> job_pb2.ProfileTaskResponse:
        raise ProviderUnsupportedError("fake")

    def close(self) -> None:
        pass


@pytest.fixture
def state():
    """Create a fresh ControllerTransitions with temp DB and log store."""
    with make_controller_state() as s:
        yield s


class MockController:
    """Mock that implements the ControllerProtocol surface used by ControllerServiceImpl."""

    def __init__(self):
        self.wake = Mock()
        self.kill_tasks_on_workers = Mock()
        self.create_scheduling_context = Mock(return_value=Mock())
        self.get_job_scheduling_diagnostics = Mock(return_value=None)
        self.autoscaler = None
        self.provider = Mock()
        self.has_direct_provider = False


@pytest.fixture
def mock_controller() -> MockController:
    return MockController()


@pytest.fixture
def log_service(state, tmp_path) -> LogServiceImpl:
    """LogServiceImpl with its own internal log store."""
    svc = LogServiceImpl(log_dir=tmp_path / "log_service_logs")
    yield svc
    svc.close()


@pytest.fixture
def controller_service(state, log_service, mock_controller, tmp_path) -> ControllerServiceImpl:
    """ControllerServiceImpl with fresh DB, log service, and mock controller."""
    return ControllerServiceImpl(
        state,
        state._db,
        controller=mock_controller,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_service=log_service,
    )


# =============================================================================
# State factory helpers
# =============================================================================


@contextmanager
def make_controller_state(**kwargs):
    """Yield a ControllerTransitions with a fresh temp DB, cleaning up on exit."""
    tmp = Path(tempfile.mkdtemp(prefix="iris_test_"))
    try:
        db = ControllerDB(db_dir=tmp)
        yield ControllerTransitions(db=db, **kwargs)
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def make_test_entrypoint() -> job_pb2.RuntimeEntrypoint:
    entrypoint = job_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    return entrypoint


def make_direct_job_request(
    name: str = "test-job",
    replicas: int = 1,
    task_image: str = "",
) -> controller_pb2.Controller.LaunchJobRequest:
    job_name = JobName.root("test-user", name)
    return controller_pb2.Controller.LaunchJobRequest(
        name=job_name.to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=replicas,
        task_image=task_image,
    )


def submit_direct_job(
    state: ControllerTransitions,
    name: str,
    replicas: int = 1,
    task_image: str = "",
) -> list[JobName]:
    jid = JobName.root("test-user", name)
    req = make_direct_job_request(name, replicas, task_image=task_image)
    state.submit_job(jid, req, Timestamp.now())
    with state._db.snapshot() as q:
        tasks = TASK_DETAIL_PROJECTION.decode(q.fetchall("SELECT * FROM tasks WHERE job_id = ?", (jid.to_wire(),)))
    return [t.task_id for t in tasks]


# =============================================================================
# DB query helpers (shared across test_scheduler, test_transitions, etc.)
# =============================================================================


def query_task(state: ControllerTransitions, task_id: JobName):
    with state._db.snapshot() as q:
        return TASK_DETAIL_PROJECTION.decode_one(
            q.fetchall("SELECT * FROM tasks WHERE task_id = ? LIMIT 1", (task_id.to_wire(),)),
        )


def query_attempt(state: ControllerTransitions, task_id: JobName, attempt_id: int):
    with state._db.snapshot() as q:
        rows = ATTEMPT_PROJECTION.decode(
            q.fetchall(
                "SELECT * FROM task_attempts WHERE task_id = ? AND attempt_id = ?",
                (task_id.to_wire(), attempt_id),
            ),
        )
    return rows[0] if rows else None


def query_job(state: ControllerTransitions, job_id: JobName) -> JobDetailRow | None:
    with state._db.snapshot() as q:
        return JOB_DETAIL_PROJECTION.decode_one(
            q.fetchall(
                f"SELECT j.*, jc.* FROM jobs j {JOB_CONFIG_JOIN} WHERE j.job_id = ? LIMIT 1",
                (job_id.to_wire(),),
            )
        )


def query_job_row(state: ControllerTransitions, job_id: JobName):
    """Query a job as a JobSchedulingRow (scheduling projection with resources/constraints)."""
    with state._db.snapshot() as q:
        return JOB_SCHEDULING_PROJECTION.decode_one(
            q.fetchall(
                f"SELECT j.*, jc.* FROM jobs j {JOB_CONFIG_JOIN} WHERE j.job_id = ? LIMIT 1",
                (job_id.to_wire(),),
            )
        )


def query_worker(state: ControllerTransitions, worker_id: WorkerId) -> WorkerRow | None:
    with state._db.snapshot() as q:
        return WORKER_ROW_PROJECTION.decode_one(
            q.fetchall("SELECT * FROM workers WHERE worker_id = ? LIMIT 1", (str(worker_id),)),
        )


def query_tasks_for_job(state: ControllerTransitions, job_id: JobName) -> list[TaskDetailRow]:
    with state._db.snapshot() as q:
        return TASK_DETAIL_PROJECTION.decode(q.fetchall("SELECT * FROM tasks WHERE job_id = ?", (job_id.to_wire(),)))


def schedulable_tasks(state: ControllerTransitions):
    """Return non-terminal tasks eligible for scheduling, in priority order."""
    terminal_placeholders = ",".join("?" for _ in TERMINAL_TASK_STATES)
    with state._db.snapshot() as q:
        tasks = TASK_DETAIL_PROJECTION.decode(
            q.fetchall(
                f"SELECT * FROM tasks WHERE state IS NOT NULL AND state NOT IN ({terminal_placeholders})"
                " ORDER BY priority_neg_depth ASC, priority_root_submitted_ms ASC, submitted_at_ms ASC, task_id ASC",
                tuple(TERMINAL_TASK_STATES),
            ),
        )
    return [t for t in tasks if check_task_can_be_scheduled(t)]


def building_counts(state: ControllerTransitions) -> dict[WorkerId, int]:
    """Count tasks in BUILDING/ASSIGNED state per worker, excluding reservation holders."""
    with state._db.snapshot() as snapshot:
        rows = snapshot.raw(
            "SELECT a.worker_id, COUNT(*) as c FROM tasks t "
            "JOIN task_attempts a ON t.task_id = a.task_id AND t.current_attempt_id = a.attempt_id "
            "JOIN jobs j ON t.job_id = j.job_id "
            "WHERE t.state IN (?, ?) AND j.is_reservation_holder = 0 "
            "GROUP BY a.worker_id ORDER BY a.worker_id ASC",
            (
                job_pb2.TASK_STATE_BUILDING,
                job_pb2.TASK_STATE_ASSIGNED,
            ),
            decoders={"worker_id": WorkerId, "c": int},
        )
    return {row.worker_id: row.c for row in rows}


def register_worker(
    state: ControllerTransitions,
    worker_id: str,
    address: str,
    metadata: job_pb2.WorkerMetadata,
    healthy: bool = True,
    slice_id: str = "",
    scale_group: str = "",
) -> WorkerId:
    wid = WorkerId(worker_id)
    state.register_or_refresh_worker(
        worker_id=wid,
        address=address,
        metadata=metadata,
        ts=Timestamp.now(),
        slice_id=slice_id,
        scale_group=scale_group,
    )
    if not healthy:
        state._db.execute("UPDATE workers SET healthy = 0 WHERE worker_id = ?", (str(wid),))
    return wid


def inject_device_constraints(request: controller_pb2.Controller.LaunchJobRequest) -> None:
    """Auto-inject device constraints from the resource spec, mirroring service.py.

    In production, the service layer merges auto-generated device constraints
    into the request before storing the job. Tests bypass the service layer,
    so we replicate that logic here.
    """
    auto = constraints_from_resources(request.resources)
    if not auto:
        return
    user = [Constraint.from_proto(c) for c in request.constraints]
    merged = merge_constraints(auto, user)
    del request.constraints[:]
    for c in merged:
        request.constraints.append(c.to_proto())


def submit_job(
    state: ControllerTransitions,
    job_id: str,
    request: controller_pb2.Controller.LaunchJobRequest,
    timestamp_ms: int | None = None,
) -> list:
    """Submit a job and return created task rows.

    Auto-injects resource-derived device constraints to mirror service-layer
    behavior, then submits and returns the created tasks.
    """
    inject_device_constraints(request)
    jid = JobName.from_string(job_id) if job_id.startswith("/") else JobName.root("test-user", job_id)
    request.name = jid.to_wire()
    state.submit_job(
        jid,
        request,
        Timestamp.from_ms(timestamp_ms) if timestamp_ms is not None else Timestamp.now(),
    )
    return query_tasks_for_job(state, jid)


# =============================================================================
# Shared test helpers (deduplicated from test_transitions, test_scheduler,
# test_service, test_dashboard, test_reservation)
# =============================================================================


def query_tasks_with_attempts(state: ControllerTransitions, job_id: JobName) -> list[TaskDetailRow]:
    with state._db.snapshot() as q:
        tasks = TASK_DETAIL_PROJECTION.decode(
            q.fetchall("SELECT * FROM tasks WHERE job_id = ? ORDER BY task_index ASC", (job_id.to_wire(),)),
        )
        if not tasks:
            return []
        task_wires = [t.task_id.to_wire() for t in tasks]
        placeholders = ",".join("?" for _ in task_wires)
        attempts = ATTEMPT_PROJECTION.decode(
            q.fetchall(
                f"SELECT * FROM task_attempts WHERE task_id IN ({placeholders})" " ORDER BY task_id ASC, attempt_id ASC",
                tuple(task_wires),
            ),
        )
    return tasks_with_attempts(tasks, attempts)


def query_task_with_attempts(state: ControllerTransitions, task_id: JobName) -> TaskDetailRow | None:
    wire = task_id.to_wire()
    with state._db.snapshot() as q:
        tasks = TASK_DETAIL_PROJECTION.decode(q.fetchall("SELECT * FROM tasks WHERE task_id = ?", (wire,)))
        attempts = ATTEMPT_PROJECTION.decode(
            q.fetchall("SELECT * FROM task_attempts WHERE task_id = ? ORDER BY attempt_id ASC", (wire,)),
        )
    hydrated = tasks_with_attempts(tasks, attempts)
    return hydrated[0] if hydrated else None


def make_job_request(
    name: str = "test-job",
    cpu: int = 1,
    memory_bytes: int = 1024**3,
    replicas: int = 1,
    max_retries_failure: int = 0,
    max_retries_preemption: int = 0,
    scheduling_timeout_seconds: int = 0,
    priority_band: int = 0,
    task_image: str = "",
) -> controller_pb2.Controller.LaunchJobRequest:
    job_name = JobName.from_string(name) if name.startswith("/") else JobName.root("test-user", name)
    request = controller_pb2.Controller.LaunchJobRequest(
        name=job_name.to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=cpu * 1000, memory_bytes=memory_bytes),
        environment=job_pb2.EnvironmentConfig(),
        max_retries_failure=max_retries_failure,
        max_retries_preemption=max_retries_preemption,
        replicas=replicas,
        priority_band=priority_band,
        task_image=task_image,
    )
    if scheduling_timeout_seconds > 0:
        request.scheduling_timeout.CopyFrom(duration_to_proto(Duration.from_seconds(scheduling_timeout_seconds)))
    return request


def make_worker_metadata(
    cpu: int = 10,
    memory_bytes: int = 10 * 1024**3,
    disk_bytes: int = 10 * 1024**3,
    gpu_count: int = 0,
    gpu_name: str = "",
    tpu_name: str = "",
) -> job_pb2.WorkerMetadata:
    """Build WorkerMetadata with device config and well-known attributes.

    Populates device-type and device-variant attributes so constraint-based
    scheduling works the same way as production.
    """
    device = job_pb2.DeviceConfig()
    if tpu_name:
        device.tpu.CopyFrom(job_pb2.TpuDevice(variant=tpu_name))
    elif gpu_count > 0:
        device.gpu.CopyFrom(job_pb2.GpuDevice(variant=gpu_name or "auto", count=gpu_count))
    else:
        device.cpu.CopyFrom(job_pb2.CpuDevice(variant="cpu"))

    meta = job_pb2.WorkerMetadata(
        hostname="test-worker",
        ip_address="127.0.0.1",
        cpu_count=cpu,
        memory_bytes=memory_bytes,
        disk_bytes=disk_bytes,
        gpu_count=gpu_count,
        gpu_name=gpu_name,
        tpu_name=tpu_name,
        device=device,
    )

    if tpu_name:
        meta.attributes[WellKnownAttribute.DEVICE_TYPE].string_value = "tpu"
        meta.attributes[WellKnownAttribute.DEVICE_VARIANT].string_value = tpu_name.lower()
    elif gpu_count > 0:
        meta.attributes[WellKnownAttribute.DEVICE_TYPE].string_value = "gpu"
        if gpu_name:
            meta.attributes[WellKnownAttribute.DEVICE_VARIANT].string_value = gpu_name.lower()
    else:
        meta.attributes[WellKnownAttribute.DEVICE_TYPE].string_value = "cpu"

    return meta


def worker_running_tasks(state: ControllerTransitions, worker_id: WorkerId) -> frozenset[JobName]:
    with state._db.snapshot() as q:
        rows = q.raw(
            "SELECT t.task_id FROM tasks t "
            "JOIN task_attempts a ON t.task_id = a.task_id AND t.current_attempt_id = a.attempt_id "
            "WHERE a.worker_id = ? AND t.state IN (?, ?, ?)",
            (str(worker_id), *ACTIVE_TASK_STATES),
            decoders={"task_id": JobName.from_wire},
        )
    return frozenset(row.task_id for row in rows)


def hydrate_worker_attributes(state: ControllerTransitions, workers: list) -> list:
    if not workers:
        return workers
    worker_ids = [str(w.worker_id) for w in workers]
    placeholders = ",".join("?" for _ in worker_ids)
    with state._db.snapshot() as q:
        attrs = q.raw(
            f"SELECT worker_id, key, value_type, str_value, int_value, float_value"
            f" FROM worker_attributes WHERE worker_id IN ({placeholders})",
            tuple(worker_ids),
        )
    attrs_by_worker = _decode_attribute_rows(attrs)
    return [
        _replace(
            w,
            attributes=attrs_by_worker.get(w.worker_id, {}),
            available_cpu_millicores=w.total_cpu_millicores - w.committed_cpu_millicores,
            available_memory=w.total_memory_bytes - w.committed_mem,
            available_gpus=w.total_gpu_count - w.committed_gpu,
            available_tpus=w.total_tpu_count - w.committed_tpu,
        )
        for w in workers
    ]


def healthy_active_workers(state: ControllerTransitions) -> list[WorkerRow]:
    with state._db.snapshot() as q:
        workers = WORKER_ROW_PROJECTION.decode(q.fetchall("SELECT * FROM workers WHERE healthy = 1 AND active = 1"))
    return hydrate_worker_attributes(state, workers)


def dispatch_task(state: ControllerTransitions, task: TaskDetailRow, worker_id: WorkerId) -> None:
    state.queue_assignments([Assignment(task_id=task.task_id, worker_id=worker_id)])
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=worker_id,
            worker_resource_snapshot=None,
            updates=[
                TaskUpdate(
                    task_id=task.task_id,
                    attempt_id=query_task(state, task.task_id).current_attempt_id,
                    new_state=job_pb2.TASK_STATE_RUNNING,
                )
            ],
        )
    )


def transition_task(
    state: ControllerTransitions,
    task_id: JobName,
    new_state: int,
    *,
    error: str | None = None,
    exit_code: int | None = None,
) -> object:
    task = query_task_with_attempts(state, task_id)
    assert task is not None
    if new_state == job_pb2.TASK_STATE_KILLED:
        return state.cancel_job(task.job_id, reason=error or "killed")
    # Compute worker_id: prefer current attempt's worker, fall back to current_worker_id.
    current_attempt = task.attempts[-1] if task.attempts else None
    worker_id = current_attempt.worker_id if current_attempt is not None else task.current_worker_id
    if worker_id is None:
        state.set_task_state_for_test(
            task_id,
            new_state,
            error=error,
            exit_code=exit_code,
        )
        return state
    return state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=worker_id,
            worker_resource_snapshot=None,
            updates=[
                TaskUpdate(
                    task_id=task_id,
                    attempt_id=task.current_attempt_id,
                    new_state=new_state,
                    error=error,
                    exit_code=exit_code,
                )
            ],
        )
    )


def fail_worker(state: ControllerTransitions, worker_id: WorkerId, error: str) -> None:
    batch = state.drain_dispatch(worker_id)
    if batch is None:
        return
    for _ in range(HEARTBEAT_FAILURE_THRESHOLD):
        state.record_heartbeat_failure(worker_id, error, batch)


# =============================================================================
# ControllerTestHarness
# =============================================================================


class ControllerTestHarness:
    """Wraps ControllerTransitions with ergonomic helpers for the common
    register-workers -> submit-jobs -> dispatch -> transition test pattern."""

    def __init__(self, state: ControllerTransitions):
        self.state = state

    def add_worker(
        self,
        worker_id: str = "w1",
        address: str | None = None,
        *,
        cpu: int = 10,
        memory_bytes: int = 10 * 1024**3,
        disk_bytes: int = 10 * 1024**3,
        gpu_count: int = 0,
        gpu_name: str = "",
        tpu_name: str = "",
        healthy: bool = True,
    ) -> WorkerId:
        meta = make_worker_metadata(
            cpu=cpu,
            memory_bytes=memory_bytes,
            disk_bytes=disk_bytes,
            gpu_count=gpu_count,
            gpu_name=gpu_name,
            tpu_name=tpu_name,
        )
        return register_worker(self.state, worker_id, address or f"{worker_id}:8080", meta, healthy=healthy)

    def submit(self, name: str = "test-job", *, cpu: int = 1, replicas: int = 1, **kwargs) -> list[TaskDetailRow]:
        req = make_job_request(name=name, cpu=cpu, replicas=replicas, **kwargs)
        return submit_job(self.state, name, req)

    def dispatch(self, task: TaskDetailRow, worker_id: WorkerId) -> None:
        dispatch_task(self.state, task, worker_id)

    def transition(self, task_id: JobName, new_state: int, **kwargs) -> None:
        transition_task(self.state, task_id, new_state, **kwargs)

    def query_task(self, task_id: JobName) -> TaskDetailRow:
        return query_task(self.state, task_id)

    def query_job(self, job_id: JobName) -> JobDetailRow:
        return query_job(self.state, job_id)


@pytest.fixture
def harness(state) -> ControllerTestHarness:
    return ControllerTestHarness(state)


# =============================================================================
# Autoscaler helpers (shared by test_autoscaler, test_demand_routing,
# test_autoscaler_integration)
# =============================================================================


DEFAULT_RESOURCES = config_pb2.ScaleGroupResources(
    cpu_millicores=128000,
    memory_bytes=128 * 1024**3,
    disk_bytes=100 * 1024**3,
    device_type=config_pb2.ACCELERATOR_TYPE_TPU,
    device_variant="v5p-8",
    device_count=8,
)


def ensure_scale_group_resources(config: config_pb2.ScaleGroupConfig) -> config_pb2.ScaleGroupConfig:
    if not config.HasField("resources"):
        config.resources.CopyFrom(DEFAULT_RESOURCES)
    if not config.HasField("num_vms"):
        config.num_vms = 1
    return config


def make_scale_group_config(**kwargs: object) -> config_pb2.ScaleGroupConfig:
    accelerator_type = kwargs.pop("accelerator_type", config_pb2.ACCELERATOR_TYPE_TPU)
    accelerator_variant = kwargs.pop("accelerator_variant", "v5p-8")
    runtime_version = kwargs.pop("runtime_version", None)
    zones = kwargs.pop("zones", None)
    capacity_type = kwargs.pop("capacity_type", None)
    config = ensure_scale_group_resources(config_pb2.ScaleGroupConfig(**kwargs))
    config.resources.device_type = accelerator_type
    if accelerator_variant:
        config.resources.device_variant = accelerator_variant
    if capacity_type is not None:
        config.slice_template.capacity_type = capacity_type
        config.resources.capacity_type = capacity_type

    # Derive slice template fields from resources, matching what
    # _derive_slice_config_from_resources() does in production config loading.
    # GcpWorkerProvider validates these fields on create_slice().
    template = config.slice_template
    template.accelerator_type = accelerator_type
    if accelerator_variant:
        template.accelerator_variant = accelerator_variant
    if accelerator_type == config_pb2.ACCELERATOR_TYPE_GPU and config.resources.device_count > 0:
        template.gpu_count = config.resources.device_count

    gcp = template.gcp
    if runtime_version:
        gcp.runtime_version = runtime_version
    elif accelerator_type == config_pb2.ACCELERATOR_TYPE_TPU and not gcp.runtime_version:
        gcp.runtime_version = "v2-alpha-tpuv5"
    if zones:
        gcp.zone = zones[0]
    elif not gcp.zone:
        gcp.zone = "us-central1-a"

    return config


def make_demand_entries(
    count: int,
    *,
    device_type: DeviceType = DeviceType.TPU,
    device_variant: str | None = "v5p-8",
    device_variants: frozenset[str] | None = None,
    capacity_type: int | None = None,
    required_regions: frozenset[str] | None = None,
    required_zones: frozenset[str] | None = None,
    task_prefix: str = "task",
) -> list[DemandEntry]:
    if count <= 0:
        return []
    resources = job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024)
    if device_type == DeviceType.TPU:
        resources.device.tpu.variant = device_variant or ""
    elif device_type == DeviceType.GPU:
        resources.device.gpu.variant = device_variant or ""
    elif device_type == DeviceType.CPU:
        resources.device.cpu.variant = ""
    effective_variants = device_variants
    if effective_variants is None and device_variant is not None:
        effective_variants = frozenset({device_variant})
    preemptible = (capacity_type == config_pb2.CAPACITY_TYPE_PREEMPTIBLE) if capacity_type is not None else None
    normalized = PlacementRequirements(
        device_type=device_type,
        device_variants=effective_variants,
        preemptible=preemptible,
        required_regions=required_regions,
        required_zones=required_zones,
    )

    constraint_list: list[Constraint] = []
    if device_type is not None:
        constraint_list.append(
            Constraint(key=WellKnownAttribute.DEVICE_TYPE, op=ConstraintOp.EQ, value=device_type.value)
        )
    if effective_variants:
        constraint_list.append(device_variant_constraint(sorted(effective_variants)))
    if capacity_type is not None:
        constraint_list.append(preemptible_constraint(capacity_type == config_pb2.CAPACITY_TYPE_PREEMPTIBLE))
    if required_regions:
        constraint_list.append(region_constraint(sorted(required_regions)))
    if required_zones:
        for z in sorted(required_zones):
            constraint_list.append(zone_constraint(z))
    proto_constraints = [c.to_proto() for c in constraint_list]

    return [
        DemandEntry(
            task_ids=[f"{task_prefix}-{i}"],
            coschedule_group_id=None,
            normalized=normalized,
            constraints=proto_constraints,
            resources=resources,
        )
        for i in range(count)
    ]


def make_big_demand_entries(
    count: int,
    *,
    cpu_millicores: int = 32000,
    memory_bytes: int = 32 * 1024**3,
    disk_bytes: int = 0,
    device_type: DeviceType = DeviceType.CPU,
    device_variants: frozenset[str] | None = None,
    task_prefix: str = "task",
    coschedule_group_id: str | None = None,
) -> list[DemandEntry]:
    """Create demand entries with explicit resource sizes for packing tests."""
    resources = job_pb2.ResourceSpecProto(
        cpu_millicores=cpu_millicores,
        memory_bytes=memory_bytes,
        disk_bytes=disk_bytes,
    )
    normalized = PlacementRequirements(
        device_type=device_type,
        device_variants=device_variants,
        preemptible=None,
        required_regions=None,
        required_zones=None,
    )
    if coschedule_group_id:
        return [
            DemandEntry(
                task_ids=[f"{task_prefix}-{i}" for i in range(count)],
                coschedule_group_id=coschedule_group_id,
                normalized=normalized,
                constraints=[],
                resources=resources,
            )
        ]
    return [
        DemandEntry(
            task_ids=[f"{task_prefix}-{i}"],
            coschedule_group_id=None,
            normalized=normalized,
            constraints=[],
            resources=resources,
        )
        for i in range(count)
    ]


def make_autoscaler(
    scale_groups: dict[str, ScalingGroup],
    config: config_pb2.AutoscalerConfig | None = None,
    platform: MagicMock | None = None,
    base_worker_config: config_pb2.WorkerConfig | None = None,
) -> Autoscaler:
    """Create an Autoscaler with the given groups."""
    mock_platform = platform or make_mock_platform()

    if config:
        return Autoscaler.from_config(
            scale_groups=scale_groups,
            config=config,
            platform=mock_platform,
            base_worker_config=base_worker_config,
        )
    else:
        return Autoscaler(
            scale_groups=scale_groups,
            evaluation_interval=Duration.from_seconds(0.1),
            platform=mock_platform,
            base_worker_config=base_worker_config,
        )


def mark_discovered_ready(group: ScalingGroup, handles: list[MagicMock], timestamp: Timestamp | None = None) -> None:
    """Mark discovered slices as READY with their worker IDs."""
    for handle in handles:
        worker_ids = [w.worker_id for w in handle.describe().workers]
        group.mark_slice_ready(handle.slice_id, worker_ids, timestamp=timestamp)


def mark_all_slices_ready(group: ScalingGroup) -> None:
    """Mark all tracked slices as READY with their worker IDs.

    Used after advance_all_tpus() to simulate the controller detecting that
    slices have finished booting and are ready for work.
    """
    for handle in group.slice_handles():
        desc = handle.describe()
        if desc.state == CloudSliceState.READY:
            worker_ids = [w.worker_id for w in desc.workers]
            group.mark_slice_ready(handle.slice_id, worker_ids)


def make_gcp_provider(
    config: config_pb2.ScaleGroupConfig,
    zone: str = "us-central1-a",
) -> tuple[GcpWorkerProvider, InMemoryGcpService]:
    """Create a GcpWorkerProvider backed by InMemoryGcpService(DRY_RUN).

    Returns both the provider and the backing service so tests can inject
    failures and advance TPU state.
    """
    service = InMemoryGcpService(mode=ServiceMode.DRY_RUN, project_id="test-project", label_prefix="iris")
    gcp_config = config_pb2.GcpPlatformConfig(project_id="test-project", zones=[zone])
    provider = GcpWorkerProvider(gcp_config=gcp_config, label_prefix="iris", gcp_service=service)
    return provider, service


def advance_all_tpus(service: InMemoryGcpService, state: str = "READY") -> None:
    """Transition all TPUs in an InMemoryGcpService to the given state."""
    for name, zone in list(service._tpus.keys()):
        if service._tpus[(name, zone)].state != state:
            service.advance_tpu_state(name, zone, state)


def set_task_band(db: ControllerDB, task_id: JobName, band: int) -> None:
    """Directly set priority_band on a task row for testing.

    Prefer setting priority_band on the LaunchJobRequest for new submissions.
    This helper is still needed for tests that change a task's band mid-flight
    (e.g., simulating admin band overrides or budget-triggered demotions).
    """
    db.execute("UPDATE tasks SET priority_band = ? WHERE task_id = ?", (band, task_id.to_wire()))
