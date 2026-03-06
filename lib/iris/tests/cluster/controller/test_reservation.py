# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the claims-based reservation system.

Tests cover:
- Worker claiming: matching workers to reservation entries, device filtering,
  one-claim-per-worker, entry count limits, multi-job independence.
- Stale claim cleanup: dead workers, finished jobs, preserved valid claims.
- Reservation gate: satisfied/unsatisfied/no-reservation checks.
- Taint injection: claimed workers get reservation-job attribute, ordering,
  and non-reservation jobs get NOT_EXISTS constraint.
"""


from iris.cluster.controller.controller import (
    RESERVATION_TAINT_KEY,
    Controller,
    ControllerConfig,
    ReservationClaim,
    _find_reservation_ancestor,
    _inject_reservation_taints,
    _inject_taint_constraints,
    _preference_pass,
    _reservation_region_constraints,
    _worker_matches_reservation_entry,
    job_requirements_from_job,
)
from iris.cluster.controller.events import (
    JobSubmittedEvent,
    TaskStateChangedEvent,
    WorkerRegisteredEvent,
)
from iris.cluster.controller.scheduler import JobRequirements, Scheduler, SchedulingContext
from iris.cluster.controller.state import ControllerState, ControllerWorker
from iris.cluster.constraints import WellKnownAttribute
from iris.cluster.constraints import (
    AttributeValue,
    device_variant_constraint,
    get_device_type,
    get_device_variant,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import cluster_pb2
from iris.time_utils import Timestamp

# =============================================================================
# Helpers
# =============================================================================


def _cpu_device() -> cluster_pb2.DeviceConfig:
    return cluster_pb2.DeviceConfig(cpu=cluster_pb2.CpuDevice(variant="cpu"))


def _gpu_device(variant: str = "H100", count: int = 8) -> cluster_pb2.DeviceConfig:
    return cluster_pb2.DeviceConfig(gpu=cluster_pb2.GpuDevice(variant=variant, count=count))


def _cpu_metadata() -> cluster_pb2.WorkerMetadata:
    meta = cluster_pb2.WorkerMetadata(
        hostname="test",
        ip_address="127.0.0.1",
        cpu_count=8,
        memory_bytes=16 * 1024**3,
        disk_bytes=100 * 1024**3,
        device=_cpu_device(),
    )
    meta.attributes[WellKnownAttribute.DEVICE_TYPE].CopyFrom(cluster_pb2.AttributeValue(string_value="cpu"))
    return meta


def _gpu_metadata(variant: str = "H100") -> cluster_pb2.WorkerMetadata:
    meta = cluster_pb2.WorkerMetadata(
        hostname="test-gpu",
        ip_address="127.0.0.1",
        cpu_count=32,
        memory_bytes=256 * 1024**3,
        disk_bytes=500 * 1024**3,
        device=_gpu_device(variant),
    )
    meta.attributes[WellKnownAttribute.DEVICE_TYPE].CopyFrom(cluster_pb2.AttributeValue(string_value="gpu"))
    meta.attributes[WellKnownAttribute.DEVICE_VARIANT].CopyFrom(cluster_pb2.AttributeValue(string_value=variant.lower()))
    return meta


def _default_attributes_for_device(device: cluster_pb2.DeviceConfig) -> dict[str, AttributeValue]:
    """Build the worker attributes that the real env_probe would set from config."""
    attrs: dict[str, AttributeValue] = {}
    dt = get_device_type(device)
    attrs[WellKnownAttribute.DEVICE_TYPE] = AttributeValue(dt)
    dv = get_device_variant(device)
    if dv:
        attrs[WellKnownAttribute.DEVICE_VARIANT] = AttributeValue(dv.lower())
    return attrs


def _make_worker(
    worker_id: str,
    metadata: cluster_pb2.WorkerMetadata | None = None,
    attributes: dict[str, AttributeValue] | None = None,
    healthy: bool = True,
) -> ControllerWorker:
    meta = metadata or _cpu_metadata()
    # Workers always have device attributes from config (Stage 3).
    # Merge explicit attributes on top of the device-derived defaults.
    default_attrs = _default_attributes_for_device(meta.device)
    if attributes:
        default_attrs.update(attributes)
    return ControllerWorker(
        worker_id=WorkerId(worker_id),
        address=f"{worker_id}:8080",
        metadata=meta,
        attributes=default_attrs,
        healthy=healthy,
    )


def _make_reservation_entry(
    device: cluster_pb2.DeviceConfig | None = None,
    constraints: list[cluster_pb2.Constraint] | None = None,
) -> cluster_pb2.ReservationEntry:
    dev = device or _cpu_device()
    return cluster_pb2.ReservationEntry(
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=dev,
        ),
        constraints=constraints or [],
    )


def _entrypoint() -> cluster_pb2.RuntimeEntrypoint:
    entrypoint = cluster_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    return entrypoint


def _make_job_request_with_reservation(
    name: str = "res-job",
    reservation_entries: list[cluster_pb2.ReservationEntry] | None = None,
) -> cluster_pb2.Controller.LaunchJobRequest:
    req = cluster_pb2.Controller.LaunchJobRequest(
        name=name,
        entrypoint=_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    if reservation_entries:
        for entry in reservation_entries:
            req.reservation.entries.append(entry)
    return req


class FakeStubFactory:
    """Minimal stub factory satisfying the WorkerStubFactory protocol."""

    def get_stub(self, address: str):
        raise NotImplementedError("stub not needed in unit tests")

    def evict(self, address: str) -> None:
        pass


def _make_controller() -> Controller:
    """Create a Controller with minimal config for unit testing."""
    config = ControllerConfig(
        bundle_prefix="file:///tmp/iris-test-bundles",
    )
    return Controller(config=config, worker_stub_factory=FakeStubFactory())


def _register_worker(
    state: ControllerState,
    worker_id: str,
    metadata: cluster_pb2.WorkerMetadata | None = None,
) -> WorkerId:
    wid = WorkerId(worker_id)
    state.handle_event(
        WorkerRegisteredEvent(
            worker_id=wid,
            address=f"{worker_id}:8080",
            metadata=metadata or _cpu_metadata(),
            timestamp=Timestamp.now(),
        )
    )
    return wid


def _submit_job(
    state: ControllerState,
    job_id: str,
    request: cluster_pb2.Controller.LaunchJobRequest,
) -> JobName:
    jid = JobName.root("test-user", job_id)
    request.name = jid.to_wire()
    state.handle_event(
        JobSubmittedEvent(
            job_id=jid,
            request=request,
            timestamp=Timestamp.now(),
        )
    )
    return jid


# =============================================================================
# _worker_matches_reservation_entry
# =============================================================================


def test_worker_matches_cpu_reservation_entry():
    """CPU worker matches a CPU reservation entry."""
    worker = _make_worker("w1", _cpu_metadata())
    entry = _make_reservation_entry(_cpu_device())
    assert _worker_matches_reservation_entry(worker, entry)


def test_worker_matches_gpu_reservation_entry():
    """GPU worker matches a GPU reservation entry of the same variant."""
    worker = _make_worker("w1", _gpu_metadata("H100"))
    entry = _make_reservation_entry(_gpu_device("H100"))
    assert _worker_matches_reservation_entry(worker, entry)


def test_worker_rejects_wrong_device_type():
    """CPU worker does not match a GPU reservation entry."""
    worker = _make_worker("w1", _cpu_metadata())
    entry = _make_reservation_entry(_gpu_device("H100"))
    assert not _worker_matches_reservation_entry(worker, entry)


def test_worker_rejects_wrong_gpu_variant():
    """H100 worker does not match an A100 reservation entry."""
    worker = _make_worker("w1", _gpu_metadata("H100"))
    entry = _make_reservation_entry(_gpu_device("A100"))
    assert not _worker_matches_reservation_entry(worker, entry)


def test_worker_matches_with_constraint():
    """Worker with matching attribute satisfies a constraint on the entry."""
    worker = _make_worker(
        "w1",
        _cpu_metadata(),
        attributes={WellKnownAttribute.REGION: AttributeValue("us-central1")},
    )
    constraint = cluster_pb2.Constraint(
        key=WellKnownAttribute.REGION,
        op=cluster_pb2.CONSTRAINT_OP_EQ,
        value=cluster_pb2.AttributeValue(string_value="us-central1"),
    )
    entry = _make_reservation_entry(_cpu_device(), constraints=[constraint])
    assert _worker_matches_reservation_entry(worker, entry)


def test_worker_rejects_unmet_constraint():
    """Worker without the required attribute fails the constraint check."""
    worker = _make_worker("w1", _cpu_metadata())
    constraint = cluster_pb2.Constraint(
        key=WellKnownAttribute.REGION,
        op=cluster_pb2.CONSTRAINT_OP_EQ,
        value=cluster_pb2.AttributeValue(string_value="us-central1"),
    )
    entry = _make_reservation_entry(_cpu_device(), constraints=[constraint])
    assert not _worker_matches_reservation_entry(worker, entry)


# =============================================================================
# _claim_workers_for_reservations (via Controller)
# =============================================================================


def test_claim_eligible_worker():
    """An eligible worker is claimed for a reservation entry."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "w1")
    req = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry()],
    )
    _submit_job(ctrl.state, "j1", req)

    ctrl._claim_workers_for_reservations()

    assert WorkerId("w1") in ctrl.reservation_claims
    claim = ctrl.reservation_claims[WorkerId("w1")]
    assert claim.job_id == JobName.root("test-user", "j1").to_wire()
    assert claim.entry_idx == 0


def test_claim_rejects_wrong_device():
    """A worker with the wrong device type is not claimed."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "w1", _cpu_metadata())
    req = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry(_gpu_device("H100"))],
    )
    _submit_job(ctrl.state, "j1", req)

    ctrl._claim_workers_for_reservations()

    assert len(ctrl.reservation_claims) == 0


def test_claim_one_per_worker():
    """A single worker cannot be claimed by two different reservation entries."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "w1")
    req = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry(), _make_reservation_entry()],
    )
    _submit_job(ctrl.state, "j1", req)

    ctrl._claim_workers_for_reservations()

    assert len(ctrl.reservation_claims) == 1
    assert WorkerId("w1") in ctrl.reservation_claims


def test_claim_respects_entry_count():
    """Two workers can satisfy a 2-entry reservation."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "w1")
    _register_worker(ctrl.state, "w2")
    req = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry(), _make_reservation_entry()],
    )
    _submit_job(ctrl.state, "j1", req)

    ctrl._claim_workers_for_reservations()

    assert len(ctrl.reservation_claims) == 2
    job_wire = JobName.root("test-user", "j1").to_wire()
    claimed_entries = {c.entry_idx for c in ctrl.reservation_claims.values() if c.job_id == job_wire}
    assert claimed_entries == {0, 1}


def test_claim_does_not_exceed_entry_count():
    """Extra workers beyond entry count are not claimed."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "w1")
    _register_worker(ctrl.state, "w2")
    _register_worker(ctrl.state, "w3")
    req = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry()],
    )
    _submit_job(ctrl.state, "j1", req)

    ctrl._claim_workers_for_reservations()

    assert len(ctrl.reservation_claims) == 1


def test_claim_independent_per_job():
    """Claims for different jobs don't interfere with each other."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "w1")
    _register_worker(ctrl.state, "w2")

    req_a = _make_job_request_with_reservation(
        name="job-a",
        reservation_entries=[_make_reservation_entry()],
    )
    _submit_job(ctrl.state, "job-a", req_a)

    req_b = _make_job_request_with_reservation(
        name="job-b",
        reservation_entries=[_make_reservation_entry()],
    )
    _submit_job(ctrl.state, "job-b", req_b)

    ctrl._claim_workers_for_reservations()

    assert len(ctrl.reservation_claims) == 2
    job_ids = {c.job_id for c in ctrl.reservation_claims.values()}
    assert job_ids == {
        JobName.root("test-user", "job-a").to_wire(),
        JobName.root("test-user", "job-b").to_wire(),
    }


def test_claim_skips_unhealthy_worker():
    """Unhealthy workers are not claimed."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "w1")
    # Mark worker unhealthy
    worker = ctrl.state.get_worker(WorkerId("w1"))
    worker.healthy = False

    req = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry()],
    )
    _submit_job(ctrl.state, "j1", req)

    ctrl._claim_workers_for_reservations()

    assert len(ctrl.reservation_claims) == 0


def test_claim_idempotent():
    """Running claiming twice doesn't duplicate claims."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "w1")
    req = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry()],
    )
    _submit_job(ctrl.state, "j1", req)

    ctrl._claim_workers_for_reservations()
    ctrl._claim_workers_for_reservations()

    assert len(ctrl.reservation_claims) == 1


# =============================================================================
# _cleanup_stale_claims
# =============================================================================


def test_cleanup_removes_dead_worker_claims():
    """Claims for workers no longer in state are removed."""
    ctrl = _make_controller()
    w1 = _register_worker(ctrl.state, "w1")
    req = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry()],
    )
    _submit_job(ctrl.state, "j1", req)
    ctrl._claim_workers_for_reservations()
    assert len(ctrl.reservation_claims) == 1

    # Simulate worker disappearing by injecting a claim for a non-existent worker
    ctrl._reservation_claims[WorkerId("dead-worker")] = ReservationClaim(
        job_id=JobName.root("test-user", "j1").to_wire(),
        entry_idx=99,
    )
    assert len(ctrl.reservation_claims) == 2

    ctrl._cleanup_stale_claims()

    # dead-worker removed, w1 preserved
    assert WorkerId("dead-worker") not in ctrl.reservation_claims
    assert w1 in ctrl.reservation_claims


def test_cleanup_removes_finished_job_claims():
    """Claims for finished jobs are removed."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "w1")
    req = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry()],
    )
    jid = _submit_job(ctrl.state, "j1", req)
    ctrl._claim_workers_for_reservations()
    assert len(ctrl.reservation_claims) == 1

    # Kill the job to mark it as finished
    tasks = ctrl.state.get_job_tasks(jid)
    for task in tasks:
        ctrl.state.handle_event(
            TaskStateChangedEvent(
                task_id=task.task_id,
                new_state=cluster_pb2.TASK_STATE_KILLED,
                attempt_id=task.current_attempt_id,
            )
        )

    job = ctrl.state.get_job(jid)
    assert job.is_finished()

    ctrl._cleanup_stale_claims()

    assert len(ctrl.reservation_claims) == 0


def test_cleanup_preserves_valid_claims():
    """Valid claims (healthy worker, active job) are preserved."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "w1")
    req = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry()],
    )
    _submit_job(ctrl.state, "j1", req)
    ctrl._claim_workers_for_reservations()

    ctrl._cleanup_stale_claims()

    assert len(ctrl.reservation_claims) == 1


# =============================================================================
# _is_reservation_satisfied (gate check)
# =============================================================================


def test_gate_satisfied_when_claims_meet_entries():
    """Gate opens when claimed workers >= reservation entries."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "w1")
    req = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry()],
    )
    jid = _submit_job(ctrl.state, "j1", req)
    ctrl._claim_workers_for_reservations()

    job = ctrl.state.get_job(jid)
    assert ctrl._is_reservation_satisfied(job)


def test_gate_unsatisfied_when_claims_below_entries():
    """Gate stays closed when fewer workers are claimed than entries required."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "w1")
    req = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry(), _make_reservation_entry()],
    )
    jid = _submit_job(ctrl.state, "j1", req)
    ctrl._claim_workers_for_reservations()

    job = ctrl.state.get_job(jid)
    # Only 1 worker available for 2 entries
    assert not ctrl._is_reservation_satisfied(job)


def test_gate_satisfied_for_jobs_without_reservation():
    """Jobs without a reservation always pass the gate."""
    ctrl = _make_controller()
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="no-res",
        entrypoint=_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    jid = _submit_job(ctrl.state, "no-res", req)

    job = ctrl.state.get_job(jid)
    assert ctrl._is_reservation_satisfied(job)


# =============================================================================
# _inject_reservation_taints
# =============================================================================


def test_taint_injection_adds_attribute_to_claimed_workers():
    """Claimed workers get the reservation-job attribute set to the job ID."""
    w1 = _make_worker("w1")
    w2 = _make_worker("w2")
    claims = {WorkerId("w1"): ReservationClaim(job_id="job-a", entry_idx=0)}

    result = _inject_reservation_taints([w1, w2], claims)

    # w1 should have the taint
    tainted = [w for w in result if w.worker_id == WorkerId("w1")]
    assert len(tainted) == 1
    assert RESERVATION_TAINT_KEY in tainted[0].attributes
    assert tainted[0].attributes[RESERVATION_TAINT_KEY] == AttributeValue("job-a")


def test_taint_injection_unclaimed_workers_no_attribute():
    """Unclaimed workers do not get the reservation-job attribute."""
    w1 = _make_worker("w1")
    w2 = _make_worker("w2")
    claims = {WorkerId("w1"): ReservationClaim(job_id="job-a", entry_idx=0)}

    result = _inject_reservation_taints([w1, w2], claims)

    unclaimed = [w for w in result if w.worker_id == WorkerId("w2")]
    assert len(unclaimed) == 1
    assert RESERVATION_TAINT_KEY not in unclaimed[0].attributes


def test_taint_injection_claimed_workers_first():
    """Claimed workers appear before unclaimed workers in the returned list."""
    w1 = _make_worker("w1")
    w2 = _make_worker("w2")
    w3 = _make_worker("w3")
    # Only w2 is claimed
    claims = {WorkerId("w2"): ReservationClaim(job_id="job-a", entry_idx=0)}

    result = _inject_reservation_taints([w1, w2, w3], claims)

    assert result[0].worker_id == WorkerId("w2")
    unclaimed_ids = [w.worker_id for w in result[1:]]
    assert set(unclaimed_ids) == {WorkerId("w1"), WorkerId("w3")}


def test_taint_injection_no_claims_returns_original_list():
    """When there are no claims, the original list is returned unchanged."""
    w1 = _make_worker("w1")
    w2 = _make_worker("w2")

    result = _inject_reservation_taints([w1, w2], {})

    assert result == [w1, w2] or result == [w1, w2]
    # With no claims, the function returns the input list directly
    assert result[0].worker_id == WorkerId("w1")
    assert result[1].worker_id == WorkerId("w2")


def test_taint_injection_does_not_mutate_original():
    """The original worker objects are not mutated."""
    w1 = _make_worker("w1")
    original_attrs = dict(w1.attributes)
    claims = {WorkerId("w1"): ReservationClaim(job_id="job-a", entry_idx=0)}

    _inject_reservation_taints([w1], claims)

    assert w1.attributes == original_attrs


# =============================================================================
# _inject_taint_constraints
# =============================================================================


def _make_job_requirements() -> JobRequirements:
    return JobRequirements(
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        constraints=[],
        is_coscheduled=False,
        coscheduling_group_by=None,
    )


def test_taint_constraint_added_to_non_reservation_jobs():
    """Non-reservation jobs get a NOT_EXISTS reservation-job constraint."""
    jobs = {
        JobName.root("test-user", "regular"): _make_job_requirements(),
    }
    has_reservation: set[JobName] = set()

    result = _inject_taint_constraints(jobs, has_reservation)

    constraints = result[JobName.root("test-user", "regular")].constraints
    not_exists = [c for c in constraints if c.key == RESERVATION_TAINT_KEY]
    assert len(not_exists) == 1
    assert not_exists[0].op == cluster_pb2.CONSTRAINT_OP_NOT_EXISTS


def test_taint_constraint_not_added_to_reservation_jobs():
    """Reservation jobs do not get the NOT_EXISTS constraint."""
    res_job = JobName.root("test-user", "reserved")
    jobs = {
        res_job: _make_job_requirements(),
    }
    has_reservation = {res_job}

    result = _inject_taint_constraints(jobs, has_reservation)

    constraints = result[res_job].constraints
    not_exists = [c for c in constraints if c.key == RESERVATION_TAINT_KEY]
    assert len(not_exists) == 0


def test_taint_constraint_mixed_jobs():
    """With both reservation and regular jobs, only regular jobs get the constraint."""
    res_job = JobName.root("test-user", "reserved")
    reg_job = JobName.root("test-user", "regular")
    jobs = {
        res_job: _make_job_requirements(),
        reg_job: _make_job_requirements(),
    }
    has_reservation = {res_job}

    result = _inject_taint_constraints(jobs, has_reservation)

    # Reserved job: no taint constraint
    res_constraints = [c for c in result[res_job].constraints if c.key == RESERVATION_TAINT_KEY]
    assert len(res_constraints) == 0

    # Regular job: has taint constraint
    reg_constraints = [c for c in result[reg_job].constraints if c.key == RESERVATION_TAINT_KEY]
    assert len(reg_constraints) == 1
    assert reg_constraints[0].op == cluster_pb2.CONSTRAINT_OP_NOT_EXISTS


def test_taint_constraint_preserves_existing_constraints():
    """Existing constraints are preserved when the taint constraint is added."""
    existing = cluster_pb2.Constraint(
        key=WellKnownAttribute.REGION,
        op=cluster_pb2.CONSTRAINT_OP_EQ,
        value=cluster_pb2.AttributeValue(string_value="us-central1"),
    )
    jobs = {
        JobName.root("test-user", "regular"): JobRequirements(
            resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            constraints=[existing],
            is_coscheduled=False,
            coscheduling_group_by=None,
        ),
    }
    has_reservation: set[JobName] = set()

    result = _inject_taint_constraints(jobs, has_reservation)

    constraints = result[JobName.root("test-user", "regular")].constraints
    assert len(constraints) == 2
    # Original constraint preserved
    region_constraints = [c for c in constraints if c.key == WellKnownAttribute.REGION]
    assert len(region_constraints) == 1
    # Taint constraint added
    taint_constraints = [c for c in constraints if c.key == RESERVATION_TAINT_KEY]
    assert len(taint_constraints) == 1


# =============================================================================
# _preference_pass
# =============================================================================


def _build_context_with_workers(
    workers: list[ControllerWorker],
    pending_tasks: list[JobName],
    jobs: dict[JobName, JobRequirements],
) -> tuple["Scheduler", "SchedulingContext"]:
    scheduler = Scheduler()
    return scheduler.create_scheduling_context(
        workers,
        pending_tasks=pending_tasks,
        jobs=jobs,
    )


def test_preference_pass_assigns_to_claimed_worker():
    """Reservation task is assigned to its claimed worker."""
    w1 = _make_worker("w1")
    w2 = _make_worker("w2")
    job_id = JobName.root("test-user", "res-job")
    task_id = job_id.task(0)
    req = _make_job_requirements()
    has_reservation = {job_id}
    claims = {WorkerId("w1"): ReservationClaim(job_id=job_id.to_wire(), entry_idx=0)}

    context = _build_context_with_workers(
        [w1, w2],
        pending_tasks=[task_id],
        jobs={job_id: req},
    )

    assignments = _preference_pass(context, has_reservation, claims)

    assert len(assignments) == 1
    assert assignments[0] == (task_id, WorkerId("w1"))
    assert task_id not in context.pending_tasks


def test_preference_pass_falls_through_on_no_capacity():
    """When claimed worker is at capacity, the task stays in pending_tasks."""
    w1 = _make_worker("w1")
    job_id = JobName.root("test-user", "res-job")
    task_id = job_id.task(0)
    # Request more CPU than the worker has
    req = JobRequirements(
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=999_000, memory_bytes=1024**3),
        constraints=[],
        is_coscheduled=False,
        coscheduling_group_by=None,
    )
    has_reservation = {job_id}
    claims = {WorkerId("w1"): ReservationClaim(job_id=job_id.to_wire(), entry_idx=0)}

    context = _build_context_with_workers(
        [w1],
        pending_tasks=[task_id],
        jobs={job_id: req},
    )

    assignments = _preference_pass(context, has_reservation, claims)

    assert len(assignments) == 0
    assert task_id in context.pending_tasks


def test_preference_pass_skips_non_reservation_tasks():
    """Non-reservation tasks are not touched by the preference pass."""
    w1 = _make_worker("w1")
    job_id = JobName.root("test-user", "regular-job")
    task_id = job_id.task(0)
    req = _make_job_requirements()
    has_reservation: set[JobName] = set()
    claims = {WorkerId("w1"): ReservationClaim(job_id="other-job", entry_idx=0)}

    context = _build_context_with_workers(
        [w1],
        pending_tasks=[task_id],
        jobs={job_id: req},
    )

    assignments = _preference_pass(context, has_reservation, claims)

    assert len(assignments) == 0
    assert task_id in context.pending_tasks


def test_preference_pass_skips_coscheduled_jobs():
    """Coscheduled reservation tasks are left for find_assignments."""
    w1 = _make_worker("w1")
    job_id = JobName.root("test-user", "cosched-job")
    task_id = job_id.task(0)
    req = JobRequirements(
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        constraints=[],
        is_coscheduled=True,
        coscheduling_group_by=WellKnownAttribute.TPU_NAME,
    )
    has_reservation = {job_id}
    claims = {WorkerId("w1"): ReservationClaim(job_id=job_id.to_wire(), entry_idx=0)}

    context = _build_context_with_workers(
        [w1],
        pending_tasks=[task_id],
        jobs={job_id: req},
    )

    assignments = _preference_pass(context, has_reservation, claims)

    assert len(assignments) == 0
    assert task_id in context.pending_tasks


def test_preference_pass_no_claims_returns_empty():
    """With no claims, preference pass is a no-op."""
    w1 = _make_worker("w1")
    job_id = JobName.root("test-user", "res-job")
    task_id = job_id.task(0)
    req = _make_job_requirements()
    has_reservation = {job_id}

    context = _build_context_with_workers(
        [w1],
        pending_tasks=[task_id],
        jobs={job_id: req},
    )

    assignments = _preference_pass(context, has_reservation, {})

    assert len(assignments) == 0
    assert task_id in context.pending_tasks


def test_preference_pass_deducts_capacity():
    """After preference assignment, the claimed worker's capacity is consumed."""
    w1 = _make_worker("w1")
    job_id = JobName.root("test-user", "res-job")
    task_id_0 = job_id.task(0)
    task_id_1 = job_id.task(1)
    # Each task wants 4000m CPU; w1 has 8000m, so only one fits.
    req = JobRequirements(
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=4000, memory_bytes=1024**3),
        constraints=[],
        is_coscheduled=False,
        coscheduling_group_by=None,
    )
    has_reservation = {job_id}
    claims = {WorkerId("w1"): ReservationClaim(job_id=job_id.to_wire(), entry_idx=0)}

    context = _build_context_with_workers(
        [w1, _make_worker("w2")],
        pending_tasks=[task_id_0, task_id_1],
        jobs={job_id: req},
    )

    assignments = _preference_pass(context, has_reservation, claims)

    # First task assigned to w1; second stays pending (w1 already scheduled this cycle)
    assert len(assignments) == 1
    assert assignments[0] == (task_id_0, WorkerId("w1"))
    assert task_id_0 not in context.pending_tasks
    assert task_id_1 in context.pending_tasks


# =============================================================================
# _reservation_region_constraints
# =============================================================================


def test_region_constraint_injected_from_claimed_workers():
    """Region constraint is injected when claimed workers have a region attribute."""
    ctrl = _make_controller()
    w1 = _register_worker(ctrl.state, "w1")
    # Set region attribute on worker
    worker = ctrl.state.get_worker(w1)
    worker.attributes[WellKnownAttribute.REGION] = AttributeValue("us-central1")

    req = _make_job_request_with_reservation(reservation_entries=[_make_reservation_entry()])
    jid = _submit_job(ctrl.state, "j1", req)
    ctrl._claim_workers_for_reservations()

    result = _reservation_region_constraints(
        jid.to_wire(),
        ctrl.reservation_claims,
        ctrl.state,
        [],
    )

    assert len(result) == 1
    assert result[0].key == WellKnownAttribute.REGION
    assert result[0].op == cluster_pb2.CONSTRAINT_OP_EQ
    assert result[0].value.string_value == "us-central1"


def test_region_constraint_not_injected_when_already_present():
    """Existing region constraint prevents injection."""
    ctrl = _make_controller()
    w1 = _register_worker(ctrl.state, "w1")
    worker = ctrl.state.get_worker(w1)
    worker.attributes[WellKnownAttribute.REGION] = AttributeValue("us-central1")

    req = _make_job_request_with_reservation(reservation_entries=[_make_reservation_entry()])
    jid = _submit_job(ctrl.state, "j1", req)
    ctrl._claim_workers_for_reservations()

    existing = cluster_pb2.Constraint(
        key=WellKnownAttribute.REGION,
        op=cluster_pb2.CONSTRAINT_OP_EQ,
        value=cluster_pb2.AttributeValue(string_value="us-east1"),
    )
    result = _reservation_region_constraints(
        jid.to_wire(),
        ctrl.reservation_claims,
        ctrl.state,
        [existing],
    )

    assert len(result) == 1
    assert result[0] is existing


def test_region_constraint_not_injected_when_no_region_attr():
    """No injection when claimed workers lack region attributes."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "w1")

    req = _make_job_request_with_reservation(reservation_entries=[_make_reservation_entry()])
    jid = _submit_job(ctrl.state, "j1", req)
    ctrl._claim_workers_for_reservations()

    result = _reservation_region_constraints(
        jid.to_wire(),
        ctrl.reservation_claims,
        ctrl.state,
        [],
    )

    assert result == []


def test_region_constraint_multiple_regions():
    """IN constraint injected when claimed workers span multiple regions."""
    ctrl = _make_controller()
    w1 = _register_worker(ctrl.state, "w1")
    w2 = _register_worker(ctrl.state, "w2")
    ctrl.state.get_worker(w1).attributes[WellKnownAttribute.REGION] = AttributeValue("us-central1")
    ctrl.state.get_worker(w2).attributes[WellKnownAttribute.REGION] = AttributeValue("us-east1")

    req = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry(), _make_reservation_entry()],
    )
    jid = _submit_job(ctrl.state, "j1", req)
    ctrl._claim_workers_for_reservations()

    result = _reservation_region_constraints(
        jid.to_wire(),
        ctrl.reservation_claims,
        ctrl.state,
        [],
    )

    assert len(result) == 1
    assert result[0].key == WellKnownAttribute.REGION
    assert result[0].op == cluster_pb2.CONSTRAINT_OP_IN
    regions = {v.string_value for v in result[0].values}
    assert regions == {"us-central1", "us-east1"}


def test_no_injection_for_non_reservation_job():
    """No claims for this job → constraints returned unchanged."""
    ctrl = _make_controller()
    w1 = _register_worker(ctrl.state, "w1")
    ctrl.state.get_worker(w1).attributes[WellKnownAttribute.REGION] = AttributeValue("us-central1")

    # Claim w1 for a different job
    req = _make_job_request_with_reservation(reservation_entries=[_make_reservation_entry()])
    _submit_job(ctrl.state, "other-job", req)
    ctrl._claim_workers_for_reservations()

    result = _reservation_region_constraints(
        "/test-user/unrelated-job",
        ctrl.reservation_claims,
        ctrl.state,
        [],
    )

    assert result == []


# =============================================================================
# _find_reservation_ancestor
# =============================================================================


def test_find_reservation_ancestor_returns_parent_with_reservation():
    """Direct parent with reservation is found."""
    ctrl = _make_controller()
    parent_req = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry()],
    )
    parent_jid = _submit_job(ctrl.state, "res-parent", parent_req)

    child_jid = JobName.from_string("/test-user/res-parent/child")
    child_req = cluster_pb2.Controller.LaunchJobRequest(
        name=child_jid.to_wire(),
        entrypoint=_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    ctrl.state.handle_event(
        JobSubmittedEvent(
            job_id=child_jid,
            request=child_req,
            timestamp=Timestamp.now(),
        )
    )

    result = _find_reservation_ancestor(ctrl.state, child_jid)
    assert result == parent_jid


def test_find_reservation_ancestor_returns_grandparent():
    """Grandparent with reservation is found when parent has none."""
    ctrl = _make_controller()
    # Grandparent with reservation
    gp_req = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry()],
    )
    gp_jid = _submit_job(ctrl.state, "gp", gp_req)

    # Parent (no reservation)
    parent_jid = JobName.from_string("/test-user/gp/parent")
    parent_req = cluster_pb2.Controller.LaunchJobRequest(
        name=parent_jid.to_wire(),
        entrypoint=_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    ctrl.state.handle_event(
        JobSubmittedEvent(
            job_id=parent_jid,
            request=parent_req,
            timestamp=Timestamp.now(),
        )
    )

    # Grandchild
    gc_jid = JobName.from_string("/test-user/gp/parent/gc")
    gc_req = cluster_pb2.Controller.LaunchJobRequest(
        name=gc_jid.to_wire(),
        entrypoint=_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    ctrl.state.handle_event(
        JobSubmittedEvent(
            job_id=gc_jid,
            request=gc_req,
            timestamp=Timestamp.now(),
        )
    )

    result = _find_reservation_ancestor(ctrl.state, gc_jid)
    assert result == gp_jid


def test_find_reservation_ancestor_returns_none_for_root_job():
    """Root job with no reservation returns None."""
    ctrl = _make_controller()
    req = cluster_pb2.Controller.LaunchJobRequest(
        name="no-res",
        entrypoint=_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    jid = _submit_job(ctrl.state, "no-res", req)
    assert _find_reservation_ancestor(ctrl.state, jid) is None


def test_find_reservation_ancestor_returns_none_when_no_ancestor_has_reservation():
    """Child of a non-reservation parent returns None."""
    ctrl = _make_controller()
    parent_req = cluster_pb2.Controller.LaunchJobRequest(
        name="plain-parent",
        entrypoint=_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    _submit_job(ctrl.state, "plain-parent", parent_req)

    child_jid = JobName.from_string("/test-user/plain-parent/child")
    child_req = cluster_pb2.Controller.LaunchJobRequest(
        name=child_jid.to_wire(),
        entrypoint=_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    ctrl.state.handle_event(
        JobSubmittedEvent(
            job_id=child_jid,
            request=child_req,
            timestamp=Timestamp.now(),
        )
    )

    assert _find_reservation_ancestor(ctrl.state, child_jid) is None


# =============================================================================
# Ancestry-based taint exemption (integration)
# =============================================================================


def test_taint_exemption_for_children_of_reservation_job():
    """Children of a reservation job are not blocked from claimed workers."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "w1", _gpu_metadata("H100"))
    _register_worker(ctrl.state, "w2", _gpu_metadata("H100"))

    # Parent job with reservation claiming both GPU workers
    parent_req = _make_job_request_with_reservation(
        reservation_entries=[
            _make_reservation_entry(_gpu_device("H100")),
            _make_reservation_entry(_gpu_device("H100")),
        ],
    )
    _submit_job(ctrl.state, "res-parent", parent_req)
    ctrl._claim_workers_for_reservations()
    assert len(ctrl.reservation_claims) == 2

    # Child job (NO reservation) requesting GPU
    child_jid = JobName.from_string("/test-user/res-parent/child")
    child_req = cluster_pb2.Controller.LaunchJobRequest(
        name=child_jid.to_wire(),
        entrypoint=_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_gpu_device("H100"),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    ctrl.state.handle_event(
        JobSubmittedEvent(
            job_id=child_jid,
            request=child_req,
            timestamp=Timestamp.now(),
        )
    )

    # Build scheduling state — child should be in has_reservation
    pending = ctrl.state.peek_pending_tasks()
    jobs: dict[JobName, JobRequirements] = {}
    has_reservation: set[JobName] = set()
    for task in pending:
        job = ctrl.state.get_job(task.job_id)
        if job and not job.is_finished():
            jobs[task.job_id] = job_requirements_from_job(job)
            if job.request.HasField("reservation"):
                has_reservation.add(task.job_id)
            elif _find_reservation_ancestor(ctrl.state, task.job_id) is not None:
                has_reservation.add(task.job_id)

    assert child_jid in has_reservation

    # Child does NOT get NOT_EXISTS constraint
    modified_jobs = _inject_taint_constraints(jobs, has_reservation)
    child_constraints = modified_jobs[child_jid].constraints
    not_exists = [c for c in child_constraints if c.key == RESERVATION_TAINT_KEY]
    assert len(not_exists) == 0


def test_grandchildren_inherit_reservation_from_ancestor():
    """Grandchildren of a reservation job inherit taint exemption."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "h1", _gpu_metadata("H100"))
    _register_worker(ctrl.state, "h2", _gpu_metadata("H100"))
    _register_worker(ctrl.state, "a1", _gpu_metadata("A100"))
    _register_worker(ctrl.state, "a2", _gpu_metadata("A100"))

    # Root job (CPU, no reservation)
    root_req = cluster_pb2.Controller.LaunchJobRequest(
        name="root",
        entrypoint=_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    _submit_job(ctrl.state, "root", root_req)

    # Child-A reserves 2 H100
    child_a_jid = JobName.from_string("/test-user/root/child-a")
    child_a_req = _make_job_request_with_reservation(
        reservation_entries=[
            _make_reservation_entry(_gpu_device("H100")),
            _make_reservation_entry(_gpu_device("H100")),
        ],
    )
    child_a_req.name = child_a_jid.to_wire()
    ctrl.state.handle_event(
        JobSubmittedEvent(
            job_id=child_a_jid,
            request=child_a_req,
            timestamp=Timestamp.now(),
        )
    )

    # Child-B reserves 2 A100
    child_b_jid = JobName.from_string("/test-user/root/child-b")
    child_b_req = _make_job_request_with_reservation(
        reservation_entries=[
            _make_reservation_entry(_gpu_device("A100")),
            _make_reservation_entry(_gpu_device("A100")),
        ],
    )
    child_b_req.name = child_b_jid.to_wire()
    ctrl.state.handle_event(
        JobSubmittedEvent(
            job_id=child_b_jid,
            request=child_b_req,
            timestamp=Timestamp.now(),
        )
    )

    ctrl._claim_workers_for_reservations()
    assert len(ctrl.reservation_claims) == 4

    # Grandchild-A (under child-A) requesting H100
    gc_a_jid = JobName.from_string("/test-user/root/child-a/gc-a")
    gc_a_req = cluster_pb2.Controller.LaunchJobRequest(
        name=gc_a_jid.to_wire(),
        entrypoint=_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_gpu_device("H100"),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    ctrl.state.handle_event(
        JobSubmittedEvent(
            job_id=gc_a_jid,
            request=gc_a_req,
            timestamp=Timestamp.now(),
        )
    )

    # Grandchild-B (under child-B) requesting A100
    gc_b_jid = JobName.from_string("/test-user/root/child-b/gc-b")
    gc_b_req = cluster_pb2.Controller.LaunchJobRequest(
        name=gc_b_jid.to_wire(),
        entrypoint=_entrypoint(),
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_gpu_device("A100"),
        ),
        environment=cluster_pb2.EnvironmentConfig(),
        replicas=1,
    )
    ctrl.state.handle_event(
        JobSubmittedEvent(
            job_id=gc_b_jid,
            request=gc_b_req,
            timestamp=Timestamp.now(),
        )
    )

    # Build scheduling state
    pending = ctrl.state.peek_pending_tasks()
    jobs: dict[JobName, JobRequirements] = {}
    has_reservation: set[JobName] = set()
    for task in pending:
        job = ctrl.state.get_job(task.job_id)
        if job and not job.is_finished():
            jobs[task.job_id] = job_requirements_from_job(job)
            if job.request.HasField("reservation"):
                has_reservation.add(task.job_id)
            elif _find_reservation_ancestor(ctrl.state, task.job_id) is not None:
                has_reservation.add(task.job_id)

    # Both grandchildren inherit taint exemption
    assert gc_a_jid in has_reservation
    assert gc_b_jid in has_reservation

    # Neither gets NOT_EXISTS constraint
    modified_jobs = _inject_taint_constraints(jobs, has_reservation)
    for gc_jid in [gc_a_jid, gc_b_jid]:
        gc_constraints = modified_jobs[gc_jid].constraints
        not_exists = [c for c in gc_constraints if c.key == RESERVATION_TAINT_KEY]
        assert len(not_exists) == 0

    # Unrelated job DOES get NOT_EXISTS constraint
    unrelated_jid = JobName.root("test-user", "unrelated")
    unrelated_req = JobRequirements(
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_gpu_device("H100"),
        ),
        constraints=[],
        is_coscheduled=False,
        coscheduling_group_by=None,
    )
    jobs[unrelated_jid] = unrelated_req
    modified_jobs = _inject_taint_constraints(jobs, has_reservation)
    unrelated_constraints = modified_jobs[unrelated_jid].constraints
    not_exists = [c for c in unrelated_constraints if c.key == RESERVATION_TAINT_KEY]
    assert len(not_exists) == 1
    assert not_exists[0].op == cluster_pb2.CONSTRAINT_OP_NOT_EXISTS


def test_unrelated_job_blocked_when_all_workers_claimed():
    """A job with no reservation ancestor gets NOT_EXISTS and is blocked from claimed workers."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "w1", _gpu_metadata("H100"))
    _register_worker(ctrl.state, "w2", _gpu_metadata("H100"))

    parent_req = _make_job_request_with_reservation(
        reservation_entries=[
            _make_reservation_entry(_gpu_device("H100")),
            _make_reservation_entry(_gpu_device("H100")),
        ],
    )
    _submit_job(ctrl.state, "res-parent", parent_req)
    ctrl._claim_workers_for_reservations()
    assert len(ctrl.reservation_claims) == 2

    # Unrelated job requesting GPU
    unrelated_jid = JobName.root("test-user", "unrelated")
    unrelated_req = JobRequirements(
        resources=cluster_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_gpu_device("H100"),
        ),
        constraints=[],
        is_coscheduled=False,
        coscheduling_group_by=None,
    )

    jobs = {unrelated_jid: unrelated_req}
    has_reservation: set[JobName] = set()

    modified_jobs = _inject_taint_constraints(jobs, has_reservation)
    constraints = modified_jobs[unrelated_jid].constraints
    not_exists = [c for c in constraints if c.key == RESERVATION_TAINT_KEY]
    assert len(not_exists) == 1
    assert not_exists[0].op == cluster_pb2.CONSTRAINT_OP_NOT_EXISTS


# =============================================================================
# _worker_matches_reservation_entry with auto-injected constraints
# =============================================================================


def test_reservation_match_auto_injects_device_constraints():
    """Reservation entry with GPU device auto-generates device constraints."""
    worker = _make_worker("w1", _gpu_metadata("H100"))
    entry = _make_reservation_entry(_gpu_device("H100"))
    assert _worker_matches_reservation_entry(worker, entry)


def test_reservation_match_user_variant_override():
    """Explicit multi-variant constraint on entry overrides auto-generated single variant."""
    worker = _make_worker("w1", _gpu_metadata("A100"))
    user_constraint = device_variant_constraint(["A100", "H100"]).to_proto()
    entry = _make_reservation_entry(_gpu_device("H100"), constraints=[user_constraint])
    # Worker is A100, entry device is H100, but explicit constraint allows A100
    assert _worker_matches_reservation_entry(worker, entry)
