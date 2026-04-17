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

from iris.cluster.controller.codec import constraints_from_json
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
from iris.cluster.controller.scheduler import JobRequirements, Scheduler, SchedulingContext

from iris.cluster.controller.store import task_row_can_be_scheduled
from iris.cluster.controller.schema import WorkerRow
from iris.cluster.controller.transitions import (
    HEARTBEAT_FAILURE_THRESHOLD,
    RESERVATION_HOLDER_JOB_NAME,
    Assignment,
    ControllerTransitions,
    HeartbeatApplyRequest,
    TaskUpdate,
)
from iris.cluster.constraints import (
    AttributeValue,
    Constraint,
    ConstraintOp,
    WellKnownAttribute,
    device_variant_constraint,
    get_device_type,
    get_device_variant,
)
from iris.cluster.types import JobName, WorkerId, is_job_finished
from iris.rpc import job_pb2
from iris.rpc import controller_pb2
from rigging.timing import Timestamp
from tests.cluster.controller._testing import (
    set_task_state as _set_task_state,
    set_worker_attribute as _set_worker_attribute,
    set_worker_health as _set_worker_health,
)
from tests.cluster.controller.conftest import (
    FakeProvider,
    hydrate_worker_attributes as _with_attrs,
    query_job as _query_job,
    query_job_row as _query_job_row,
    query_task as _query_task,
    query_task_with_attempts as _query_task_with_attempts,
    query_tasks_for_job as _query_tasks_for_job,
    query_worker as _query_worker,
    schedulable_tasks as _schedulable_tasks,
    worker_running_tasks as _worker_running_tasks,
)


def _cpu_device() -> job_pb2.DeviceConfig:
    return job_pb2.DeviceConfig(cpu=job_pb2.CpuDevice(variant="cpu"))


def _gpu_device(variant: str = "H100", count: int = 8) -> job_pb2.DeviceConfig:
    return job_pb2.DeviceConfig(gpu=job_pb2.GpuDevice(variant=variant, count=count))


def _cpu_metadata() -> job_pb2.WorkerMetadata:
    meta = job_pb2.WorkerMetadata(
        hostname="test",
        ip_address="127.0.0.1",
        cpu_count=8,
        memory_bytes=16 * 1024**3,
        disk_bytes=100 * 1024**3,
        device=_cpu_device(),
    )
    meta.attributes[WellKnownAttribute.DEVICE_TYPE].CopyFrom(job_pb2.AttributeValue(string_value="cpu"))
    return meta


def _gpu_metadata(variant: str = "H100") -> job_pb2.WorkerMetadata:
    meta = job_pb2.WorkerMetadata(
        hostname="test-gpu",
        ip_address="127.0.0.1",
        cpu_count=32,
        memory_bytes=256 * 1024**3,
        disk_bytes=500 * 1024**3,
        device=_gpu_device(variant),
    )
    meta.attributes[WellKnownAttribute.DEVICE_TYPE].CopyFrom(job_pb2.AttributeValue(string_value="gpu"))
    meta.attributes[WellKnownAttribute.DEVICE_VARIANT].CopyFrom(job_pb2.AttributeValue(string_value=variant.lower()))
    return meta


def _default_attributes_for_device(device: job_pb2.DeviceConfig) -> dict[str, AttributeValue]:
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
    metadata: job_pb2.WorkerMetadata | None = None,
    attributes: dict[str, AttributeValue] | None = None,
    healthy: bool = True,
) -> WorkerRow:
    meta = metadata or _cpu_metadata()
    # Workers always have device attributes from config (Stage 3).
    # Merge explicit attributes on top of the device-derived defaults.
    default_attrs = _default_attributes_for_device(meta.device)
    if attributes:
        default_attrs.update(attributes)
    dt = get_device_type(meta.device)
    dv = get_device_variant(meta.device) or ""
    total_cpu = meta.cpu_count * 1000
    total_mem = meta.memory_bytes
    total_gpu = meta.gpu_count
    total_tpu = 1 if meta.tpu_name else 0
    return WorkerRow(
        worker_id=WorkerId(worker_id),
        address=f"{worker_id}:8080",
        healthy=healthy,
        active=True,
        consecutive_failures=0,
        last_heartbeat=Timestamp.now(),
        committed_cpu_millicores=0,
        committed_mem=0,
        committed_gpu=0,
        committed_tpu=0,
        total_cpu_millicores=total_cpu,
        total_memory_bytes=total_mem,
        total_gpu_count=total_gpu,
        total_tpu_count=total_tpu,
        device_type=dt,
        device_variant=dv,
        attributes=default_attrs,
        available_cpu_millicores=total_cpu,
        available_memory=total_mem,
        available_gpus=total_gpu,
        available_tpus=total_tpu,
    )


def _make_reservation_entry(
    device: job_pb2.DeviceConfig | None = None,
    constraints: list[job_pb2.Constraint] | None = None,
) -> job_pb2.ReservationEntry:
    dev = device or _cpu_device()
    return job_pb2.ReservationEntry(
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=dev,
        ),
        constraints=constraints or [],
    )


def _entrypoint() -> job_pb2.RuntimeEntrypoint:
    entrypoint = job_pb2.RuntimeEntrypoint()
    entrypoint.run_command.argv[:] = ["python", "-c", "pass"]
    return entrypoint


def _make_job_request_with_reservation(
    name: str = "res-job",
    reservation_entries: list[job_pb2.ReservationEntry] | None = None,
) -> controller_pb2.Controller.LaunchJobRequest:
    req = controller_pb2.Controller.LaunchJobRequest(
        name=name,
        entrypoint=_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    if reservation_entries:
        for entry in reservation_entries:
            req.reservation.entries.append(entry)
    return req


def _make_controller() -> Controller:
    """Create a Controller with minimal config for unit testing."""
    config = ControllerConfig(
        remote_state_dir="file:///tmp/iris-test-bundles",
    )
    return Controller(config=config, provider=FakeProvider())


def _register_worker(
    state: ControllerTransitions,
    worker_id: str,
    metadata: job_pb2.WorkerMetadata | None = None,
) -> WorkerId:
    wid = WorkerId(worker_id)
    state.register_or_refresh_worker(
        worker_id=wid,
        address=f"{worker_id}:8080",
        metadata=metadata or _cpu_metadata(),
        ts=Timestamp.now(),
    )
    return wid


def _submit_job(
    state: ControllerTransitions,
    job_id: str,
    request: controller_pb2.Controller.LaunchJobRequest,
) -> JobName:
    jid = JobName.root("test-user", job_id)
    request.name = jid.to_wire()
    state.submit_job(jid, request, Timestamp.now())
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
    constraint = job_pb2.Constraint(
        key=WellKnownAttribute.REGION,
        op=job_pb2.CONSTRAINT_OP_EQ,
        value=job_pb2.AttributeValue(string_value="us-central1"),
    )
    entry = _make_reservation_entry(_cpu_device(), constraints=[constraint])
    assert _worker_matches_reservation_entry(worker, entry)


def test_worker_rejects_unmet_constraint():
    """Worker without the required attribute fails the constraint check."""
    worker = _make_worker("w1", _cpu_metadata())
    constraint = job_pb2.Constraint(
        key=WellKnownAttribute.REGION,
        op=job_pb2.CONSTRAINT_OP_EQ,
        value=job_pb2.AttributeValue(string_value="us-central1"),
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
    _set_worker_health(ctrl.state._db, WorkerId("w1"), healthy=False)

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
    claims = dict(ctrl.reservation_claims)
    claims[WorkerId("dead-worker")] = ReservationClaim(
        job_id=JobName.root("test-user", "j1").to_wire(),
        entry_idx=99,
    )
    ctrl.state.replace_reservation_claims(claims)
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

    # Kill the job to mark it as finished.
    ctrl.state.cancel_job(jid, reason="test")

    job = _query_job(ctrl.state, jid)
    assert is_job_finished(job.state)

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

    job = _query_job(ctrl.state, jid)
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

    job = _query_job(ctrl.state, jid)
    # Only 1 worker available for 2 entries
    assert not ctrl._is_reservation_satisfied(job)


def test_gate_satisfied_for_jobs_without_reservation():
    """Jobs without a reservation always pass the gate."""
    ctrl = _make_controller()
    req = controller_pb2.Controller.LaunchJobRequest(
        name="no-res",
        entrypoint=_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    jid = _submit_job(ctrl.state, "no-res", req)

    job = _query_job(ctrl.state, jid)
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
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
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
    assert not_exists[0].op == ConstraintOp.NOT_EXISTS


def test_taint_constraint_not_added_to_reservation_jobs():
    """Direct reservation jobs get an EQ constraint forcing them onto claimed workers."""
    res_job = JobName.root("test-user", "reserved")
    jobs = {
        res_job: _make_job_requirements(),
    }
    has_reservation = {res_job}
    has_direct_reservation = {res_job}

    result = _inject_taint_constraints(jobs, has_reservation, has_direct_reservation)

    constraints = result[res_job].constraints
    eq = [c for c in constraints if c.key == RESERVATION_TAINT_KEY]
    assert len(eq) == 1
    assert eq[0].op == ConstraintOp.EQ
    assert eq[0].values[0].value == res_job.to_wire()


def test_taint_constraint_mixed_jobs():
    """Direct reservation gets EQ, descendant gets nothing, regular gets NOT_EXISTS."""
    res_job = JobName.root("test-user", "reserved")
    descendant_job = JobName.from_string("/test-user/reserved/child")
    reg_job = JobName.root("test-user", "regular")
    jobs = {
        res_job: _make_job_requirements(),
        descendant_job: _make_job_requirements(),
        reg_job: _make_job_requirements(),
    }
    has_reservation = {res_job, descendant_job}
    has_direct_reservation = {res_job}

    result = _inject_taint_constraints(jobs, has_reservation, has_direct_reservation)

    # Direct reservation job: EQ constraint
    res_constraints = [c for c in result[res_job].constraints if c.key == RESERVATION_TAINT_KEY]
    assert len(res_constraints) == 1
    assert res_constraints[0].op == ConstraintOp.EQ
    assert res_constraints[0].values[0].value == res_job.to_wire()

    # Descendant: no taint constraint
    desc_constraints = [c for c in result[descendant_job].constraints if c.key == RESERVATION_TAINT_KEY]
    assert len(desc_constraints) == 0

    # Regular job: NOT_EXISTS constraint
    reg_constraints = [c for c in result[reg_job].constraints if c.key == RESERVATION_TAINT_KEY]
    assert len(reg_constraints) == 1
    assert reg_constraints[0].op == ConstraintOp.NOT_EXISTS


def test_taint_constraint_preserves_existing_constraints():
    """Existing constraints are preserved when the taint constraint is added."""
    existing = Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.EQ, value="us-central1")
    jobs = {
        JobName.root("test-user", "regular"): JobRequirements(
            resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            constraints=[existing],
            is_coscheduled=False,
            coscheduling_group_by=None,
        ),
    }
    has_reservation: set[JobName] = set()
    has_direct_reservation: set[JobName] = set()

    result = _inject_taint_constraints(jobs, has_reservation, has_direct_reservation)

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
    workers: list[WorkerRow],
    pending_tasks: list[JobName],
    jobs: dict[JobName, JobRequirements],
) -> SchedulingContext:
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
        resources=job_pb2.ResourceSpecProto(cpu_millicores=999_000, memory_bytes=1024**3),
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
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
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
        resources=job_pb2.ResourceSpecProto(cpu_millicores=4000, memory_bytes=1024**3),
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
    _set_worker_attribute(ctrl.state._db, w1, WellKnownAttribute.REGION, AttributeValue("us-central1"))

    req = _make_job_request_with_reservation(reservation_entries=[_make_reservation_entry()])
    jid = _submit_job(ctrl.state, "j1", req)
    ctrl._claim_workers_for_reservations()

    result = _reservation_region_constraints(
        jid.to_wire(),
        ctrl.reservation_claims,
        ctrl._stores,
        [],
    )

    assert len(result) == 1
    assert result[0].key == WellKnownAttribute.REGION
    assert result[0].op == ConstraintOp.EQ
    assert result[0].values[0].value == "us-central1"


def test_region_constraint_not_injected_when_already_present():
    """Existing region constraint prevents injection."""
    ctrl = _make_controller()
    w1 = _register_worker(ctrl.state, "w1")
    _set_worker_attribute(ctrl.state._db, w1, WellKnownAttribute.REGION, AttributeValue("us-central1"))

    req = _make_job_request_with_reservation(reservation_entries=[_make_reservation_entry()])
    jid = _submit_job(ctrl.state, "j1", req)
    ctrl._claim_workers_for_reservations()

    existing = Constraint.create(key=WellKnownAttribute.REGION, op=ConstraintOp.EQ, value="us-east1")
    result = _reservation_region_constraints(
        jid.to_wire(),
        ctrl.reservation_claims,
        ctrl._stores,
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
        ctrl._stores,
        [],
    )

    assert result == []


def test_region_constraint_multiple_regions():
    """IN constraint injected when claimed workers span multiple regions."""
    ctrl = _make_controller()
    w1 = _register_worker(ctrl.state, "w1")
    w2 = _register_worker(ctrl.state, "w2")
    _set_worker_attribute(ctrl.state._db, w1, WellKnownAttribute.REGION, AttributeValue("us-central1"))
    _set_worker_attribute(ctrl.state._db, w2, WellKnownAttribute.REGION, AttributeValue("us-east1"))

    req = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry(), _make_reservation_entry()],
    )
    jid = _submit_job(ctrl.state, "j1", req)
    ctrl._claim_workers_for_reservations()

    result = _reservation_region_constraints(
        jid.to_wire(),
        ctrl.reservation_claims,
        ctrl._stores,
        [],
    )

    assert len(result) == 1
    assert result[0].key == WellKnownAttribute.REGION
    assert result[0].op == ConstraintOp.IN
    assert {v.value for v in result[0].values} == {"us-central1", "us-east1"}


def test_no_injection_for_non_reservation_job():
    """No claims for this job → constraints returned unchanged."""
    ctrl = _make_controller()
    w1 = _register_worker(ctrl.state, "w1")
    _set_worker_attribute(ctrl.state._db, w1, WellKnownAttribute.REGION, AttributeValue("us-central1"))

    # Claim w1 for a different job
    req = _make_job_request_with_reservation(reservation_entries=[_make_reservation_entry()])
    _submit_job(ctrl.state, "other-job", req)
    ctrl._claim_workers_for_reservations()

    result = _reservation_region_constraints(
        "/test-user/unrelated-job",
        ctrl.reservation_claims,
        ctrl._stores,
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
    child_req = controller_pb2.Controller.LaunchJobRequest(
        name=child_jid.to_wire(),
        entrypoint=_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    ctrl.state.submit_job(child_jid, child_req, Timestamp.now())

    result = _find_reservation_ancestor(ctrl._db, child_jid)
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
    parent_req = controller_pb2.Controller.LaunchJobRequest(
        name=parent_jid.to_wire(),
        entrypoint=_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    ctrl.state.submit_job(parent_jid, parent_req, Timestamp.now())

    # Grandchild
    gc_jid = JobName.from_string("/test-user/gp/parent/gc")
    gc_req = controller_pb2.Controller.LaunchJobRequest(
        name=gc_jid.to_wire(),
        entrypoint=_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    ctrl.state.submit_job(gc_jid, gc_req, Timestamp.now())

    result = _find_reservation_ancestor(ctrl._db, gc_jid)
    assert result == gp_jid


def test_find_reservation_ancestor_returns_none_for_root_job():
    """Root job with no reservation returns None."""
    ctrl = _make_controller()
    req = controller_pb2.Controller.LaunchJobRequest(
        name="no-res",
        entrypoint=_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    jid = _submit_job(ctrl.state, "no-res", req)
    assert _find_reservation_ancestor(ctrl._db, jid) is None


def test_find_reservation_ancestor_returns_none_when_no_ancestor_has_reservation():
    """Child of a non-reservation parent returns None."""
    ctrl = _make_controller()
    parent_req = controller_pb2.Controller.LaunchJobRequest(
        name="plain-parent",
        entrypoint=_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    _submit_job(ctrl.state, "plain-parent", parent_req)

    child_jid = JobName.from_string("/test-user/plain-parent/child")
    child_req = controller_pb2.Controller.LaunchJobRequest(
        name=child_jid.to_wire(),
        entrypoint=_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    ctrl.state.submit_job(child_jid, child_req, Timestamp.now())

    assert _find_reservation_ancestor(ctrl._db, child_jid) is None


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
    child_req = controller_pb2.Controller.LaunchJobRequest(
        name=child_jid.to_wire(),
        entrypoint=_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_gpu_device("H100"),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    ctrl.state.submit_job(child_jid, child_req, Timestamp.now())

    # Build scheduling state — child should be in has_reservation
    pending = _schedulable_tasks(ctrl.state)
    jobs: dict[JobName, JobRequirements] = {}
    has_reservation: set[JobName] = set()
    for task in pending:
        job_row = _query_job_row(ctrl.state, task.job_id)
        job_detail = _query_job(ctrl.state, task.job_id)
        if job_row and not is_job_finished(job_row.state):
            jobs[task.job_id] = job_requirements_from_job(job_row)
            if job_detail and job_detail.reservation_json is not None:
                has_reservation.add(task.job_id)
            elif _find_reservation_ancestor(ctrl._db, task.job_id) is not None:
                has_reservation.add(task.job_id)

    assert child_jid in has_reservation

    # Track direct reservations
    has_direct_reservation: set[JobName] = set()
    for task in pending:
        job_detail = _query_job(ctrl.state, task.job_id)
        if job_detail and not is_job_finished(job_detail.state) and job_detail.reservation_json is not None:
            has_direct_reservation.add(task.job_id)

    # Child does NOT get NOT_EXISTS constraint (descendant, no constraint at all)
    modified_jobs = _inject_taint_constraints(jobs, has_reservation, has_direct_reservation)
    child_constraints = modified_jobs[child_jid].constraints
    taint = [c for c in child_constraints if c.key == RESERVATION_TAINT_KEY]
    assert len(taint) == 0

    # Parent (direct reservation) gets EQ constraint
    parent_jid = JobName.root("test-user", "res-parent")
    if parent_jid in modified_jobs:
        parent_constraints = [c for c in modified_jobs[parent_jid].constraints if c.key == RESERVATION_TAINT_KEY]
        assert len(parent_constraints) == 1
        assert parent_constraints[0].op == job_pb2.CONSTRAINT_OP_EQ


def test_grandchildren_inherit_reservation_from_ancestor():
    """Grandchildren of a reservation job inherit taint exemption."""
    ctrl = _make_controller()
    _register_worker(ctrl.state, "h1", _gpu_metadata("H100"))
    _register_worker(ctrl.state, "h2", _gpu_metadata("H100"))
    _register_worker(ctrl.state, "a1", _gpu_metadata("A100"))
    _register_worker(ctrl.state, "a2", _gpu_metadata("A100"))

    # Root job (CPU, no reservation)
    root_req = controller_pb2.Controller.LaunchJobRequest(
        name="root",
        entrypoint=_entrypoint(),
        resources=job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
        environment=job_pb2.EnvironmentConfig(),
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
    ctrl.state.submit_job(child_a_jid, child_a_req, Timestamp.now())

    # Child-B reserves 2 A100
    child_b_jid = JobName.from_string("/test-user/root/child-b")
    child_b_req = _make_job_request_with_reservation(
        reservation_entries=[
            _make_reservation_entry(_gpu_device("A100")),
            _make_reservation_entry(_gpu_device("A100")),
        ],
    )
    child_b_req.name = child_b_jid.to_wire()
    ctrl.state.submit_job(child_b_jid, child_b_req, Timestamp.now())

    ctrl._claim_workers_for_reservations()
    assert len(ctrl.reservation_claims) == 4

    # Grandchild-A (under child-A) requesting H100
    gc_a_jid = JobName.from_string("/test-user/root/child-a/gc-a")
    gc_a_req = controller_pb2.Controller.LaunchJobRequest(
        name=gc_a_jid.to_wire(),
        entrypoint=_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_gpu_device("H100"),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    ctrl.state.submit_job(gc_a_jid, gc_a_req, Timestamp.now())

    # Grandchild-B (under child-B) requesting A100
    gc_b_jid = JobName.from_string("/test-user/root/child-b/gc-b")
    gc_b_req = controller_pb2.Controller.LaunchJobRequest(
        name=gc_b_jid.to_wire(),
        entrypoint=_entrypoint(),
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_gpu_device("A100"),
        ),
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
    )
    ctrl.state.submit_job(gc_b_jid, gc_b_req, Timestamp.now())

    # Build scheduling state
    pending = _schedulable_tasks(ctrl.state)
    jobs: dict[JobName, JobRequirements] = {}
    has_reservation: set[JobName] = set()
    for task in pending:
        job_row = _query_job_row(ctrl.state, task.job_id)
        job_detail = _query_job(ctrl.state, task.job_id)
        if job_row and not is_job_finished(job_row.state):
            jobs[task.job_id] = job_requirements_from_job(job_row)
            if job_detail and job_detail.reservation_json is not None:
                has_reservation.add(task.job_id)
            elif _find_reservation_ancestor(ctrl._db, task.job_id) is not None:
                has_reservation.add(task.job_id)

    # Both grandchildren inherit taint exemption
    assert gc_a_jid in has_reservation
    assert gc_b_jid in has_reservation

    # Track direct reservations
    has_direct_reservation: set[JobName] = set()
    for task in pending:
        job_detail = _query_job(ctrl.state, task.job_id)
        if job_detail and not is_job_finished(job_detail.state) and job_detail.reservation_json is not None:
            has_direct_reservation.add(task.job_id)

    # Neither grandchild gets any taint constraint (descendants)
    modified_jobs = _inject_taint_constraints(jobs, has_reservation, has_direct_reservation)
    for gc_jid in [gc_a_jid, gc_b_jid]:
        gc_constraints = modified_jobs[gc_jid].constraints
        taint = [c for c in gc_constraints if c.key == RESERVATION_TAINT_KEY]
        assert len(taint) == 0

    # Direct reservation jobs get EQ constraint
    for direct_jid in [child_a_jid, child_b_jid]:
        if direct_jid in modified_jobs:
            direct_constraints = [c for c in modified_jobs[direct_jid].constraints if c.key == RESERVATION_TAINT_KEY]
            assert len(direct_constraints) == 1
            assert direct_constraints[0].op == job_pb2.CONSTRAINT_OP_EQ

    # Unrelated job DOES get NOT_EXISTS constraint
    unrelated_jid = JobName.root("test-user", "unrelated")
    unrelated_req = JobRequirements(
        resources=job_pb2.ResourceSpecProto(
            cpu_millicores=1000,
            memory_bytes=1024**3,
            device=_gpu_device("H100"),
        ),
        constraints=[],
        is_coscheduled=False,
        coscheduling_group_by=None,
    )
    jobs[unrelated_jid] = unrelated_req
    modified_jobs = _inject_taint_constraints(jobs, has_reservation, has_direct_reservation)
    unrelated_constraints = modified_jobs[unrelated_jid].constraints
    not_exists = [c for c in unrelated_constraints if c.key == RESERVATION_TAINT_KEY]
    assert len(not_exists) == 1
    assert not_exists[0].op == job_pb2.CONSTRAINT_OP_NOT_EXISTS


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
        resources=job_pb2.ResourceSpecProto(
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
    assert not_exists[0].op == job_pb2.CONSTRAINT_OP_NOT_EXISTS


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


# =============================================================================
# Holder task worker-death handling
# =============================================================================


def test_holder_task_worker_death_no_failure_record(state):
    """Holder tasks return to PENDING with no WORKER_FAILED record when their worker dies.

    Holder tasks are virtual (never dispatched). When a worker dies, they must
    be silently requeued — no attempt accumulation, no retry budget burn — so
    that they can survive an arbitrary number of worker cycles without leaking
    memory or eventually going terminal.
    """
    request = _make_job_request_with_reservation(reservation_entries=[_make_reservation_entry(_cpu_device())])
    parent_job_id = _submit_job(state, "res-job", request)
    holder_job_id = parent_job_id.child(RESERVATION_HOLDER_JOB_NAME)

    holder_tasks = _query_tasks_for_job(state, holder_job_id)
    assert len(holder_tasks) == 1
    holder_task = holder_tasks[0]

    # Simulate multiple worker-death cycles to confirm no accumulation.
    for cycle in range(3):
        worker_id = _register_worker(state, f"worker-{cycle}")

        # Assign the holder task to the worker (mimics what the scheduler does).
        state.queue_assignments([Assignment(task_id=holder_task.task_id, worker_id=worker_id)])
        current_holder = _query_task_with_attempts(state, holder_task.task_id)
        assert current_holder is not None
        assert current_holder.state == job_pb2.TASK_STATE_ASSIGNED
        # active_worker_id: non-None when state is not PENDING
        assert current_holder.state != job_pb2.TASK_STATE_PENDING
        current_attempt = current_holder.attempts[-1] if current_holder.attempts else None
        active_wid = current_attempt.worker_id if current_attempt is not None else current_holder.current_worker_id
        assert active_wid == worker_id

        # Kill the worker — holder task must NOT go through WORKER_FAILED.
        batch = state.drain_dispatch(worker_id)
        assert batch is not None
        for _ in range(HEARTBEAT_FAILURE_THRESHOLD):
            state.record_heartbeat_failure(worker_id, "simulated crash", batch)

        holder_task = _query_task_with_attempts(state, holder_task.task_id)
        assert holder_task is not None
        current_state = _query_task(state, holder_task.task_id).state
        assert (
            current_state == job_pb2.TASK_STATE_PENDING
        ), f"cycle {cycle}: expected PENDING, got {job_pb2.TaskState.Name(current_state)}"
        assert holder_task.preemption_count == 0, f"cycle {cycle}: preemption_count leaked"
        assert holder_task.failure_count == 0, f"cycle {cycle}: failure_count leaked"
        assert (
            len(holder_task.attempts) == 0
        ), f"cycle {cycle}: attempt list leaked ({len(holder_task.attempts)} entries)"
        # active_worker_id should be None when PENDING
        assert holder_task.state == job_pb2.TASK_STATE_PENDING, "no active worker after death"
        assert task_row_can_be_scheduled(holder_task), "holder task must be schedulable again"


def test_holder_task_removed_from_worker_when_parent_succeeds(state):
    """Holder task is cleaned from worker.running_tasks when the parent job succeeds.

    PATH A (task-driven termination): a parent task succeeds → on_task_transition
    returns JOB_STATE_SUCCEEDED → _finalize_job_state → _cancel_child_jobs →
    _on_job_cancelled(holder) → _cleanup_task_resources removes the holder task
    from the worker's running_tasks set.

    Previously untested; the existing cancel test only covers PATH B
    (explicit JobCancelledEvent), not this completion-driven path.
    """
    request = _make_job_request_with_reservation(reservation_entries=[_make_reservation_entry(_cpu_device())])
    parent_job_id = _submit_job(state, "res-job", request)
    holder_job_id = parent_job_id.child(RESERVATION_HOLDER_JOB_NAME)

    holder_tasks = _query_tasks_for_job(state, holder_job_id)
    parent_tasks = _query_tasks_for_job(state, parent_job_id)
    assert len(holder_tasks) == 1
    assert len(parent_tasks) == 1

    holder_task = holder_tasks[0]
    parent_task = parent_tasks[0]

    wid_holder = _register_worker(state, "worker-holder")
    wid_parent = _register_worker(state, "worker-parent")

    state.queue_assignments([Assignment(task_id=holder_task.task_id, worker_id=wid_holder)])
    state.queue_assignments([Assignment(task_id=parent_task.task_id, worker_id=wid_parent)])

    assert holder_task.task_id in _worker_running_tasks(state, wid_holder)

    # Parent task succeeds → _finalize_job_state(SUCCEEDED) → _cancel_child_jobs
    # → holder task killed → running_tasks entry discarded.
    parent_task = _query_task_with_attempts(state, parent_task.task_id)
    state.apply_task_updates(
        HeartbeatApplyRequest(
            worker_id=wid_parent,
            worker_resource_snapshot=None,
            updates=[
                TaskUpdate(
                    task_id=parent_task.task_id,
                    attempt_id=parent_task.current_attempt_id,
                    new_state=job_pb2.TASK_STATE_SUCCEEDED,
                )
            ],
        )
    )

    holder_task = _query_task(state, holder_task.task_id)
    assert holder_task is not None
    holder_state = _query_task(state, holder_task.task_id).state
    assert holder_state == job_pb2.TASK_STATE_KILLED, (
        "expected holder task KILLED after parent success, " f"got {job_pb2.TaskState.Name(holder_state)}"
    )
    assert holder_task.task_id not in _worker_running_tasks(state, wid_holder)


def test_holder_task_removed_from_worker_when_parent_cancelled_all_tasks_already_terminal(state):
    """Holder is cleaned even when JobCancelledEvent arrives after all parent tasks finished.

    The gap: _on_job_cancelled's task loop skips all terminal tasks, so
    _on_task_state_changed is never invoked, _finalize_job_state is never
    reached, and _cancel_child_jobs is never called on the normal cascade path.
    Without the explicit _cancel_child_jobs call at the end of _on_job_cancelled,
    the holder task would stay in worker.running_tasks indefinitely, making the
    worker appear busy and blocking scale-down idle detection.
    """
    request = _make_job_request_with_reservation(reservation_entries=[_make_reservation_entry(_cpu_device())])
    parent_job_id = _submit_job(state, "res-job", request)
    holder_job_id = parent_job_id.child(RESERVATION_HOLDER_JOB_NAME)

    holder_tasks = _query_tasks_for_job(state, holder_job_id)
    parent_tasks = _query_tasks_for_job(state, parent_job_id)
    holder_task = holder_tasks[0]
    parent_task = parent_tasks[0]

    wid = _register_worker(state, "worker")
    state.queue_assignments([Assignment(task_id=holder_task.task_id, worker_id=wid)])

    assert holder_task.task_id in _worker_running_tasks(state, wid)

    # Directly mark the parent task terminal WITHOUT going through the event
    # system. This simulates the race: the parent task already finished (and
    # _finalize_job_state already ran, cleaning the holder), but then the job
    # is submitted to cancel anyway. More importantly it lets us verify that
    # _on_job_cancelled handles the case where all parent tasks are terminal
    # (loop body never executes, so the old code never reached _cancel_child_jobs).
    parent_task_ref = _query_task(state, parent_task.task_id)
    assert parent_task_ref is not None
    _set_task_state(state._db, parent_task.task_id, job_pb2.TASK_STATE_KILLED)

    # Fire JobCancelledEvent. All parent tasks are now terminal so the loop
    # skips them. Only the explicit _cancel_child_jobs call at the end of
    # _on_job_cancelled can clean up the holder.
    state.cancel_job(parent_job_id, reason="manual cancel")

    holder_task = _query_task(state, holder_task.task_id)
    assert holder_task is not None
    assert (
        _query_task(state, holder_task.task_id).state == job_pb2.TASK_STATE_KILLED
    ), f"expected holder task KILLED, got {job_pb2.TaskState.Name(_query_task(state, holder_task.task_id).state)}"
    assert holder_task.task_id not in _worker_running_tasks(state, wid), (
        "holder task must be removed from worker.running_tasks; " "stale entry would block scale-down idle detection"
    )


# =============================================================================
# Holder task device-type constraint injection
# =============================================================================


def _tpu_device(variant: str = "v5p-64", count: int = 4) -> job_pb2.DeviceConfig:
    return job_pb2.DeviceConfig(tpu=job_pb2.TpuDevice(variant=variant, count=count))


def _tpu_metadata(variant: str = "v5p-64", region: str | None = None) -> job_pb2.WorkerMetadata:
    meta = job_pb2.WorkerMetadata(
        hostname="test-tpu",
        ip_address="127.0.0.1",
        cpu_count=32,
        memory_bytes=64 * 1024**3,
        disk_bytes=500 * 1024**3,
        device=_tpu_device(variant),
    )
    meta.attributes[WellKnownAttribute.DEVICE_TYPE].CopyFrom(job_pb2.AttributeValue(string_value="tpu"))
    meta.attributes[WellKnownAttribute.DEVICE_VARIANT].CopyFrom(job_pb2.AttributeValue(string_value=variant.lower()))
    if region:
        meta.attributes["region"].CopyFrom(job_pb2.AttributeValue(string_value=region))
    return meta


def _region_constraint(region: str) -> job_pb2.Constraint:
    """Create a region=<value> constraint proto."""
    return Constraint.create(key="region", op=ConstraintOp.EQ, value=region).to_proto()


def test_holder_task_gets_device_constraints_from_tpu_entry(state):
    """Holder task for a TPU reservation entry must have device-type constraints.

    When an entry has explicit constraints (e.g. region) but no device-type,
    the holder job must still get auto-injected device constraints from the
    entry's resource spec. Without this, the holder could land on a CPU worker.
    """

    # Entry has TPU resources + region constraint, but NO device-type constraint.
    entry = _make_reservation_entry(
        device=_tpu_device("v5p-64", count=4),
        constraints=[_region_constraint("us-central2")],
    )
    request = _make_job_request_with_reservation(reservation_entries=[entry])
    parent_job_id = _submit_job(state, "res-job", request)
    holder_job_id = parent_job_id.child(RESERVATION_HOLDER_JOB_NAME)

    holder_job = _query_job(state, holder_job_id)
    assert holder_job is not None
    constraint_keys = [c.key for c in constraints_from_json(holder_job.constraints_json)]

    assert (
        WellKnownAttribute.DEVICE_TYPE in constraint_keys
    ), f"holder job missing device-type constraint; got keys: {constraint_keys}"
    assert (
        WellKnownAttribute.DEVICE_VARIANT in constraint_keys
    ), f"holder job missing device-variant constraint; got keys: {constraint_keys}"
    assert "region" in constraint_keys, "holder job should still have the explicit region constraint"


def test_holder_task_not_scheduled_on_wrong_device_type(state):
    """Holder task for a TPU entry must not be assigned to a CPU worker.

    End-to-end test: submit a reservation with a TPU entry that has only a
    region constraint (no device-type), register both a TPU and CPU worker,
    and verify the scheduler assigns the holder to the TPU worker only.
    """

    entry = _make_reservation_entry(
        device=_tpu_device("v5p-64", count=4),
        constraints=[_region_constraint("us-central2")],
    )
    request = _make_job_request_with_reservation(reservation_entries=[entry])
    parent_job_id = _submit_job(state, "res-job", request)
    holder_job_id = parent_job_id.child(RESERVATION_HOLDER_JOB_NAME)

    # Register a CPU worker and a TPU worker, both in the same region.
    cpu_meta = _cpu_metadata()
    cpu_meta.attributes["region"].CopyFrom(job_pb2.AttributeValue(string_value="us-central2"))
    cpu_wid = _register_worker(state, "cpu-worker", metadata=cpu_meta)
    tpu_wid = _register_worker(state, "tpu-worker", metadata=_tpu_metadata("v5p-64", region="us-central2"))

    holder_tasks = _query_tasks_for_job(state, holder_job_id)
    assert len(holder_tasks) == 1
    holder_task = holder_tasks[0]

    # Build scheduling context with both workers and run find_assignments.
    cpu_worker = _query_worker(state, cpu_wid)
    tpu_worker = _query_worker(state, tpu_wid)

    holder_job = _query_job_row(state, holder_job_id)
    assert holder_job is not None
    holder_req = job_requirements_from_job(holder_job)
    context = _build_context_with_workers(
        _with_attrs(state, [cpu_worker, tpu_worker]),
        pending_tasks=[holder_task.task_id],
        jobs={holder_job_id: holder_req},
    )
    scheduler = Scheduler()
    result = scheduler.find_assignments(context)

    assigned_workers = [wid for _, wid in result.assignments]
    assert (
        WorkerId("tpu-worker") in assigned_workers
    ), f"holder task should be assigned to TPU worker, got: {assigned_workers}"
    assert WorkerId("cpu-worker") not in assigned_workers, "holder task must NOT land on CPU worker"


def test_preference_pass_routes_holder_to_claimed_worker():
    """Preference pass resolves holder tasks through their parent's claims."""
    parent_job_id = JobName.root("test-user", "res-job")
    holder_job_id = parent_job_id.child(RESERVATION_HOLDER_JOB_NAME)
    holder_task_id = holder_job_id.task(0)

    w1 = _make_worker("w1")
    w2 = _make_worker("w2")

    req = _make_job_requirements()
    has_reservation = {holder_job_id}

    # Claim is keyed by the parent job's wire ID.
    claims = {WorkerId("w1"): ReservationClaim(job_id=parent_job_id.to_wire(), entry_idx=0)}

    context = _build_context_with_workers(
        [w1, w2],
        pending_tasks=[holder_task_id],
        jobs={holder_job_id: req},
    )

    assignments = _preference_pass(context, has_reservation, claims)

    assert len(assignments) == 1
    assert assignments[0] == (
        holder_task_id,
        WorkerId("w1"),
    ), f"holder task should land on claimed worker w1, got: {assignments}"
    assert holder_task_id not in context.pending_tasks
