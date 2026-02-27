# Copyright 2025 The Marin Authors
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
    _inject_reservation_taints,
    _inject_taint_constraints,
    _worker_matches_reservation_entry,
)
from iris.cluster.controller.events import (
    JobSubmittedEvent,
    TaskStateChangedEvent,
    WorkerRegisteredEvent,
)
from iris.cluster.controller.scheduler import JobRequirements
from iris.cluster.controller.state import ControllerState, ControllerWorker
from iris.cluster.types import AttributeValue, JobName, WorkerId
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
    return cluster_pb2.WorkerMetadata(
        hostname="test",
        ip_address="127.0.0.1",
        cpu_count=8,
        memory_bytes=16 * 1024**3,
        disk_bytes=100 * 1024**3,
        device=_cpu_device(),
    )


def _gpu_metadata(variant: str = "H100") -> cluster_pb2.WorkerMetadata:
    return cluster_pb2.WorkerMetadata(
        hostname="test-gpu",
        ip_address="127.0.0.1",
        cpu_count=32,
        memory_bytes=256 * 1024**3,
        disk_bytes=500 * 1024**3,
        device=_gpu_device(variant),
    )


def _make_worker(
    worker_id: str,
    metadata: cluster_pb2.WorkerMetadata | None = None,
    attributes: dict[str, AttributeValue] | None = None,
    healthy: bool = True,
) -> ControllerWorker:
    meta = metadata or _cpu_metadata()
    return ControllerWorker(
        worker_id=WorkerId(worker_id),
        address=f"{worker_id}:8080",
        metadata=meta,
        attributes=attributes or {},
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
    jid = JobName.root(job_id)
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
        attributes={"region": AttributeValue("us-central1")},
    )
    constraint = cluster_pb2.Constraint(
        key="region",
        op=cluster_pb2.CONSTRAINT_OP_EQ,
        value=cluster_pb2.AttributeValue(string_value="us-central1"),
    )
    entry = _make_reservation_entry(_cpu_device(), constraints=[constraint])
    assert _worker_matches_reservation_entry(worker, entry)


def test_worker_rejects_unmet_constraint():
    """Worker without the required attribute fails the constraint check."""
    worker = _make_worker("w1", _cpu_metadata())
    constraint = cluster_pb2.Constraint(
        key="region",
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
    assert claim.job_id == JobName.root("j1").to_wire()
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
    job_wire = JobName.root("j1").to_wire()
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
        JobName.root("job-a").to_wire(),
        JobName.root("job-b").to_wire(),
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
        job_id=JobName.root("j1").to_wire(),
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
        JobName.root("regular"): _make_job_requirements(),
    }
    has_reservation: set[JobName] = set()

    result = _inject_taint_constraints(jobs, has_reservation)

    constraints = result[JobName.root("regular")].constraints
    not_exists = [c for c in constraints if c.key == RESERVATION_TAINT_KEY]
    assert len(not_exists) == 1
    assert not_exists[0].op == cluster_pb2.CONSTRAINT_OP_NOT_EXISTS


def test_taint_constraint_not_added_to_reservation_jobs():
    """Reservation jobs do not get the NOT_EXISTS constraint."""
    res_job = JobName.root("reserved")
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
    res_job = JobName.root("reserved")
    reg_job = JobName.root("regular")
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
        key="region",
        op=cluster_pb2.CONSTRAINT_OP_EQ,
        value=cluster_pb2.AttributeValue(string_value="us-central1"),
    )
    jobs = {
        JobName.root("regular"): JobRequirements(
            resources=cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3),
            constraints=[existing],
            is_coscheduled=False,
            coscheduling_group_by=None,
        ),
    }
    has_reservation: set[JobName] = set()

    result = _inject_taint_constraints(jobs, has_reservation)

    constraints = result[JobName.root("regular")].constraints
    assert len(constraints) == 2
    # Original constraint preserved
    region_constraints = [c for c in constraints if c.key == "region"]
    assert len(region_constraints) == 1
    # Taint constraint added
    taint_constraints = [c for c in constraints if c.key == RESERVATION_TAINT_KEY]
    assert len(taint_constraints) == 1
