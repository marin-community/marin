# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reservation reclaim: a waiting higher-band slice takes chips back from lower-band slices in its pool."""

import dataclasses
from dataclasses import dataclass, field

from iris.cluster.config import AutoscalerConfig, ReservationReclaimConfig
from iris.cluster.constraints import DeviceType
from iris.cluster.controller.autoscaler.reclaim import (
    BLOCKED_AWAITING_CAPACITY,
    PoolSlice,
    ReclaimDecision,
    plan_reservation_reclaim,
)
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.cluster.controller.backend import (
    AutoscaleRequest,
    AutoscaleResult,
    BackendDescriptor,
    BackendObservation,
    BackendObservationRequest,
    BackendRecoveryRequest,
    BackendRecoveryResult,
    JobFeasibilityRequest,
    ReconcileObservation,
    ReconcileRequest,
    RemoveCapacityRequest,
    RemoveCapacityResult,
    ScheduleRequest,
    ScheduleResult,
    WorkerFleetReconcileRequest,
)
from iris.cluster.controller.controller import with_assignments
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.scheduling.scheduler import Scheduler
from iris.cluster.controller.worker_health import WorkerHealthEvent, WorkerHealthEventKind
from iris.cluster.types import CapacityType, JobName, WorkerId, WorkerStatus
from iris.rpc import controller_pb2, job_pb2
from iris.testing.backends import make_mock_platform, make_mock_slice_handle
from iris.testing.controller import (
    dispatch_task,
    make_autoscaler,
    make_demand_entries,
    make_job_request,
    make_scale_group_config,
    make_worker_metadata,
    query_task,
    query_worker,
    register_worker,
    submit_job,
    worker_backend_descriptor,
)
from iris.testing.controller_state import ControllerTestState
from rigging.timing import RateLimiter, Timestamp

POOL = "v4-reserved/us-central2-b"
PRODUCTION = job_pb2.PRIORITY_BAND_PRODUCTION
INTERACTIVE = job_pb2.PRIORITY_BAND_INTERACTIVE
BATCH = job_pb2.PRIORITY_BAND_BATCH
MINUTE_MS = 60_000
NOW = Timestamp.from_ms(60 * MINUTE_MS)
ON = ReservationReclaimConfig(quota_pools=[POOL])


def _slice(
    slice_id: str,
    group: str,
    chips: int,
    *,
    ready: bool = True,
    created_min: int = 0,
    pool: str = POOL,
    workers: int = 1,
) -> PoolSlice:
    return PoolSlice(
        slice_id=slice_id,
        group=group,
        quota_pool=pool,
        chips=chips,
        ready=ready,
        waiting=not ready,
        created_at=Timestamp.from_ms(created_min * MINUTE_MS),
        worker_ids=tuple(f"{slice_id}-w{i}" for i in range(workers)),
    )


def _status(slices: list[PoolSlice], bands: dict[str, int | None]) -> dict[str, WorkerStatus]:
    """Every worker of a slice runs a task of the slice's band (None = idle)."""
    status = {}
    for slice_ in slices:
        band = bands.get(slice_.slice_id)
        for worker_id in slice_.worker_ids:
            status[worker_id] = WorkerStatus(
                worker_id=worker_id,
                running_task_ids=frozenset() if band is None else frozenset({f"{worker_id}/task"}),
                running_bands=frozenset() if band is None else frozenset({band}),
            )
    return status


def _plan(slices, bands, *, config=ON, waiting=None, last_reclaim=None, now=NOW):
    return plan_reservation_reclaim(
        slices=slices,
        waiting_bands=waiting if waiting is not None else {"v4-1024": PRODUCTION},
        worker_status=_status(slices, bands),
        config=config,
        last_reclaim=last_reclaim or {},
        now=now,
    )


def _fragmented_pool() -> tuple[list[PoolSlice], dict[str, int | None]]:
    """A full reserved pool of smaller, lower-band slices, with a production v4-1024 waiting."""
    slices = [
        _slice("wait-1024", "v4-1024", 512, ready=False, created_min=30),
        _slice("i-512", "v4-512", 256, created_min=5),
        _slice("b-256", "v4-256", 128, created_min=6),
        _slice("b-128", "v4-128", 64, created_min=7),
        *[_slice(f"b-8-{i}", "v4-8", 4, created_min=10 + i) for i in range(96)],
        _slice("idle-64", "v4-64", 32, created_min=8),
    ]
    bands: dict[str, int | None] = {s.slice_id: BATCH for s in slices}
    bands["i-512"] = INTERACTIVE
    bands["idle-64"] = None
    return slices, bands


def test_fragmented_pool_frees_idle_then_smallest_batch_slices_until_the_waiting_slice_fits():
    slices, bands = _fragmented_pool()
    plan = _plan(slices, bands)

    assert plan.blocked == []
    (decision,) = plan.decisions
    assert decision.waiting_slice_id == "wait-1024"
    assert decision.chips_needed == 512
    victims = [v.slice_id for v in decision.victims]
    # Idle first, then batch by size (the 96 v4-8 slices, youngest first), then the larger batch slices.
    assert victims[0] == "idle-64"
    assert victims[1:97] == [f"b-8-{i}" for i in reversed(range(96))]
    assert victims[97:] == ["b-128", "b-256"]
    assert decision.chips_freed == 32 + 96 * 4 + 64 + 128 >= 512
    # Batch sufficed, so the interactive slice is untouched.
    assert "i-512" not in victims


def test_a_slice_with_one_equal_band_task_is_skipped_whole():
    waiting = _slice("wait", "big", 8, ready=False)
    mixed = _slice("mixed", "small", 8, workers=2)
    slices = [waiting, mixed]
    status = _status(slices, {"mixed": BATCH})
    status["mixed-w1"] = dataclasses.replace(status["mixed-w1"], running_bands=frozenset({BATCH, PRODUCTION}))

    plan = plan_reservation_reclaim(slices, {"big": PRODUCTION}, status, ON, {}, NOW)

    assert plan.decisions == []
    (blocked,) = plan.blocked
    assert blocked.chips_needed == 8 and blocked.chips_reclaimable == 0


def test_equal_and_higher_band_slices_are_never_victims():
    slices = [
        _slice("wait", "big", 8, ready=False),
        _slice("prod", "small", 8),
        _slice("system", "small", 8),
    ]
    plan = _plan(slices, {"prod": PRODUCTION, "system": job_pb2.PRIORITY_BAND_SYSTEM}, waiting={"big": PRODUCTION})
    assert plan.decisions == []
    assert plan.blocked[0].chips_reclaimable == 0


def test_too_few_reclaimable_chips_reclaims_nothing():
    slices = [_slice("wait", "big", 16, ready=False), _slice("b1", "small", 4), _slice("b2", "small", 4)]
    plan = _plan(slices, {"b1": BATCH, "b2": BATCH}, waiting={"big": PRODUCTION})
    assert plan.decisions == []
    (blocked,) = plan.blocked
    assert (blocked.chips_needed, blocked.chips_reclaimable) == (16, 8)


def test_highest_victim_band_batch_keeps_interactive_slices():
    slices = [_slice("wait", "big", 8, ready=False), _slice("i", "small", 8), _slice("b", "small", 8)]
    bands = {"i": INTERACTIVE, "b": BATCH}
    plan = _plan(
        slices, bands, config=ON.model_copy(update={"highest_victim_band": "batch"}), waiting={"big": PRODUCTION}
    )
    assert [v.slice_id for v in plan.decisions[0].victims] == ["b"]

    only_interactive = [s for s in slices if s.slice_id != "b"]
    plan = _plan(
        only_interactive,
        bands,
        config=ON.model_copy(update={"highest_victim_band": "batch"}),
        waiting={"big": PRODUCTION},
    )
    assert plan.decisions == []


def test_interactive_demand_reclaims_only_batch():
    slices = [_slice("wait", "big", 8, ready=False), _slice("i", "small", 8), _slice("b", "small", 8)]
    plan = _plan(slices, {"i": INTERACTIVE, "b": BATCH}, waiting={"big": INTERACTIVE})
    assert [v.slice_id for v in plan.decisions[0].victims] == ["b"]


def test_no_reclaim_before_min_wait_or_inside_cooldown():
    slices = [_slice("wait", "big", 8, ready=False, created_min=55), _slice("b", "small", 8)]
    bands = {"b": BATCH}
    # Waited 5 minutes; min_wait is 10.
    assert _plan(slices, bands, waiting={"big": PRODUCTION}).decisions == []
    # Waited long enough, but the pool reclaimed 20 minutes ago (cooldown 30).
    old = [_slice("wait", "big", 8, ready=False, created_min=0), _slice("b", "small", 8)]
    recent = {POOL: Timestamp.from_ms(40 * MINUTE_MS)}
    assert _plan(old, bands, waiting={"big": PRODUCTION}, last_reclaim=recent).decisions == []
    expired = {POOL: Timestamp.from_ms(25 * MINUTE_MS)}
    assert len(_plan(old, bands, waiting={"big": PRODUCTION}, last_reclaim=expired).decisions) == 1


def test_off_by_default_and_other_pools_are_never_touched():
    slices = [_slice("wait", "big", 8, ready=False), _slice("b", "small", 8)]
    bands = {"b": BATCH}
    assert _plan(slices, bands, config=ReservationReclaimConfig(), waiting={"big": PRODUCTION}).decisions == []

    other_pool = [_slice("wait", "big", 8, ready=False), _slice("b", "small", 8, pool="v5p-reserved/us-east5-a")]
    plan = _plan(other_pool, bands, waiting={"big": PRODUCTION})
    assert plan.decisions == [] and plan.blocked[0].chips_reclaimable == 0


def test_a_worker_missing_from_the_snapshot_excludes_its_slice():
    slices = [_slice("wait", "big", 8, ready=False), _slice("b", "small", 8, workers=2)]
    status = _status(slices, {"b": BATCH})
    del status["b-w1"]
    assert plan_reservation_reclaim(slices, {"big": PRODUCTION}, status, ON, {}, NOW).decisions == []


def _reserved_group(name: str, variant: str, num_vms: int, handles) -> ScalingGroup:
    config = make_scale_group_config(
        name=name,
        accelerator_variant=variant,
        capacity_type=CapacityType.RESERVED,
        num_vms=num_vms,
        max_slices=4,
        zones=["us-central2-b"],
        quota_pool=POOL,
    )
    config.resources.device_count = 4
    group = ScalingGroup(config, make_mock_platform(slices_to_discover=handles))
    group.reconcile()
    return group


@dataclass
class _ReclaimScenario:
    """Two busy batch v4-8 slices fill the pool; a production v4-16 slice ``big-0`` waits on the cloud."""

    autoscaler: object
    small: ScalingGroup
    big: ScalingGroup
    busy: dict[str, WorkerStatus]
    demand: list
    small_demand: list


def _reclaim_scenario() -> _ReclaimScenario:
    small_handles = [make_mock_slice_handle(f"small-{i}", scale_group="small", all_ready=True) for i in range(2)]
    small = _reserved_group("small", "v4-8", 1, small_handles)
    for handle in small_handles:
        small.mark_slice_ready(handle.slice_id, [w.worker_id for w in handle.describe().workers])
    big = _reserved_group("big", "v4-16", 2, [make_mock_slice_handle("big-0", scale_group="big", created_at_ms=0)])
    autoscaler = make_autoscaler(
        {"small": small, "big": big},
        config=AutoscalerConfig(reservation_reclaim=ReservationReclaimConfig(quota_pools=[POOL])),
    )
    busy = {
        worker.worker_id: WorkerStatus(
            worker_id=worker.worker_id, running_task_ids=frozenset({"t"}), running_bands=frozenset({BATCH})
        )
        for handle in small_handles
        for worker in handle.describe().workers
    }
    # One gang of two tasks, one per VM of the waiting v4-16 slice.
    first, second = make_demand_entries(2, device_variant="v4-16")
    demand = [
        dataclasses.replace(
            first,
            task_ids=first.task_ids + second.task_ids,
            coschedule_group_id="/user/big-job",
            priority_band=PRODUCTION,
        )
    ]
    small_demand = [
        dataclasses.replace(entry, priority_band=BATCH)
        for entry in make_demand_entries(3, device_type=DeviceType.TPU, device_variant="v4-8", task_prefix="b")
    ]
    return _ReclaimScenario(autoscaler, small, big, busy, demand, small_demand)


def _at_min(minutes: int) -> Timestamp:
    return Timestamp.from_ms(minutes * MINUTE_MS)


def _actions(autoscaler, action_type: str) -> list:
    return [a for a in autoscaler.get_status().recent_actions if a.action_type == action_type]


def test_autoscaler_reclaims_for_a_waiting_production_slice_and_holds_victim_scale_ups():
    sc = _reclaim_scenario()
    try:
        sc.autoscaler.run_once(sc.demand, sc.busy, timestamp=_at_min(11))
        (decision,) = sc.autoscaler.take_reclaims()
        assert decision.group == "big" and decision.chips_needed == 8
        assert sorted(v.slice_id for v in decision.victims) == ["small-0", "small-1"]
        assert sorted(decision.worker_ids) == sorted(sc.busy)
        assert sc.autoscaler.take_reclaims() == []
        assert _actions(sc.autoscaler, "reservation_reclaim")

        # While the waiting slice still waits, the victims' own group may not scale back up into the freed chips.
        sc.autoscaler.run_once(sc.small_demand, sc.busy, timestamp=_at_min(12))
        assert sc.small.slice_count() == 2
    finally:
        sc.autoscaler.shutdown()


def test_hold_lifts_once_the_waiting_slice_is_ready():
    sc = _reclaim_scenario()
    try:
        sc.autoscaler.run_once(sc.demand, sc.busy, timestamp=_at_min(11))
        assert len(sc.autoscaler.take_reclaims()) == 1

        # The cloud fills the waiting slice well inside the cooldown; other groups may scale up again.
        sc.big.mark_slice_ready("big-0", ["big-0-w0", "big-0-w1"])
        sc.autoscaler.run_once(sc.small_demand, sc.busy, timestamp=_at_min(12))
        assert sc.small.slice_count() > 2
    finally:
        sc.autoscaler.shutdown()


def test_a_slice_the_cloud_never_fills_is_reclaimed_for_only_once():
    sc = _reclaim_scenario()
    try:
        sc.autoscaler.run_once(sc.demand, sc.busy, timestamp=_at_min(11))
        assert len(sc.autoscaler.take_reclaims()) == 1

        # Past the cooldown, big-0 still waits and the pool again holds busy batch slices
        # (e.g. other slices took the freed chips). No second reclaim for the same waiting slice.
        sc.autoscaler.run_once(sc.demand, sc.busy, timestamp=_at_min(11 + 31))
        assert sc.autoscaler.take_reclaims() == []
        blocked = _actions(sc.autoscaler, "reclaim_blocked")
        assert blocked and blocked[-1].slice_id == "big-0"
        assert "awaiting_capacity" in blocked[-1].reason
        assert len(_actions(sc.autoscaler, "reservation_reclaim")) == 1
    finally:
        sc.autoscaler.shutdown()


def test_an_outstanding_reclaim_blocks_its_pool_until_the_slice_stops_waiting():
    slices = [_slice("wait", "big", 8, ready=False), _slice("b", "small", 8)]
    bands = {"b": BATCH}
    status = _status(slices, bands)
    plan = plan_reservation_reclaim(slices, {"big": PRODUCTION}, status, ON, {}, NOW, outstanding={POOL: "wait"})
    assert plan.decisions == []
    (blocked,) = plan.blocked
    assert (blocked.waiting_slice_id, blocked.reason) == ("wait", BLOCKED_AWAITING_CAPACITY)

    # Another pool's outstanding reclaim does not block this one.
    plan = plan_reservation_reclaim(
        slices, {"big": PRODUCTION}, status, ON, {}, NOW, outstanding={"v5p-reserved/us-east5-a": "x"}
    )
    assert len(plan.decisions) == 1


@dataclass
class _ReclaimingProvider:
    """Worker backend whose autoscale phase returns the given reclaims once; every worker stays reachable."""

    reclaims: list[ReclaimDecision]
    remove_capacity_calls: list[list[WorkerId]] = field(default_factory=list)
    descriptor: BackendDescriptor = field(default_factory=worker_backend_descriptor)
    _scheduler: Scheduler = field(default_factory=Scheduler, init=False, repr=False)

    def initialize(self, request: BackendRecoveryRequest) -> BackendRecoveryResult:
        return BackendRecoveryResult()

    def observe(self, request: BackendObservationRequest) -> BackendObservation:
        return BackendObservation()

    def runtime_image(self, requested_image: str) -> str:
        return requested_image

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        return ScheduleResult()

    def reconcile(self, request: ReconcileRequest) -> ReconcileObservation:
        assert isinstance(request, WorkerFleetReconcileRequest)
        events = [WorkerHealthEvent(t.plan.worker_id, WorkerHealthEventKind.REACHED) for t in request.targets]
        return ReconcileObservation(task_updates=[], worker_health_events=events)

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        reclaims, self.reclaims = self.reclaims, []
        return AutoscaleResult(reclaims=reclaims)

    def remove_capacity(self, request: RemoveCapacityRequest) -> RemoveCapacityResult:
        self.remove_capacity_calls.append(list(request.worker_ids))
        return RemoveCapacityResult()

    def validate_job(self, request: controller_pb2.Controller.LaunchJobRequest) -> str | None:
        return None

    def job_feasibility(self, request: JobFeasibilityRequest) -> str | None:
        return None

    def close(self):
        pass


def test_controller_fails_and_removes_reclaimed_workers_and_retries_their_tasks(make_controller):
    victim_slice = _slice("victim", "small", 4)
    decision = ReclaimDecision(
        quota_pool=POOL,
        group="big",
        waiting_slice_id="wait",
        preemptor_band=PRODUCTION,
        chips_needed=4,
        victims=(dataclasses.replace(victim_slice, worker_ids=("worker-victim",)),),
    )
    provider = _ReclaimingProvider(reclaims=[decision])
    ctrl = make_controller(provider=provider)
    state = ControllerTestState(ctrl._db, health=ctrl.worker_health)
    victim = register_worker(state, "worker-victim", "worker-victim:8080", make_worker_metadata())
    keeper = register_worker(state, "worker-keeper", "worker-keeper:8080", make_worker_metadata())
    [task] = submit_job(state, "batch-job", make_job_request("batch-job", max_retries_preemption=1))
    dispatch_task(state, task, victim)

    ctrl._control_tick(
        woken=False,
        schedule_limiter=RateLimiter(interval_seconds=1e9),
        reconcile_limiter=RateLimiter(interval_seconds=1e9),
        autoscale_limiter=RateLimiter(interval_seconds=1e9),
    )

    assert provider.remove_capacity_calls == [[victim]]
    assert query_worker(state, victim) is None
    assert query_worker(state, keeper) is not None
    row = query_task(state, task.task_id)
    # The worker-failure path retries the task on its preemption budget.
    assert row.state == job_pb2.TASK_STATE_PENDING
    assert row.preemption_count == 1


def test_unknown_bands_never_trigger_reclaim_and_never_make_victims():
    unknown = job_pb2.PRIORITY_BAND_INHERIT  # a sentinel, never a resolved band
    slices = [_slice("wait", "big", 8, ready=False), _slice("b", "small", 8)]
    assert _plan(slices, {"b": BATCH}, waiting={"big": unknown}).decisions == []
    plan = _plan(slices, {"b": unknown}, waiting={"big": PRODUCTION})
    assert plan.decisions == [] and plan.blocked[0].chips_reclaimable == 0


@dataclass
class _AssignThenPlanProvider(_ReclaimingProvider):
    """Schedules one task onto an idle worker, then plans reclaim from the worker view the controller hands over."""

    assignments: list[Assignment] = field(default_factory=list)
    waiting: PoolSlice | None = None
    victim: PoolSlice | None = None

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        return ScheduleResult(assignments=self.assignments)

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        assert self.waiting is not None and self.victim is not None
        plan = plan_reservation_reclaim(
            [self.waiting, self.victim], {self.waiting.group: PRODUCTION}, request.worker_status, ON, {}, NOW
        )
        return AutoscaleResult(reclaims=plan.decisions)


def test_a_task_assigned_this_tick_keeps_its_idle_worker_out_of_reclaim(make_controller):
    waiting = _slice("wait", "big", 4, ready=False)
    victim = dataclasses.replace(_slice("victim", "small", 4), worker_ids=("worker-victim",))
    provider = _AssignThenPlanProvider(reclaims=[], waiting=waiting, victim=victim)
    ctrl = make_controller(provider=provider)
    state = ControllerTestState(ctrl._db, health=ctrl.worker_health)
    worker = register_worker(state, "worker-victim", "worker-victim:8080", make_worker_metadata())
    [task] = submit_job(state, "prod-job", make_job_request("prod-job"))
    # The worker is idle in the tick's snapshot; this tick's scheduling places a production task on it.
    provider.assignments = [Assignment(task_id=task.task_id, worker_id=worker, priority_band=PRODUCTION)]

    ctrl._control_tick(
        woken=False,
        schedule_limiter=RateLimiter(interval_seconds=1e9),
        reconcile_limiter=RateLimiter(interval_seconds=1e9),
        autoscale_limiter=RateLimiter(interval_seconds=1e9),
    )

    assert provider.remove_capacity_calls == []
    assert query_worker(state, worker) is not None
    row = query_task(state, task.task_id)
    assert row.current_worker_id == worker
    assert row.state != job_pb2.TASK_STATE_WORKER_FAILED


def test_with_assignments_marks_workers_busy_at_the_assigned_band():
    idle = WorkerStatus(worker_id="w", running_task_ids=frozenset())
    task_id = JobName.from_wire("/user/job/0")
    merged = with_assignments(
        {WorkerId("w"): idle},
        [
            Assignment(task_id=task_id, worker_id=WorkerId("w"), priority_band=BATCH),
            Assignment(task_id=JobName.from_wire("/user/job/1"), worker_id=WorkerId("gone")),
        ],
    )
    assert merged[WorkerId("w")].running_bands == frozenset({BATCH})
    assert merged[WorkerId("w")].running_task_ids == frozenset({task_id.to_wire()})
    assert WorkerId("gone") not in merged
    # No band known: an unknown band, which reclaim never preempts.
    no_band = with_assignments({WorkerId("w"): idle}, [Assignment(task_id=task_id, worker_id=WorkerId("w"))])
    assert no_band[WorkerId("w")].running_bands == frozenset({job_pb2.PRIORITY_BAND_INHERIT})
