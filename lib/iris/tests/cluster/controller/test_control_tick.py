# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the unified control tick (``single_control_tick=True``).

The control tick collapses the legacy scheduling/polling/autoscaler loops into one
driver thread: each tick builds a single read snapshot, runs the phases that are
due (or, on a wake, a schedule-only mini-tick), folds backend-observed health, and
commits through a single end-of-tick write transaction. These tests pin that
contract — one read + one write per steady-state tick, wake-driven assignment
latency, reconcile-fail teardown, and the runtime fallback flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from iris.cluster.controller.backend import (
    AutoscaleResult,
    BackendCapability,
    ReconcileResult,
    ScheduleInput,
    ScheduleResult,
    plans_from_snapshot,
    run_scheduling_decision,
)
from iris.cluster.controller.reads import ControlSnapshot
from iris.cluster.controller.reconcile.worker import WorkerReconcilePlan, WorkerReconcileResult
from iris.cluster.controller.scheduling.scheduler import Scheduler
from iris.cluster.controller.worker_health import WorkerHealthEvent, WorkerHealthEventKind
from iris.rpc import job_pb2
from rigging.timing import Duration, RateLimiter

from tests.cluster.controller._test_support import ControllerTestState
from tests.cluster.controller.conftest import (
    make_job_request,
    make_worker_metadata,
    query_task,
    query_worker,
    register_worker,
    submit_job,
)

_W1 = "worker-1"
_W1_ADDR = "worker-1:8080"


@dataclass
class _RecordingProvider:
    """Worker-daemon backend that runs the real scheduler and records phase calls.

    ``reconcile`` reports every reached worker healthy (REACHED) with no
    observations; ``schedule`` runs the real Iris pipeline; ``autoscale`` is a
    no-op. The call counters let tests assert which phases a tick actually ran.
    """

    name: str = "worker"
    capabilities: ClassVar[frozenset[BackendCapability]] = frozenset(
        {BackendCapability.WORKER_DAEMON, BackendCapability.IRIS_AUTOSCALER}
    )
    autoscaler: object | None = None
    schedule_calls: int = 0
    reconcile_calls: int = 0
    autoscale_calls: int = 0
    unreachable: set[str] = field(default_factory=set)
    _scheduler: Scheduler = field(default_factory=Scheduler, init=False, repr=False)

    def schedule(self, snapshot: ScheduleInput) -> ScheduleResult:
        self.schedule_calls += 1
        return run_scheduling_decision(self._scheduler, snapshot)

    def reconcile(self, snapshot: ControlSnapshot) -> ReconcileResult:
        self.reconcile_calls += 1
        plans = plans_from_snapshot(snapshot)
        worker_results: list[tuple[WorkerReconcilePlan, WorkerReconcileResult]] = []
        events: list[WorkerHealthEvent] = []
        for plan in plans:
            if str(plan.worker_id) in self.unreachable:
                worker_results.append(
                    (plan, WorkerReconcileResult(worker_id=plan.worker_id, observations=[], error="rpc unreachable"))
                )
                events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.UNREACHABLE))
            else:
                worker_results.append(
                    (plan, WorkerReconcileResult(worker_id=plan.worker_id, observations=[], error=None))
                )
                events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.REACHED))
        return ReconcileResult(worker_results=worker_results, health_events=events)

    def autoscale(self, snapshot: ControlSnapshot, residual_demand, dead_workers) -> AutoscaleResult:
        self.autoscale_calls += 1
        # Dead workers tear down with no slice siblings in these tests.
        return AutoscaleResult(removed_workers=list(dead_workers))

    def attach_autoscaler(self, autoscaler) -> None:
        self.autoscaler = autoscaler

    def set_log_sink(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


class _DbCallCounter:
    """Counts ``read_snapshot`` / ``transaction`` opens on a ControllerDB instance.

    Wraps the bound methods so a test can assert how many DB transactions one
    control tick issued. Install after construction so seeding/setup reads do not
    count toward the per-tick total.
    """

    def __init__(self, db) -> None:
        self.reads = 0
        self.writes = 0
        orig_read = db.read_snapshot
        orig_txn = db.transaction

        def read_snapshot(*args, **kwargs):
            self.reads += 1
            return orig_read(*args, **kwargs)

        def transaction(*args, **kwargs):
            self.writes += 1
            return orig_txn(*args, **kwargs)

        db.read_snapshot = read_snapshot
        db.transaction = transaction


def _due_limiters() -> tuple[RateLimiter, RateLimiter, RateLimiter]:
    """Three limiters that all report due (interval 0)."""
    return RateLimiter(0.0), RateLimiter(0.0), RateLimiter(0.0)


def _state(ctrl) -> ControllerTestState:
    return ControllerTestState(
        ctrl._db,
        health=ctrl._health,
        endpoints=ctrl._endpoints,
        worker_attrs=ctrl._worker_attrs,
        run_template_cache=ctrl._run_template_cache,
    )


def test_full_tick_is_one_read_snapshot_and_one_write_txn(make_controller):
    """A steady-state tick (schedule + reconcile + autoscale, no worker death)
    issues exactly one read snapshot and one write transaction."""
    provider = _RecordingProvider()
    ctrl = make_controller(provider=provider)
    state = _state(ctrl)

    register_worker(state, _W1, _W1_ADDR, make_worker_metadata())
    submit_job(state, "tick-job", make_job_request(name="tick-job"))

    counter = _DbCallCounter(ctrl._db)
    sched, recon, auto = _due_limiters()
    ctrl._control_tick(woken=False, schedule_limiter=sched, reconcile_limiter=recon, autoscale_limiter=auto)

    assert provider.schedule_calls == 1
    assert provider.reconcile_calls == 1
    assert provider.autoscale_calls == 1
    assert counter.reads == 1, f"expected one read snapshot per tick, got {counter.reads}"
    assert counter.writes == 1, f"expected one write txn per tick, got {counter.writes}"


def test_wake_runs_schedule_only_mini_tick(make_controller):
    """A wake assigns the pending task (submit->assign = schedule time) without
    running reconcile or autoscale when neither is due."""
    provider = _RecordingProvider()
    ctrl = make_controller(provider=provider)
    state = _state(ctrl)

    register_worker(state, _W1, _W1_ADDR, make_worker_metadata())
    tasks = submit_job(state, "wake-job", make_job_request(name="wake-job"))
    task_id = tasks[0].task_id

    # Prime the limiters so none are due; only the wake should trigger schedule.
    sched, recon, auto = RateLimiter(9999.0), RateLimiter(9999.0), RateLimiter(9999.0)
    sched.mark_run()
    recon.mark_run()
    auto.mark_run()

    ctrl._control_tick(woken=True, schedule_limiter=sched, reconcile_limiter=recon, autoscale_limiter=auto)

    assert provider.schedule_calls == 1
    assert provider.reconcile_calls == 0, "wake mini-tick must not reconcile when reconcile is not due"
    assert provider.autoscale_calls == 0, "wake mini-tick must not autoscale when autoscale is not due"
    assert query_task(state, task_id).state == job_pb2.TASK_STATE_ASSIGNED


def test_autoscale_tick_always_schedules_for_fresh_demand(make_controller):
    """An autoscale-due tick schedules first so it provisions against this tick's
    residual demand (no cross-tick _last_residual_demand handoff). A pending job
    with no worker keeps the scheduler busy so the schedule phase reaches the
    backend."""
    provider = _RecordingProvider()
    ctrl = make_controller(provider=provider)
    state = _state(ctrl)
    submit_job(state, "unplaceable-job", make_job_request(name="unplaceable-job"))

    # Schedule and reconcile not due; only autoscale due. Schedule must still run.
    sched, recon = RateLimiter(9999.0), RateLimiter(9999.0)
    sched.mark_run()
    recon.mark_run()
    auto = RateLimiter(0.0)

    ctrl._control_tick(woken=False, schedule_limiter=sched, reconcile_limiter=recon, autoscale_limiter=auto)

    assert provider.autoscale_calls == 1
    assert provider.schedule_calls == 1, "autoscale must pair with a fresh schedule for same-tick demand"


def test_reconcile_failure_tears_down_worker_in_unified_tick(make_controller):
    """An unreachable worker over threshold is failed + torn down within the
    unified tick's post-commit health fold — no separate ping channel."""
    provider = _RecordingProvider(unreachable={_W1})
    # grace / poll = 1 → a single failed reconcile crosses the threshold.
    ctrl = make_controller(
        provider=provider,
        worker_unreachable_grace=Duration.from_seconds(1.0),
        poll_interval=Duration.from_seconds(1.0),
    )
    state = _state(ctrl)
    wid = register_worker(state, _W1, _W1_ADDR, make_worker_metadata())

    sched, recon, auto = _due_limiters()
    ctrl._control_tick(woken=False, schedule_limiter=sched, reconcile_limiter=recon, autoscale_limiter=auto)

    # Teardown removes the worker row entirely and forgets its liveness.
    assert query_worker(state, wid) is None, "unreachable worker over threshold must be torn down"
    assert not ctrl._health.liveness(wid).active
    # Teardown routes through autoscale(dead_workers=...): the provisioning call
    # plus the dead-worker teardown call.
    assert provider.autoscale_calls >= 2


def test_single_control_tick_flag_selects_thread_set(make_controller):
    """The fallback flag swaps the loop structure: one control thread when on,
    the legacy scheduling + polling + autoscaler threads when off."""
    unified = make_controller(provider=_RecordingProvider(), single_control_tick=True)
    unified.start()
    assert unified._control_thread is not None
    assert unified._scheduling_thread is None
    assert unified._polling_thread is None
    assert unified._autoscaler_thread is None

    legacy = make_controller(provider=_RecordingProvider(), single_control_tick=False)
    legacy.start()
    assert legacy._control_thread is None
    assert legacy._scheduling_thread is not None
    assert legacy._polling_thread is not None
    assert legacy._autoscaler_thread is not None
