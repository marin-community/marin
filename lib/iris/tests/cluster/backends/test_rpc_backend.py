# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RpcTaskBackend scheduling-tick behavior."""

from iris.cluster.backends.rpc.backend import RpcTaskBackend
from iris.cluster.controller.autoscaler.reserved_pool import ReservationLedger
from iris.cluster.controller.backend import ScheduleInput
from iris.cluster.controller.scheduling.scheduler import SchedulingContext
from iris.cluster.types import UserBudgetDefaults


class _RecordingAutoscaler:
    """Stand-in autoscaler that records whether the reservation ledger was consulted."""

    def __init__(self):
        self.ledger_calls = 0

    def zone_capabilities(self):
        return {}

    def reservation_ledger(self):
        self.ledger_calls += 1
        return ReservationLedger(pools={}, worker_pool={}, worker_slice={}, variant_pool={}, chips_per_variant={})


def _empty_context() -> SchedulingContext:
    return SchedulingContext(
        workers=[],
        building_counts={},
        max_building_tasks=0,
        max_assignments_per_worker=0,
        pending_tasks=[],
        jobs={},
        pending_task_rows=[],
        user_spend={},
        user_budget_limits={},
        requested_bands={},
        user_budget_defaults=UserBudgetDefaults(),
    )


def test_reservation_ledger_only_built_when_autoscale_runs():
    """Cross-variant preemption is gated on the autoscaler running this tick.

    A schedule-only mini-tick (a submit wake) commits its preemptions but never
    runs the drain, so building the ledger there would let it finalize
    cross-variant victims to PENDING with no slice teardown to reclaim their
    reserved chips. The backend must consult the reservation ledger only when the
    autoscaler will act on the resulting drain.
    """
    backend = RpcTaskBackend(stub_factory=object())
    autoscaler = _RecordingAutoscaler()
    backend.attach_autoscaler(autoscaler)

    backend.schedule(ScheduleInput(context=_empty_context(), max_tasks_per_job_per_cycle=1, autoscale_runs=False))
    assert autoscaler.ledger_calls == 0

    backend.schedule(ScheduleInput(context=_empty_context(), max_tasks_per_job_per_cycle=1, autoscale_runs=True))
    assert autoscaler.ledger_calls == 1
