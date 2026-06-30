# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RpcTaskBackend scheduling-tick behavior."""

from pathlib import Path

from iris.cluster.backends.rpc.backend import RpcTaskBackend
from iris.cluster.controller.autoscaler.reserved_pool import ReservationLedger
from iris.cluster.controller.backend import BackendRuntime, ScheduleRequest
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.projections.worker_attrs import WorkerAttrsProjection
from iris.cluster.controller.run_template import new_run_template_cache
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


def _bound_backend(db: ControllerDB, autoscaler: _RecordingAutoscaler) -> RpcTaskBackend:
    backend = RpcTaskBackend(stub_factory=object())
    backend.autoscaler = autoscaler
    backend.bind_runtime(
        BackendRuntime(
            db=db,
            endpoints=EndpointsProjection(db),
            run_template_cache=new_run_template_cache(),
            worker_attrs=WorkerAttrsProjection(db),
            owns_scale_group=lambda _: True,
            budget_defaults=UserBudgetDefaults(),
        )
    )
    return backend


def _request(*, autoscale_runs: bool) -> ScheduleRequest:
    return ScheduleRequest(
        pending_task_rows=[],
        requested_bands={},
        user_spend={},
        user_budget_limits={},
        user_budget_defaults=UserBudgetDefaults(),
        max_tasks_per_job_per_cycle=1,
        autoscale_runs=autoscale_runs,
    )


def test_reservation_ledger_only_built_when_autoscale_runs(tmp_path: Path):
    """Cross-variant preemption is gated on the autoscaler running this tick.

    A schedule-only mini-tick (a submit wake) commits its preemptions but never
    runs the drain, so building the ledger there would let it finalize
    cross-variant victims to PENDING with no slice teardown to reclaim their
    reserved chips. The backend must consult the reservation ledger only when the
    autoscaler will act on the resulting drain.
    """
    db = ControllerDB(db_dir=tmp_path)
    try:
        autoscaler = _RecordingAutoscaler()
        backend = _bound_backend(db, autoscaler)

        backend.schedule(_request(autoscale_runs=False))
        assert autoscaler.ledger_calls == 0

        backend.schedule(_request(autoscale_runs=True))
        assert autoscaler.ledger_calls == 1
    finally:
        db.close()
