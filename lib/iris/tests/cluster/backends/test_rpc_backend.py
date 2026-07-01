# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RpcTaskBackend scheduling-tick behavior."""

from pathlib import Path

from iris.cluster.backends.rpc.backend import RpcTaskBackend
from iris.cluster.controller import reads
from iris.cluster.controller.autoscaler.reserved_pool import ReservationLedger, build_reservation_ledger
from iris.cluster.controller.autoscaler.scaling_group import ScalingGroup
from iris.cluster.controller.backend import BackendRuntime, ScheduleRequest
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.run_template import new_run_template_cache
from iris.cluster.types import JobName, UserBudgetDefaults, WorkerId
from iris.rpc import controller_pb2, job_pb2, vm_pb2
from tests.cluster.backends.conftest import make_fake_slice_handle, make_mock_platform
from tests.cluster.controller._test_support import ControllerTestState
from tests.cluster.controller.conftest import (
    dispatch_task,
    make_scale_group_config,
    make_test_entrypoint,
    make_worker_metadata,
    register_worker,
    submit_job,
)

POOL = "v4-res/zone"
BATCH = job_pb2.PRIORITY_BAND_BATCH
PRODUCTION = job_pb2.PRIORITY_BAND_PRODUCTION


class _StubAutoscaler:
    """Stand-in autoscaler that builds a real ledger from real ``ScalingGroup`` state."""

    def __init__(self, groups: list[ScalingGroup]):
        self._groups = groups

    def zone_capabilities(self) -> dict[str, frozenset[str]]:
        return {}

    def reservation_ledger(self) -> ReservationLedger:
        return build_reservation_ledger(self._groups)


def _bound_backend(db: ControllerDB, autoscaler: _StubAutoscaler) -> RpcTaskBackend:
    backend = RpcTaskBackend(stub_factory=object())
    backend.autoscaler = autoscaler
    backend.bind_runtime(
        BackendRuntime(
            db=db,
            endpoints=EndpointsProjection(db),
            run_template_cache=new_run_template_cache(),
            owns_scale_group=lambda _: True,
            budget_defaults=UserBudgetDefaults(),
        )
    )
    return backend


def _tpu_job_request(name: str, variant: str, band: int) -> controller_pb2.Controller.LaunchJobRequest:
    resources = job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3)
    resources.device.tpu.variant = variant
    return controller_pb2.Controller.LaunchJobRequest(
        name=JobName.root("test-user", name).to_wire(),
        entrypoint=make_test_entrypoint(),
        resources=resources,
        environment=job_pb2.EnvironmentConfig(),
        replicas=1,
        priority_band=band,
    )


def _request(*, autoscale_runs: bool, pending_task_rows: list) -> ScheduleRequest:
    return ScheduleRequest(
        pending_task_rows=pending_task_rows,
        requested_bands={},
        user_spend={},
        user_budget_limits={},
        user_budget_defaults=UserBudgetDefaults(),
        max_tasks_per_job_per_cycle=1,
        autoscale_runs=autoscale_runs,
    )


def test_reserved_pool_drain_only_committed_when_autoscale_runs(tmp_path: Path):
    """Cross-variant preemption is gated on the autoscaler running this tick.

    A schedule-only mini-tick (a submit wake) commits its preemptions but never
    runs the drain, so building the ledger there would let it finalize
    cross-variant victims to PENDING with no slice teardown to reclaim their
    reserved chips. The backend must consult the reservation ledger only when the
    autoscaler will act on the resulting drain.

    Scenario: an 8-chip fungible pool holds one running v4-8 BATCH task (4
    chips) and a pending v4-16 PRODUCTION task (needs 8 chips, only 4 free).
    Cross-variant preemption can free the deficit by evicting the BATCH slice.
    """
    db = ControllerDB(db_dir=tmp_path)
    try:
        test_state = ControllerTestState(db, run_template_cache=new_run_template_cache())

        # A running v4-8 BATCH task on a worker belonging to the fungible pool.
        register_worker(
            test_state,
            "batch1-vm-0",
            "batch1-vm-0:8080",
            make_worker_metadata(tpu_name="v4-8"),
            scale_group="v4-8-batch",
        )
        batch_tasks = submit_job(test_state, "batch-job", _tpu_job_request("batch-job", "v4-8", BATCH))
        dispatch_task(test_state, batch_tasks[0], WorkerId("batch1-vm-0"))

        # A pending v4-16 PRODUCTION task; no v4-16 worker exists, so normal
        # placement can't satisfy it and it falls to the reserved-pool pass.
        submit_job(test_state, "prod-job", _tpu_job_request("prod-job", "v4-16", PRODUCTION))
        with db.read_snapshot() as tx:
            pending = reads.pending_tasks_with_jobs(tx)
        assert len(pending) == 1  # the prod task; the batch task is RUNNING, not PENDING

        # Real ScalingGroup state backing the ledger: the batch group's one READY
        # slice matches the worker registered above; the prod group has none.
        batch_config = make_scale_group_config(name="v4-8-batch", accelerator_variant="v4-8", max_slices=8)
        batch_config.quota_pool = POOL
        batch_config.reservation_chips = 8
        batch_handle = make_fake_slice_handle("batch1", scale_group="v4-8-batch", vm_states=[vm_pb2.VM_STATE_READY])
        batch_group = ScalingGroup(batch_config, make_mock_platform(slices_to_discover=[batch_handle]))
        batch_group.reconcile()
        batch_group.mark_slice_ready(batch_handle.slice_id, [w.worker_id for w in batch_handle.describe().workers])

        prod_config = make_scale_group_config(name="v4-16-prod", accelerator_variant="v4-16", max_slices=8)
        prod_config.quota_pool = POOL
        prod_config.reservation_chips = 8
        prod_group = ScalingGroup(prod_config, make_mock_platform())

        backend = _bound_backend(db, _StubAutoscaler([batch_group, prod_group]))

        schedule_only = backend.schedule(_request(autoscale_runs=False, pending_task_rows=pending))
        assert schedule_only.reserved_drain_workers == []

        with_autoscale = backend.schedule(_request(autoscale_runs=True, pending_task_rows=pending))
        assert with_autoscale.reserved_drain_workers == [WorkerId("batch1-vm-0")]
    finally:
        db.close()
