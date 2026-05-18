# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-function unit tests for reconcile_worker.

Each test constructs WorkerReconcileInputs by hand (no DB, no SQLite) and
asserts on the resulting WorkerReconcilePlan. Tests cover the row-axis of the
§5.3 dispatch matrix: what the pure function emits for each DB task state.

Worker-observation column effects (AttemptMissingOnWorker, state transitions)
are NOT tested here — those belong to the apply-layer tests in A.4.
"""

import pytest
from iris.cluster.controller.db import (
    TASK_STATE_KILLED,
    TASK_STATE_PREEMPTED,
)
from iris.cluster.controller.reconcile import (
    ReconcileRow,
    StopReason,
    WorkerReconcileInputs,
    WorkerRow,
    reconcile_worker,
)
from iris.cluster.types import JobName, WorkerId
from iris.rpc import job_pb2
from rigging.timing import Timestamp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_W1 = WorkerId("worker-1")
_JOB1 = JobName.from_string("/alice/job1")
_JOB2 = JobName.from_string("/alice/job2")
_TASK1 = JobName.from_string("/alice/job1/0")
_TASK2 = JobName.from_string("/alice/job1/1")
_NOW = Timestamp(1_000_000_000)
_SPEC1 = object()  # opaque spec object; pure function treats it as Any


def _worker(worker_id: WorkerId = _W1) -> WorkerRow:
    return WorkerRow(
        worker_id=worker_id,
        address="localhost:9999",
        total_cpu_millicores=4000,
        total_memory_bytes=8 * 1024**3,
        total_gpu_count=0,
        total_tpu_count=0,
        device_type="cpu",
        device_variant="",
    )


def _row(
    *,
    task_id: JobName = _TASK1,
    attempt_id: int = 1,
    task_state: int,
    job_id: JobName = _JOB1,
    worker_id: WorkerId = _W1,
) -> ReconcileRow:
    return ReconcileRow(
        worker_id=worker_id,
        task_id=task_id,
        attempt_id=attempt_id,
        task_state=task_state,
        attempt_state=task_state,
        job_id=job_id,
    )


def _inputs(
    rows: list[ReconcileRow],
    job_specs: dict[JobName, object] | None = None,
    worker: WorkerRow | None = None,
) -> WorkerReconcileInputs:
    return WorkerReconcileInputs(
        worker=worker or _worker(),
        rows=rows,
        job_specs=job_specs if job_specs is not None else {_JOB1: _SPEC1},
        now=_NOW,
    )


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------


def test_empty_rows_produces_empty_plan() -> None:
    plan = reconcile_worker(_inputs(rows=[]))
    assert plan.request.desired == []
    assert plan.db_writes == []


# ---------------------------------------------------------------------------
# ASSIGNED row — spec inline
# ---------------------------------------------------------------------------


def test_assigned_spec_inline() -> None:
    plan = reconcile_worker(_inputs(rows=[_row(task_state=job_pb2.TASK_STATE_ASSIGNED)], job_specs={_JOB1: _SPEC1}))
    assert len(plan.request.desired) == 1
    da = plan.request.desired[0]
    assert da.intent_run is not None
    assert da.intent_run.request is _SPEC1
    assert da.intent_stop is None
    assert da.task_id == _TASK1.to_wire()
    assert da.attempt_id == 1
    assert plan.db_writes == []


def test_assigned_with_missing_spec_skips() -> None:
    plan = reconcile_worker(
        _inputs(
            rows=[_row(task_state=job_pb2.TASK_STATE_ASSIGNED, job_id=_JOB2)],
            job_specs={},  # job2 not in map
        )
    )
    assert plan.request.desired == []


def test_assigned_spec_none_in_map_skips() -> None:
    """job_specs entry explicitly None (reservation holder) → skip."""
    plan = reconcile_worker(
        _inputs(
            rows=[_row(task_state=job_pb2.TASK_STATE_ASSIGNED)],
            job_specs={_JOB1: None},  # type: ignore[dict-item]
        )
    )
    assert plan.request.desired == []


# ---------------------------------------------------------------------------
# BUILDING row — run intent, spec omitted
# ---------------------------------------------------------------------------


def test_building_no_spec() -> None:
    plan = reconcile_worker(_inputs(rows=[_row(task_state=job_pb2.TASK_STATE_BUILDING)]))
    assert len(plan.request.desired) == 1
    da = plan.request.desired[0]
    assert da.intent_run is not None
    assert da.intent_run.request is None
    assert da.intent_stop is None
    assert plan.db_writes == []


# ---------------------------------------------------------------------------
# RUNNING row — run intent, spec omitted
# ---------------------------------------------------------------------------


def test_running_no_spec() -> None:
    plan = reconcile_worker(_inputs(rows=[_row(task_state=job_pb2.TASK_STATE_RUNNING)]))
    assert len(plan.request.desired) == 1
    da = plan.request.desired[0]
    assert da.intent_run is not None
    assert da.intent_run.request is None
    assert da.intent_stop is None
    assert plan.db_writes == []


# ---------------------------------------------------------------------------
# CANCELLED (KILLED) row — stop intent
# ---------------------------------------------------------------------------


def test_cancelled_emits_stop() -> None:
    plan = reconcile_worker(_inputs(rows=[_row(task_state=TASK_STATE_KILLED)]))
    assert len(plan.request.desired) == 1
    da = plan.request.desired[0]
    assert da.intent_stop == StopReason.CANCELLED
    assert da.intent_run is None
    assert plan.db_writes == []


# ---------------------------------------------------------------------------
# PREEMPTED row — stop intent
# ---------------------------------------------------------------------------


def test_preempted_emits_stop() -> None:
    plan = reconcile_worker(_inputs(rows=[_row(task_state=TASK_STATE_PREEMPTED)]))
    assert len(plan.request.desired) == 1
    da = plan.request.desired[0]
    assert da.intent_stop == StopReason.PREEMPTED
    assert da.intent_run is None
    assert plan.db_writes == []


# ---------------------------------------------------------------------------
# Terminal DB states — omitted from desired
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "terminal_state",
    [
        job_pb2.TASK_STATE_SUCCEEDED,
        job_pb2.TASK_STATE_FAILED,
        job_pb2.TASK_STATE_WORKER_FAILED,
        job_pb2.TASK_STATE_UNSCHEDULABLE,
        job_pb2.TASK_STATE_COSCHED_FAILED,
    ],
)
def test_terminal_omitted(terminal_state: int) -> None:
    plan = reconcile_worker(_inputs(rows=[_row(task_state=terminal_state)]))
    assert plan.request.desired == []
    assert plan.db_writes == []


# ---------------------------------------------------------------------------
# worker_id propagated to request
# ---------------------------------------------------------------------------


def test_worker_id_in_request() -> None:
    plan = reconcile_worker(_inputs(rows=[], worker=_worker(WorkerId("w-xyz"))))
    assert plan.request.worker_id == "w-xyz"


# ---------------------------------------------------------------------------
# Multiple rows — each dispatched independently
# ---------------------------------------------------------------------------


def test_multiple_rows_mixed_states() -> None:
    rows = [
        _row(task_id=_TASK1, attempt_id=1, task_state=job_pb2.TASK_STATE_ASSIGNED),
        _row(task_id=_TASK2, attempt_id=2, task_state=job_pb2.TASK_STATE_RUNNING),
    ]
    plan = reconcile_worker(_inputs(rows=rows, job_specs={_JOB1: _SPEC1}))
    assert len(plan.request.desired) == 2
    by_attempt = {da.attempt_id: da for da in plan.request.desired}
    assert by_attempt[1].intent_run is not None
    assert by_attempt[1].intent_run.request is _SPEC1  # ASSIGNED → spec inline
    assert by_attempt[2].intent_run is not None
    assert by_attempt[2].intent_run.request is None  # RUNNING → no spec


def test_multiple_rows_terminal_filtered() -> None:
    rows = [
        _row(task_id=_TASK1, attempt_id=1, task_state=job_pb2.TASK_STATE_RUNNING),
        _row(task_id=_TASK2, attempt_id=2, task_state=job_pb2.TASK_STATE_SUCCEEDED),
    ]
    plan = reconcile_worker(_inputs(rows=rows))
    assert len(plan.request.desired) == 1
    assert plan.request.desired[0].attempt_id == 1


@pytest.mark.parametrize(
    "state,expect_spec_inline",
    [
        (job_pb2.TASK_STATE_ASSIGNED, True),
        (job_pb2.TASK_STATE_BUILDING, False),
        (job_pb2.TASK_STATE_RUNNING, False),
    ],
)
def test_spec_inline_only_for_assigned(state: int, expect_spec_inline: bool) -> None:
    plan = reconcile_worker(_inputs(rows=[_row(task_state=state)], job_specs={_JOB1: _SPEC1}))
    da = plan.request.desired[0]
    assert da.intent_run is not None
    if expect_spec_inline:
        assert da.intent_run.request is _SPEC1
    else:
        assert da.intent_run.request is None


@pytest.mark.parametrize(
    "state,expected_reason",
    [
        (TASK_STATE_KILLED, StopReason.CANCELLED),
        (TASK_STATE_PREEMPTED, StopReason.PREEMPTED),
    ],
)
def test_stop_reason_mapping(state: int, expected_reason: StopReason) -> None:
    plan = reconcile_worker(_inputs(rows=[_row(task_state=state)]))
    assert plan.request.desired[0].intent_stop == expected_reason


def test_attempt_compat_fields_populated() -> None:
    """task_id and attempt_id compat fields are set for Phase-B translator."""
    plan = reconcile_worker(
        _inputs(
            rows=[_row(task_id=_TASK1, attempt_id=42, task_state=job_pb2.TASK_STATE_RUNNING)],
        )
    )
    da = plan.request.desired[0]
    assert da.task_id == _TASK1.to_wire()
    assert da.attempt_id == 42
