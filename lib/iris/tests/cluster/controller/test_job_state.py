# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`iris.cluster.controller.job_state`.

Two pillars:

* ``compute_job_state`` parity table — pin the formula's outputs against a
  representative grid of ``(counts, max_task_failures, started_at_ms)``
  combinations. The expected column reproduces the eight-arm cascade by
  hand so any drift in the SQL view, the pure function, or the controller
  recompute path lights up here.
* ``jobs_with_state`` view smoke — insert a job + tasks via the live store
  API, then assert that ``SELECT state FROM jobs_with_state WHERE job_id``
  returns the same value as ``JobStore.read_state`` for every job state in
  the parity table.
"""

from __future__ import annotations

import pytest
from iris.cluster.controller.job_state import compute_job_state
from iris.cluster.controller.stores import JobConfigInsertParams, JobInsertParams, TaskInsertParams
from iris.cluster.types import JobName
from iris.rpc import job_pb2

# Shorthand for the parametrize ids / table.
S = job_pb2
PENDING = S.TASK_STATE_PENDING
BUILDING = S.TASK_STATE_BUILDING
RUNNING = S.TASK_STATE_RUNNING
SUCCEEDED = S.TASK_STATE_SUCCEEDED
FAILED = S.TASK_STATE_FAILED
KILLED = S.TASK_STATE_KILLED
WORKER_FAILED = S.TASK_STATE_WORKER_FAILED
UNSCHEDULABLE = S.TASK_STATE_UNSCHEDULABLE
ASSIGNED = S.TASK_STATE_ASSIGNED
PREEMPTED = S.TASK_STATE_PREEMPTED

JOB_RUNNING = S.JOB_STATE_RUNNING
JOB_PENDING = S.JOB_STATE_PENDING
JOB_SUCCEEDED = S.JOB_STATE_SUCCEEDED
JOB_FAILED = S.JOB_STATE_FAILED
JOB_KILLED = S.JOB_STATE_KILLED
JOB_WORKER_FAILED = S.JOB_STATE_WORKER_FAILED
JOB_UNSCHEDULABLE = S.JOB_STATE_UNSCHEDULABLE


# ``(label, counts, max_task_failures, started_at_ms, expected_job_state)``.
# Each row tests one branch of the cascade. The cascade order is significant:
# SUCCEEDED beats KILLED, KILLED beats WORKER_FAILED, etc.
PARITY_CASES: list[tuple[str, dict[int, int], int, int | None, int]] = [
    # All-succeeded → SUCCEEDED, even if started_at_ms is unset.
    ("all_succeeded", {SUCCEEDED: 3}, 0, None, JOB_SUCCEEDED),
    ("all_succeeded_with_started", {SUCCEEDED: 2}, 0, 1000, JOB_SUCCEEDED),
    # FAILED > budget → FAILED (beats KILLED/WORKER_FAILED/RUNNING below).
    ("failed_over_budget", {FAILED: 2, RUNNING: 1}, 1, 500, JOB_FAILED),
    ("failed_under_budget_running", {FAILED: 1, RUNNING: 1}, 1, 500, JOB_RUNNING),
    # UNSCHEDULABLE present (any count) → UNSCHEDULABLE, even with active siblings.
    ("any_unschedulable", {UNSCHEDULABLE: 1, RUNNING: 2}, 0, 500, JOB_UNSCHEDULABLE),
    # KILLED present → KILLED, dominates WORKER_FAILED / RUNNING.
    ("any_killed", {KILLED: 1, RUNNING: 1}, 0, 500, JOB_KILLED),
    ("killed_and_succeeded", {KILLED: 1, SUCCEEDED: 2}, 0, 500, JOB_KILLED),
    # All-terminal with at least one WORKER_FAILED or PREEMPTED, no failure/killed → WORKER_FAILED.
    ("all_terminal_worker_failed", {WORKER_FAILED: 1, SUCCEEDED: 2}, 0, 500, JOB_WORKER_FAILED),
    ("all_terminal_preempted", {PREEMPTED: 1, SUCCEEDED: 2}, 0, 500, JOB_WORKER_FAILED),
    # Active states → RUNNING.
    ("any_running", {RUNNING: 1, PENDING: 1}, 0, None, JOB_RUNNING),
    ("any_building", {BUILDING: 1, PENDING: 1}, 0, None, JOB_RUNNING),
    ("any_assigned", {ASSIGNED: 1, PENDING: 1}, 0, None, JOB_RUNNING),
    # No active task but started_at_ms set (retry-back-to-pending) → RUNNING.
    ("retry_pending_after_start", {PENDING: 2}, 0, 500, JOB_RUNNING),
    # No active task, never started, tasks exist → PENDING.
    ("only_pending_never_started", {PENDING: 3}, 0, None, JOB_PENDING),
    # FAILED at exactly the budget threshold is not yet a failure.
    ("failed_at_budget_pending", {FAILED: 1, PENDING: 1}, 1, None, JOB_PENDING),
    # WORKER_FAILED alone (terminal task) with no started_at_ms ⇒ WORKER_FAILED.
    ("worker_failed_only", {WORKER_FAILED: 2}, 0, None, JOB_WORKER_FAILED),
    # PREEMPTED only is terminal too — same branch as WORKER_FAILED above.
    ("preempted_only", {PREEMPTED: 2}, 0, None, JOB_WORKER_FAILED),
    # Mixed terminal + non-terminal (PENDING) blocks the WORKER_FAILED arm;
    # falls through to the active-states / started_at_ms / PENDING tail.
    ("worker_failed_plus_pending_started", {WORKER_FAILED: 1, PENDING: 1}, 0, 500, JOB_RUNNING),
    ("worker_failed_plus_pending_no_start", {WORKER_FAILED: 1, PENDING: 1}, 0, None, JOB_PENDING),
]


@pytest.mark.parametrize(
    ("label", "counts", "max_task_failures", "started_at_ms", "expected"),
    PARITY_CASES,
    ids=[case[0] for case in PARITY_CASES],
)
def test_compute_job_state_parity(label, counts, max_task_failures, started_at_ms, expected):
    """``compute_job_state`` produces the hand-pinned expected state for every case."""
    assert compute_job_state(counts, max_task_failures=max_task_failures, started_at_ms=started_at_ms) == expected, label


def test_compute_job_state_empty_counts():
    """No tasks (yet): PENDING when never started, RUNNING once started_at_ms is stamped."""
    assert compute_job_state({}, max_task_failures=0, started_at_ms=None) == JOB_PENDING
    # ``started_at_ms`` is set ⇒ the "retry put everything back to PENDING but
    # we've been running" branch fires even when counts are empty. The
    # transitions wrapper short-circuits this case so it never writes anyway.
    assert compute_job_state({}, max_task_failures=0, started_at_ms=1) == JOB_RUNNING


# =============================================================================
# View smoke: ``jobs_with_state`` agrees with ``JobStore.read_state``.
# =============================================================================


def _insert_job_with_tasks(
    state,
    job_wire: str,
    *,
    counts: dict[int, int],
    max_task_failures: int,
    started_at_ms: int | None,
) -> JobName:
    """Insert a synthetic job and one task per ``count`` entry; return JobName.

    The job/task rows are inserted via the store API so the schema definition
    (including the ``jobs_with_state`` view) is exercised exactly as the
    controller would.
    """
    store = state._store
    job_id = JobName.from_wire(job_wire)
    now_ms = 100
    with store.transaction() as cur:
        store.jobs.ensure_user(cur, "user-test", now_ms)
        store.jobs.insert(
            cur,
            JobInsertParams(
                job_id=job_id,
                user_id="user-test",
                parent_job_id=None,
                root_job_id=job_wire,
                depth=0,
                state=job_pb2.JOB_STATE_PENDING,
                submitted_at_ms=now_ms,
                root_submitted_at_ms=now_ms,
                started_at_ms=started_at_ms,
                finished_at_ms=None,
                scheduling_deadline_epoch_ms=None,
                error=None,
                exit_code=None,
                num_tasks=sum(counts.values()),
                is_reservation_holder=False,
                name=job_wire,
                has_reservation=False,
            ),
        )
        store.jobs.insert_config(
            cur,
            JobConfigInsertParams(
                job_id=job_id,
                name=job_wire,
                has_reservation=False,
                res_cpu_millicores=1000,
                res_memory_bytes=1024,
                res_disk_bytes=0,
                res_device_json=None,
                constraints_json="[]",
                has_coscheduling=False,
                coscheduling_group_by="",
                scheduling_timeout_ms=None,
                max_task_failures=max_task_failures,
                entrypoint_json="{}",
                environment_json="{}",
                bundle_id="",
                ports_json="[]",
                max_retries_failure=0,
                max_retries_preemption=0,
                timeout_ms=None,
                preemption_policy=0,
                existing_job_policy=0,
                priority_band=2,
                task_image="",
                submit_argv_json="[]",
                reservation_json=None,
                fail_if_exists=False,
            ),
        )
        task_idx = 0
        for task_state, count in counts.items():
            for _ in range(count):
                task_wire = f"{job_wire}/task-{task_idx}"
                store.tasks.insert(
                    cur,
                    TaskInsertParams(
                        task_id=JobName.from_wire(task_wire),
                        job_id=job_id,
                        task_index=task_idx,
                        state=task_state,
                        submitted_at_ms=now_ms,
                        max_retries_failure=0,
                        max_retries_preemption=0,
                        priority_neg_depth=0,
                        priority_root_submitted_ms=now_ms,
                        priority_insertion=task_idx,
                        priority_band=2,
                    ),
                )
                task_idx += 1
    return job_id


# Subset of the parity table whose counts are non-empty (the view requires
# at least one task row to materialize a non-default state).
_VIEW_CASES = [c for c in PARITY_CASES if sum(c[1].values()) > 0]


@pytest.mark.parametrize(
    ("label", "counts", "max_task_failures", "started_at_ms", "expected"),
    _VIEW_CASES,
    ids=[case[0] for case in _VIEW_CASES],
)
def test_jobs_with_state_view_smoke(state, label, counts, max_task_failures, started_at_ms, expected):
    """``SELECT state FROM jobs_with_state`` matches the pure function."""
    job_id = _insert_job_with_tasks(
        state,
        f"/user-test/job-{label}",
        counts=counts,
        max_task_failures=max_task_failures,
        started_at_ms=started_at_ms,
    )

    store = state._store
    with store.read_snapshot() as snap:
        view_row = snap.fetchone(
            "SELECT state FROM jobs_with_state WHERE job_id = ?",
            (job_id.to_wire(),),
        )
        read_state_result = store.jobs.read_state(snap, job_id)

    assert view_row is not None, label
    assert int(view_row["state"]) == expected, label
    assert read_state_result == expected, label


def test_read_state_missing_job(state):
    """``read_state`` returns ``None`` for an unknown job id."""
    store = state._store
    with store.read_snapshot() as snap:
        assert store.jobs.read_state(snap, JobName.from_wire("/user-test/does-not-exist")) is None
