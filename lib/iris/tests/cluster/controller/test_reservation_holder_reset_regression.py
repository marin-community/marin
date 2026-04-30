# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression test for reservation-holder reset misapplied to non-holder tasks.

Bug (iris-outage 2026-04): when a worker hosting both a reservation-holder task
and one or more non-holder tasks fails, `_remove_failed_worker` sometimes
applies the holder-reset branch (DELETE from task_attempts, preemption_count=0,
started_at_ms=NULL, current_attempt_id=-1) to NON-holder tasks.

Observable production fallout: dashboard shows zero attempts / no run history
for tasks that clearly ran; worker attribution lost; preemption_count wiped.

This test pins the correct behavior: a non-holder task that was running on a
failed worker must go through `_terminate_task` (preemption_count increments,
attempt row is preserved with a terminal state, not DELETEd). See
transitions.py:2030-2111.
"""

from iris.cluster.controller.transitions import (
    Assignment,
    HeartbeatApplyRequest,
    RESERVATION_HOLDER_JOB_NAME,
    TaskUpdate,
)
from iris.rpc import job_pb2

from tests.cluster.controller.conftest import (
    fail_worker,
    query_task,
    query_task_with_attempts,
    query_tasks_for_job,
)
from tests.cluster.controller.conftest import (
    make_job_request,
    submit_job,
)
from tests.cluster.controller.test_reservation import (
    _make_job_request_with_reservation,
    _make_reservation_entry,
    _register_worker,
    _submit_job,
)


def test_non_holder_task_not_reset_like_reservation_holder_on_worker_failure(state):
    """Regression: non-holder task co-located with holder on a failed worker
    must NOT take the holder-reset branch in `_remove_failed_worker`.

    The holder-reset branch:
      - DELETEs the task's current attempt row
      - zeroes preemption_count
      - NULLs started_at_ms, current_worker_id, current_worker_address
      - sets current_attempt_id = -1

    For a non-holder task the correct path is `_terminate_task` with a
    preemption_count increment and a preserved (terminal) attempt row.
    """
    request = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry()],
    )
    # Give the non-holder task retry budget so WORKER_FAILED requeues to PENDING
    # (preemption path), which makes preemption_count observable.
    request.max_retries_preemption = 5
    parent_job_id = _submit_job(state, "res-job", request)
    holder_job_id = parent_job_id.child(RESERVATION_HOLDER_JOB_NAME)

    holder_tasks = query_tasks_for_job(state, holder_job_id)
    parent_tasks = query_tasks_for_job(state, parent_job_id)
    assert len(holder_tasks) == 1
    assert len(parent_tasks) == 1
    holder_task = holder_tasks[0]
    parent_task = parent_tasks[0]

    # Co-locate holder + non-holder on the same worker. Holder tasks legitimately
    # don't consume real resources, so this pairing is a supported configuration
    # (transitions.py:1589, 1364).
    # A second, unrelated non-holder root job also lands on the same worker.
    # In production the bug manifests across jobs, matching the pattern seen in
    # controller.sqlite3 for worker `0046-7af1c6c8-worker-0`: three active
    # tasks spanning two root jobs, one of which was the holder.
    other_req = make_job_request("other-job")
    other_req.max_retries_preemption = 5
    other_tasks = submit_job(state, "other-job", other_req)
    other_task = other_tasks[0]

    worker_id = _register_worker(state, "co-located-worker")
    with state._store.transaction() as cur:
        state.queue_assignments(
            cur,
            [
                Assignment(task_id=holder_task.task_id, worker_id=worker_id),
                Assignment(task_id=parent_task.task_id, worker_id=worker_id),
                Assignment(task_id=other_task.task_id, worker_id=worker_id),
            ],
        )

    # Move the non-holder task to RUNNING so WORKER_FAILED counts as a
    # preemption (not a delivery failure). This matches the production shape:
    # task had actually executed on the worker before it died.
    with state._store.transaction() as cur:
        state.apply_task_updates(
            cur,
            HeartbeatApplyRequest(
                worker_id=worker_id,
                worker_resource_snapshot=None,
                updates=[
                    TaskUpdate(
                        task_id=parent_task.task_id,
                        attempt_id=query_task(state, parent_task.task_id).current_attempt_id,
                        new_state=job_pb2.TASK_STATE_RUNNING,
                    ),
                    TaskUpdate(
                        task_id=other_task.task_id,
                        attempt_id=query_task(state, other_task.task_id).current_attempt_id,
                        new_state=job_pb2.TASK_STATE_RUNNING,
                    ),
                ],
            ),
        )

    # Capture the non-holder task's pre-failure attempt state so we can
    # assert on preservation.
    before = query_task_with_attempts(state, parent_task.task_id)
    assert before is not None
    assert before.state == job_pb2.TASK_STATE_RUNNING
    pre_attempt_id = before.current_attempt_id
    assert pre_attempt_id >= 0, "non-holder task should have a real attempt"
    assert len(before.attempts) == 1, "non-holder task should have one attempt row"

    # Also double-check the holder really is a holder at the DB level.
    holder_before = query_task_with_attempts(state, holder_task.task_id)
    assert holder_before is not None

    # Trigger the worker failure codepath -> _remove_failed_worker.
    fail_worker(state, worker_id, "simulated crash")

    # --- Assertions on BOTH non-holder tasks ---
    # In production the cross-job non-holder is the observed victim (53 rows in
    # controller.sqlite3), so check both.
    for victim_label, victim_task_id in (
        ("same-job-non-holder", parent_task.task_id),
        ("cross-job-non-holder", other_task.task_id),
    ):
        after = query_task_with_attempts(state, victim_task_id)
        assert after is not None, f"{victim_label}: task missing after worker failure"

        # The task's attempt row must still exist. The holder-branch DELETE
        # would have removed it.
        assert len(after.attempts) == 1, (
            f"{victim_label}: task_attempts row was deleted "
            f"(len={len(after.attempts)}); holder-reset branch was misapplied"
        )
        # current_attempt_id must NOT be -1 (holder-branch marker).
        assert after.current_attempt_id != -1, (
            f"{victim_label}: current_attempt_id was reset to -1; " "holder-reset branch was misapplied"
        )
        # preemption_count must be > 0 (incremented from RUNNING -> WORKER_FAILED).
        assert after.preemption_count >= 1, (
            f"{victim_label}: preemption_count was zeroed "
            f"(={after.preemption_count}); holder-reset branch was misapplied"
        )
        # started_at_ms must NOT be wiped to NULL.
        assert after.started_at is not None, (
            f"{victim_label}: started_at was NULLed; " "holder-reset branch was misapplied"
        )
    # Also verify the same-job surviving attempt kept its id.
    after_same = query_task_with_attempts(state, parent_task.task_id)
    assert after_same is not None
    surviving_attempt = after_same.attempts[0]
    assert surviving_attempt.attempt_id == pre_attempt_id, (
        f"surviving attempt_id changed: {surviving_attempt.attempt_id} vs " f"pre-failure {pre_attempt_id}"
    )


def test_resubmitted_task_does_not_inherit_prior_worker_task_history(state):
    """Regression (iris-outage 2026-04-16): ``worker_task_history.task_id`` has
    no foreign key to ``tasks.task_id``. When a finished job is removed via
    ``remove_finished_job`` the FK-cascade deletes the job's ``tasks`` and
    ``task_attempts`` rows — but orphans its ``worker_task_history`` rows. If
    the same job name is then re-submitted, the newly inserted task rows
    (schema defaults: ``current_attempt_id=-1``, ``started_at_ms=NULL``,
    ``preemption_count=0``, ``failure_count=0``) are silently re-attached to
    the stale history rows.

    In production this fingerprint was mis-diagnosed as the reservation-holder
    reset branch (transitions.py:2063-2073) misfiring on non-holder tasks.
    Forensics on the outage DB showed 106/106 "victim" tasks had
    ``submitted_at_ms`` *strictly later than* the max ``assigned_at_ms`` of
    their attached history rows, and 0/106 shared a worker with a reservation
    holder — i.e. the history was orphaned from a prior incarnation of the
    same job name, not written against the current row.

    This test pins the intended invariant: a task's ``worker_task_history`` set
    must only reflect its own lifetime. A freshly re-submitted task must not
    appear to have run on a worker that hosted its predecessor.
    """
    # === Round 1: submit, assign, cancel, remove ===================
    request = make_job_request("reusable-job-name", max_retries_preemption=0)
    tasks = submit_job(state, "reusable-job-name", request)
    assert len(tasks) == 1
    task_v1 = tasks[0]
    job_id = task_v1.task_id.parent

    worker_id = _register_worker(state, "history-worker")
    # queue_assignments inserts worker_task_history(worker_id, task_id, ...).
    with state._store.transaction() as cur:
        state.queue_assignments(cur, [Assignment(task_id=task_v1.task_id, worker_id=worker_id)])
    # Cancel to terminal, then remove the finished job. The job's tasks + attempts
    # cascade; worker_task_history does NOT (no FK on task_id).
    with state._store.transaction() as cur:
        state.cancel_job(cur, job_id, "Terminated by user")
    with state._store.transaction() as cur:
        assert state.remove_finished_job(cur, job_id) is True

    # === Round 2: re-submit the same job name ========================
    request2 = make_job_request("reusable-job-name", max_retries_preemption=0)
    tasks2 = submit_job(state, "reusable-job-name", request2)
    assert len(tasks2) == 1
    task_v2 = tasks2[0]
    # Sanity: the two incarnations share the wire task_id.
    assert task_v2.task_id.to_wire() == task_v1.task_id.to_wire()

    # The fresh task must not be silently tagged with its predecessor's
    # worker_task_history. A user inspecting this task expects "never ran".
    with state._db.snapshot() as q:
        orphan_rows = q.fetchall(
            "SELECT worker_id, assigned_at_ms FROM worker_task_history WHERE task_id = ?",
            (task_v2.task_id.to_wire(),),
        )
    assert len(orphan_rows) == 0, (
        f"newly-submitted task {task_v2.task_id.to_wire()} inherited "
        f"{len(orphan_rows)} stale worker_task_history row(s) from its prior "
        f"incarnation; this is the misread that produced the 106 'reset' "
        f"victims in controller.sqlite3: the default-valued task row "
        f"(current_attempt_id=-1, started_at_ms=NULL, preemption_count=0) "
        f"appears as a holder-reset victim when paired with orphan history."
    )

    # Cross-check the task row is in its pristine as-submitted state — this
    # exactly matches the 'victim' fingerprint seen in production:
    after = query_task_with_attempts(state, task_v2.task_id)
    assert after is not None
    assert after.current_attempt_id == -1
    assert after.started_at is None
    assert after.preemption_count == 0
    assert after.failure_count == 0
    assert len(after.attempts) == 0
    # The combination of the above with orphan history is what made these rows
    # indistinguishable from holder-reset victims.


def test_reservation_holder_task_is_still_reset_on_worker_failure(state):
    """Control: the holder task itself MUST still get the reset treatment.

    Ensures the regression test above doesn't inadvertently pass a fix that
    removes the holder-reset behavior entirely.
    """
    request = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry()],
    )
    parent_job_id = _submit_job(state, "res-job", request)
    holder_job_id = parent_job_id.child(RESERVATION_HOLDER_JOB_NAME)
    holder_task = query_tasks_for_job(state, holder_job_id)[0]

    worker_id = _register_worker(state, "w-holder-only")
    with state._store.transaction() as cur:
        state.queue_assignments(cur, [Assignment(task_id=holder_task.task_id, worker_id=worker_id)])

    fail_worker(state, worker_id, "crash")

    after = query_task(state, holder_task.task_id)
    assert after is not None
    assert after.state == job_pb2.TASK_STATE_PENDING
    assert after.preemption_count == 0


def test_reservation_holder_reassignment_across_successive_worker_failures(state):
    """Regression (iris-outage 2026-04): the old holder-reset branch reset
    ``tasks.current_attempt_id = -1`` while only DELETing the single current
    attempt row. Across repeated worker failures this left orphan attempt rows
    in ``task_attempts`` whose primary key collided with the next
    assignment attempt insert, raising ``sqlite3.IntegrityError`` and killing the
    scheduling thread.

    The fix routes holders through ``_terminate_task``, so the attempt row is
    preserved in a terminal state and ``current_attempt_id`` advances
    monotonically. This test fails holder+non-holder reservation tasks across
    three successive workers and pins the advancing-attempt invariant.
    """
    request = _make_job_request_with_reservation(
        reservation_entries=[_make_reservation_entry()],
    )
    parent_job_id = _submit_job(state, "res-job", request)
    holder_job_id = parent_job_id.child(RESERVATION_HOLDER_JOB_NAME)
    holder_task = query_tasks_for_job(state, holder_job_id)[0]
    non_holder_task = query_tasks_for_job(state, parent_job_id)[0]

    expected_attempts = 0
    for cycle_idx, worker_name in enumerate(("w-res-1", "w-res-2", "w-res-3")):
        worker_id = _register_worker(state, worker_name)
        # Assignment issued by scheduler: attempt_id = current_attempt_id + 1.
        with state._store.transaction() as cur:
            state.queue_assignments(
                cur,
                [
                    Assignment(task_id=holder_task.task_id, worker_id=worker_id),
                    Assignment(task_id=non_holder_task.task_id, worker_id=worker_id),
                ],
            )
        expected_attempts += 1

        holder_assigned = query_task_with_attempts(state, holder_task.task_id)
        assert holder_assigned is not None
        assert holder_assigned.current_attempt_id == cycle_idx, (
            f"cycle {cycle_idx}: holder current_attempt_id did not advance "
            f"monotonically (got {holder_assigned.current_attempt_id}, "
            f"expected {cycle_idx})"
        )
        assert len(holder_assigned.attempts) == expected_attempts, (
            f"cycle {cycle_idx}: expected {expected_attempts} attempt rows on "
            f"holder, got {len(holder_assigned.attempts)}; orphan/missing rows "
            f"indicate the legacy holder-reset branch is still live"
        )

        non_holder_assigned = query_task_with_attempts(state, non_holder_task.task_id)
        assert non_holder_assigned is not None
        assert non_holder_assigned.state == job_pb2.TASK_STATE_ASSIGNED, (
            f"cycle {cycle_idx}: non-holder was not scheduled onto the new worker "
            f"(state={non_holder_assigned.state}); holder failure left the "
            f"reservation in an unschedulable state"
        )

        # Fail the worker. On old code the third pass raises IntegrityError
        # from queue_assignments above (not here); we assert no exception
        # propagates through this call either.
        fail_worker(state, worker_id, f"simulated crash {cycle_idx}")

        holder_after_fail = query_task_with_attempts(state, holder_task.task_id)
        assert holder_after_fail is not None
        assert holder_after_fail.state == job_pb2.TASK_STATE_PENDING
        # preemption_count for holders stays at 0: the fix preserves the
        # reset-preemption-count semantic for reservation holders so they can
        # re-acquire a worker without hitting retry caps.
        assert holder_after_fail.preemption_count == 0
        # Attempt rows must be preserved across the failure (terminal state
        # written, no DELETE). The old branch deleted the current attempt row.
        assert len(holder_after_fail.attempts) == expected_attempts, (
            f"cycle {cycle_idx}: holder attempt rows were DELETEd by the "
            f"legacy holder-reset branch (have {len(holder_after_fail.attempts)}, "
            f"expected {expected_attempts})"
        )
        latest_attempt = max(holder_after_fail.attempts, key=lambda a: a.attempt_id)
        assert latest_attempt.state == job_pb2.TASK_STATE_WORKER_FAILED, (
            f"cycle {cycle_idx}: latest holder attempt should be marked "
            f"WORKER_FAILED, got state={latest_attempt.state}"
        )

    # Final invariant: three cycles produced attempts 0,1,2 and the holder is
    # PENDING again, ready for the next scheduling pass without any orphan
    # task_attempts row that would collide on INSERT.
    final = query_task_with_attempts(state, holder_task.task_id)
    assert final is not None
    attempt_ids = sorted(a.attempt_id for a in final.attempts)
    assert attempt_ids == [0, 1, 2], f"holder attempt_id sequence was not monotonic across failures: " f"{attempt_ids}"
