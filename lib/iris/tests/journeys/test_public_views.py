# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Public list and diagnostic views over completed journeys."""

import logging

from iris.rpc import job_pb2


def test_list_jobs_filters_terminal_and_pending_jobs_and_cancel_finished_is_noop(journey):
    finished = journey.submit("finished")
    journey.settle()
    journey.succeed(finished[0])
    journey.settle()
    pending = journey.submit("pending")

    assert [job.identity.key.resource_id for job in journey.jobs(state="succeeded")] == [finished.wire_id]
    assert [job.identity.key.resource_id for job in journey.jobs(state="pending")] == [pending.wire_id]

    journey.cancel(finished)

    assert journey.job(finished).summary.state == job_pb2.JOB_STATE_SUCCEEDED


def test_failed_task_status_distills_root_cause_from_real_log_transport(journey):
    job = journey.submit("root-cause")
    journey.settle()
    journey.fail(job[0], error="application failed")
    journey.settle()
    journey.push_task_logs(
        job[0],
        [
            " 50%|#####     | 500/1000 [00:10<00:10, 5.0it/s]",
            "Traceback (most recent call last):",
            "RuntimeError: CUDA error: an illegal memory access was encountered",
        ],
    )

    detail = journey.task(job[0])

    assert detail.summary.state == job_pb2.TASK_STATE_FAILED
    assert "RuntimeError: CUDA error: an illegal memory access was encountered" in detail.root_cause_highlights
    assert not any("500/1000" in line for line in detail.root_cause_highlights)


def test_succeeded_task_status_omits_failure_highlights_even_with_error_like_logs(journey):
    job = journey.submit("clean-status")
    journey.settle()
    journey.succeed(job[0])
    journey.settle()
    journey.push_task_logs(job[0], ["RuntimeError: stale text from a successful task"])

    detail = journey.task(job[0])

    assert detail.summary.state == job_pb2.TASK_STATE_SUCCEEDED
    assert detail.root_cause_highlights == ()


def test_failed_task_status_survives_unavailable_log_transport(journey, monkeypatch, caplog):
    job = journey.submit("unavailable-root-cause")
    journey.settle()
    journey.fail(job[0], error="application failed")
    journey.settle()

    def unavailable(_request):
        raise ConnectionError("finelog unavailable")

    monkeypatch.setattr(journey.log_stack.client, "fetch_logs", unavailable)
    with caplog.at_level(logging.WARNING, logger="iris.cluster.controller.controller"):
        detail = journey.task(job[0])

    assert detail.root_cause_highlights == ()
    finelog_statuses = [status for status in detail.source_statuses if status.source_id.startswith("finelog:")]
    assert len(finelog_statuses) == 1
    assert finelog_statuses[0].error_code == "finelog_unavailable"
    assert finelog_statuses[0].error_message == "finelog unavailable"
    assert any(
        record.name == "iris.cluster.controller.controller" and record.levelno == logging.WARNING
        for record in caplog.records
    )


def test_list_tasks_reports_current_timing_and_detail_keeps_attempt_history(journey):
    job = journey.submit("bounded-attempt-list", failure_retries=2)
    journey.settle()
    journey.fail(job[0], error="first failure")
    journey.settle()
    journey.fail(job[0], error="latest failure")
    journey.settle()

    (listed,) = journey.tasks(job)
    detail = journey.task(job[0])

    assert listed.state == job_pb2.TASK_STATE_RUNNING
    assert listed.started_at.epoch_ms() > 0
    assert listed.current_attempt is not None
    assert listed.current_attempt.attempt_number == 2
    assert listed.started_at is not None
    assert [attempt.identity.attempt_number for attempt in detail.attempts] == [0, 1, 2]
    assert detail.attempts[1].error_message == "latest failure"
    assert detail.attempts[-1].started_at == listed.started_at
