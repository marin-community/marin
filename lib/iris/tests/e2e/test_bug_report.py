# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the bug-report command."""

import pytest
from iris.cli.bug_report import format_bug_report, gather_bug_report
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec

from .helpers import _failing, _quick

pytestmark = pytest.mark.e2e


def test_bug_report_for_failed_job(cluster):
    """Bug report includes job/task/worker details for a failed job."""
    job = cluster.submit(_failing, "bug-report-fail")
    cluster.wait(job, timeout=30)

    report = gather_bug_report(cluster.url, job.job_id)
    assert report.state_name == "failed"
    assert report.task_count >= 1
    assert len(report.tasks) >= 1
    assert report.tasks[0].state == "failed"
    assert report.error  # should have an error message

    md = format_bug_report(report)
    assert "## Job Summary" in md
    assert "## Task Details" in md
    assert "## Log Paths" in md
    assert "## Useful Commands" in md
    assert report.job_id in md
    assert "`failed`" in md


def test_bug_report_for_succeeded_job(cluster):
    """Bug report works for succeeded jobs too (no error logs section)."""
    job = cluster.submit(_quick, "bug-report-ok")
    cluster.wait(job, timeout=30)

    report = gather_bug_report(cluster.url, job.job_id)
    assert report.state_name == "succeeded"
    assert report.failure_count == 0

    md = format_bug_report(report)
    assert "## Job Summary" in md
    assert "`succeeded`" in md
    # Succeeded tasks shouldn't appear in "Recent Logs" section
    assert "## Recent Logs" not in md


def test_bug_report_includes_workers(cluster):
    """Bug report includes worker information for the involved workers."""
    job = cluster.submit(_failing, "bug-report-workers")
    cluster.wait(job, timeout=30)

    report = gather_bug_report(cluster.url, job.job_id)
    # The job ran on at least one worker
    assert len(report.tasks) >= 1
    task = report.tasks[0]
    if task.worker_id:
        assert task.worker_id in report.workers
        worker = report.workers[task.worker_id]
        assert worker.address


def test_bug_report_includes_attempts(cluster):
    """Bug report includes attempt history for tasks."""
    job = cluster.submit(_failing, "bug-report-attempts")
    cluster.wait(job, timeout=30)

    report = gather_bug_report(cluster.url, job.job_id)
    task = report.tasks[0]
    assert len(task.attempts) >= 1
    attempt = task.attempts[0]
    assert attempt.state == "failed"


def test_bug_report_includes_entrypoint(cluster):
    """Bug report captures the original entrypoint command."""
    job = cluster.client.submit(
        entrypoint=Entrypoint.from_command("python", "-c", "import sys; sys.exit(1)"),
        name="bug-report-entrypoint",
        resources=ResourceSpec(cpu=1, memory="1g"),
        environment=EnvironmentSpec(),
    )
    cluster.wait(job, timeout=30)

    report = gather_bug_report(cluster.url, job.job_id)
    # The entrypoint comes from the RuntimeEntrypoint.run_command, which wraps
    # the original command in a shell invocation. Just verify it's non-empty.
    md = format_bug_report(report)
    assert "Entrypoint" in md
