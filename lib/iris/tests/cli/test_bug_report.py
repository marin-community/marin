# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for bug-report log-highlight extraction and rendering."""

from iris.cli.bug_report import BugReport, TaskReport, _build_task_report, format_bug_report
from iris.rpc import job_pb2


def _task_status(task_id: str = "/u/j/0", state: int = job_pb2.TASK_STATE_FAILED) -> job_pb2.TaskStatus:
    return job_pb2.TaskStatus(task_id=task_id, state=state)


def _report(tasks: list[TaskReport]) -> BugReport:
    return BugReport(
        job_id="/u/j",
        job_name="",
        state_name="failed",
        error_summary="boom",
        error="boom",
        submitted_at="-",
        started_at="-",
        finished_at="-",
        duration="-",
        resources="-",
        task_count=len(tasks),
        completed_count=0,
        failure_count=1,
        preemption_count=0,
        task_state_counts={},
        pending_reason="",
        tasks=tasks,
    )


def test_build_task_report_extracts_root_cause_from_noisy_logs():
    logs = [
        " 50%|#####     | 500/1000 [00:10<00:10,  5.0it/s]",
        "Traceback (most recent call last):",
        "RuntimeError: CUDA error: an illegal memory access was encountered",
    ]
    report = _build_task_report(_task_status(), logs)
    assert report.root_cause_lines == [
        "Traceback (most recent call last):",
        "RuntimeError: CUDA error: an illegal memory access was encountered",
    ]


def test_build_task_report_root_cause_empty_without_logs():
    report = _build_task_report(_task_status(), [])
    assert report.root_cause_lines == []


def test_format_bug_report_includes_likely_root_cause_section():
    task = _build_task_report(
        _task_status(),
        ["Fatal Python error: Segmentation fault", "core dumped"],
    )
    markdown = format_bug_report(_report([task]))
    assert "## Likely Root Cause" in markdown
    root_cause_section = markdown.split("## Likely Root Cause")[1].split("## Recent Logs")[0]
    assert "Fatal Python error: Segmentation fault" in root_cause_section


def test_format_bug_report_omits_likely_root_cause_section_when_no_logs():
    task = _build_task_report(_task_status(), [])
    markdown = format_bug_report(_report([task]))
    assert "## Likely Root Cause" not in markdown


def test_format_bug_report_omits_likely_root_cause_for_succeeded_task():
    task = _build_task_report(
        _task_status(state=job_pb2.TASK_STATE_SUCCEEDED),
        ["all good", "still good"],
    )
    markdown = format_bug_report(_report([task]))
    assert "## Likely Root Cause" not in markdown
