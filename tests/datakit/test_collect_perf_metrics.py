# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scripts/datakit/collect_perf_metrics.py.

Each test defends a piece of non-trivial behavior that's easy to silently
break: log-line regexes, the failure-bucket heuristic, and the report
assembly that combines summary + logs + status.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "datakit" / "collect_perf_metrics.py"
_spec = importlib.util.spec_from_file_location("collect_perf_metrics", _MODULE_PATH)
assert _spec and _spec.loader
collect_perf_metrics = importlib.util.module_from_spec(_spec)
# Register before exec_module so @dataclass can resolve __module__ via sys.modules.
sys.modules[_spec.name] = collect_perf_metrics
_spec.loader.exec_module(collect_perf_metrics)


_STEP_RUNNER_LOGS = """\
2026-05-06 07:00:00 INFO Step datakit-smoke/download_a1b2c3d4 succeeded in 0:00:12
2026-05-06 07:14:00 INFO Step datakit-smoke/normalize_e5f6a7b8 succeeded in 0:14:05.123
2026-05-06 07:21:00 INFO Step datakit-smoke/minhash_aabbccdd succeeded in 0:06:50.0
2026-05-06 07:24:00 INFO Step datakit-smoke/fuzzy_dups_11223344 succeeded in 0:03:08
2026-05-06 07:27:00 INFO Step datakit-smoke/consolidate_55667788 succeeded in 0:03:40
2026-05-06 07:30:00 INFO Step datakit-smoke/tokenize_99aabbcc succeeded in 0:03:17
"""


def test_parse_timedelta_str_handles_both_python_formats():
    """Python's str(timedelta) emits 'H:MM:SS' or 'N day(s), H:MM:SS'."""
    assert collect_perf_metrics.parse_timedelta_str("0:13:25.412345") == pytest.approx(805.412345)
    assert collect_perf_metrics.parse_timedelta_str("2 days, 1:00:00") == pytest.approx(2 * 86400 + 3600)


def test_parse_stage_wall_seconds_strips_step_hash_and_pipeline_prefix():
    """Step labels are '<prefix>/<step>_<hex-hash>'. Captured key must be just <step>."""
    durations, cached = collect_perf_metrics.parse_stage_wall_seconds(_STEP_RUNNER_LOGS)
    assert set(durations) == {"download", "normalize", "minhash", "fuzzy_dups", "consolidate", "tokenize"}
    assert durations["fuzzy_dups"] == pytest.approx(188.0)  # underscore in step name preserved
    assert cached == []


def test_parse_stage_wall_seconds_records_cache_hits_at_zero():
    """Cache-hit steps emit 'completed by another worker' (no duration)."""
    logs = (
        "Step datakit-smoke/download_a1b2c3d4 completed by another worker\n"
        "Step datakit-smoke/normalize_e5f6a7b8 succeeded in 0:14:05.123\n"
    )
    durations, cached = collect_perf_metrics.parse_stage_wall_seconds(logs)
    assert durations == {"download": 0.0, "normalize": pytest.approx(845.123)}
    assert cached == ["download"]


def test_parse_total_wall_seconds_picks_up_rigging_log_time_line():
    """Fallback path when iris summary lacks per-task durations."""
    logs = "Datakit ferry total wall time took 0:50:54.612345\n"
    assert collect_perf_metrics.parse_total_wall_seconds_from_logs(logs) == pytest.approx(3054.612345)


@pytest.mark.parametrize(
    ("state", "exit_code", "error", "expected"),
    [
        # One representative case per bucket — the heuristic is the load-bearing logic.
        ("succeeded", 0, "", None),
        ("preempted", None, "", "preempted"),
        ("failed", 137, "killed (oom)", "oom"),
        ("failed", 1, "TPU device fault", "hardware_fault"),
        ("unschedulable", None, "no capacity", "scheduling_timeout"),
        ("failed", 1, "ValueError: bad input", "application_failure"),
        ("running", None, "", "other"),
    ],
)
def test_classify_task_failure_buckets(state, exit_code, error, expected):
    assert collect_perf_metrics.classify_task_failure(state, exit_code, error) == expected


def test_classify_failures_aggregates_and_preserves_legacy_fields():
    """classify_failures must return both bucket counts and the legacy ooms/failed_shards
    fields used by downstream comparison tooling."""
    tasks = [
        {"state": "succeeded", "exit_code": 0, "error": ""},
        {"state": "preempted", "exit_code": None, "error": ""},
        {"state": "failed", "exit_code": 137, "error": "oomkiller"},
        {"state": "failed", "exit_code": 137, "error": "oomkiller"},
        {"state": "failed", "exit_code": 1, "error": "ValueError"},
    ]
    buckets, ooms, failed_shards = collect_perf_metrics.classify_failures(tasks)
    assert buckets["preempted"] == 1
    assert buckets["oom"] == 2
    assert buckets["application_failure"] == 1
    assert ooms == 2
    assert failed_shards == 1


def _fake_summary() -> dict:
    return {
        "job_id": "iris-run-abc",
        "state": "succeeded",
        "failure_count": 1,
        "preemption_count": 2,
        "task_count": 6,
        "completed_count": 6,
        "task_state_counts": {"succeeded": 5, "preempted": 1},
        "tasks": [
            {
                "task_id": "iris-run-abc/0",
                "state": "succeeded",
                "exit_code": 0,
                "duration_ms": 3_054_612,
                "memory_peak_mb": 14202,
                "error": "",
            },
            {
                "task_id": "iris-run-abc/1",
                "state": "preempted",
                "exit_code": None,
                "duration_ms": 100,
                "memory_peak_mb": 0,
                "error": "preempted by gcp",
            },
        ],
    }


def test_build_report_combines_summary_logs_and_status():
    report = collect_perf_metrics.build_report(
        job_id="iris-run-abc",
        summary=_fake_summary(),
        job_tree=_fake_job_tree(),
        logs=_STEP_RUNNER_LOGS,
        status={"status": "succeeded", "marin_prefix": "gs://marin-us-central1"},
        workflow_env={"run_id": "42", "run_attempt": "1", "workflow": "tier1", "commit_sha": "deadbeef"},
    )
    assert report.status == "succeeded"
    assert report.marin_prefix == "gs://marin-us-central1"
    # Counts come from the job tree (3 children + parent), not the parent-only summary's preemption_count=2.
    assert report.preemption_count == 3
    assert report.peak_worker_memory_mb == 14202
    assert report.wall_seconds_total == pytest.approx(3054.612)
    # infra_failures still derives from parent-only per-task heuristic.
    assert report.infra_failures["preempted"] == 1
    # Stage durations come from the parsed logs, not the iris summary.
    assert set(report.stage_wall_seconds) >= set(collect_perf_metrics.EXPECTED_STEPS)


def test_build_report_falls_back_to_log_wall_time_when_summary_unavailable():
    """If iris summary fails, total wall comes from the rigging log_time line."""
    logs = "Datakit ferry total wall time took 0:50:54.612345\n"
    report = collect_perf_metrics.build_report(
        job_id="iris-run-abc",
        summary=None,
        job_tree=None,
        logs=logs,
        status=None,
        workflow_env={"run_id": None, "run_attempt": None, "workflow": None, "commit_sha": None},
    )
    assert report.wall_seconds_total == pytest.approx(3054.612345)


def test_build_report_treats_missing_download_as_normal_when_cache_hit():
    """download is intentionally absent from EXPECTED_STEPS — some ferries skip it,
    and a cache-hit download surfaces only via cached_steps. Neither should warn."""
    logs = "Step datakit-smoke/download_a1b2c3d4 completed by another worker\n" + _STEP_RUNNER_LOGS.replace(
        "Step datakit-smoke/download_a1b2c3d4 succeeded in 0:00:12\n", ""
    )
    report = collect_perf_metrics.build_report(
        job_id="iris-run-abc",
        summary=_fake_summary(),
        job_tree=_fake_job_tree(),
        logs=logs,
        status=None,
        workflow_env={"run_id": None, "run_attempt": None, "workflow": None, "commit_sha": None},
    )
    assert report.cached_steps == ["download"]
    assert "download" not in collect_perf_metrics.EXPECTED_STEPS
    assert not any("missing expected steps" in w for w in report.warnings)


def _fake_job_tree() -> list[dict]:
    """Realistic shape: parent (launcher) + two child jobs with worker tasks."""
    return [
        # Parent — launcher task only
        {"job_id": "iris-run-abc", "preemption_count": 0, "failure_count": 0, "task_state_counts": {"succeeded": 1}},
        # Coordinator job
        {
            "job_id": "iris-run-abc/zephyr-cache-copy-p0-a0",
            "preemption_count": 0,
            "failure_count": 0,
            "task_state_counts": {"succeeded": 1},
        },
        # Workers that actually fan out — these are the ones that get preempted/killed
        {
            "job_id": "iris-run-abc/zephyr-cache-copy-p0-a0/workers-a0",
            "preemption_count": 3,
            "failure_count": 1,
            "task_state_counts": {"succeeded": 79, "killed": 27, "preempted": 3},
        },
    ]


def test_aggregate_job_tree_sums_across_children():
    """Parent's iris job summary only sees the launcher; children carry the real
    preemption / failure / task-state counts. aggregate_job_tree must sum them."""
    agg = collect_perf_metrics.aggregate_job_tree(_fake_job_tree())
    assert agg["preemption_count"] == 3
    assert agg["failure_count"] == 1
    assert agg["task_state_counts"] == {"succeeded": 81, "killed": 27, "preempted": 3}
    assert agg["job_count"] == 3


def test_build_report_uses_tree_aggregation_for_counts():
    """Counts must come from the whole job tree, not the parent-only summary."""
    parent_only_summary = {
        "preemption_count": 0,
        "failure_count": 0,
        "task_state_counts": {"succeeded": 1},
        "tasks": [
            {
                "task_id": "iris-run-abc/0",
                "state": "succeeded",
                "exit_code": 0,
                "duration_ms": 3_000_000,
                "memory_peak_mb": 100,
                "error": "",
            },
        ],
    }
    report = collect_perf_metrics.build_report(
        job_id="iris-run-abc",
        summary=parent_only_summary,
        job_tree=_fake_job_tree(),
        logs="",
        status=None,
        workflow_env={"run_id": None, "run_attempt": None, "workflow": None, "commit_sha": None},
    )
    # Aggregated counts override the parent-only zeros.
    assert report.preemption_count == 3
    assert report.failure_count == 1
    assert report.task_state_counts["killed"] == 27
    assert report.tree_job_count == 3


def test_build_report_falls_back_to_parent_summary_when_tree_unavailable():
    summary = {
        "preemption_count": 5,
        "failure_count": 2,
        "task_state_counts": {"succeeded": 1},
        "tasks": [],
    }
    report = collect_perf_metrics.build_report(
        job_id="iris-run-abc",
        summary=summary,
        job_tree=None,
        logs="",
        status=None,
        workflow_env={"run_id": None, "run_attempt": None, "workflow": None, "commit_sha": None},
    )
    assert report.preemption_count == 5
    assert report.failure_count == 2
    assert report.tree_job_count == 0
    assert any("iris job list --prefix" in w for w in report.warnings)
