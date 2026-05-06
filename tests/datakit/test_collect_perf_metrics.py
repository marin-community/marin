# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scripts/datakit/collect_perf_metrics.py.

Each test defends a piece of non-trivial behavior that's easy to silently
break: tree-based per-step duration accounting, the failure-bucket heuristic,
and the report assembly that combines summary + tree + status.
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


# --------------------------------------------------------------------------- #
# compute_stage_wall_seconds
# --------------------------------------------------------------------------- #


def _job(job_id: str, started_ms: int, finished_ms: int, **kw) -> dict:
    """Build a minimal iris-job-list-style dict for tests."""
    return {
        "job_id": job_id,
        "started_at": {"epoch_ms": str(started_ms)},
        "finished_at": {"epoch_ms": str(finished_ms)},
        **kw,
    }


def test_compute_stage_wall_seconds_maps_prefixes_sums_phases_skips_workers():
    """Three things at once, because they're the load-bearing semantics:
    - zephyr-* prefixes map to the right ferry step name
    - multi-phase jobs (fuzzy_dups p0..p4) sum across phases
    - worker jobs nested under coordinators are skipped (would double-count)"""
    parent = "/u/run-abc"
    jobs = [
        # Direct children — these contribute durations.
        _job(f"{parent}/zephyr-normalize-aaa-p0-a0", 1_000_000, 1_060_000),  # 60s
        _job(f"{parent}/zephyr-minhash-attrs-bbb-p0-a0", 2_000_000, 2_120_000),  # 120s
        _job(f"{parent}/zephyr-fuzzy-dups-ccc-p0-a0", 3_000_000, 3_300_000),  # 300s
        _job(f"{parent}/zephyr-fuzzy-dups-ccc-p1-a0", 3_300_000, 3_500_000),  # 200s -> sums with p0
        _job(f"{parent}/zephyr-consolidate-filter-ddd-p0-a0", 4_000_000, 4_030_000),  # 30s
        _job(f"{parent}/zephyr-tokenize-train-eee-p0-a0", 5_000_000, 5_180_000),  # 180s
        _job(f"{parent}/zephyr-levanter-cache-copy-fff-p0-a0", 5_180_000, 5_200_000),  # 20s -> tokenize
        # Worker job nested two levels deep — must NOT add to the parent step's time.
        _job(f"{parent}/zephyr-tokenize-train-eee-p0-a0/zephyr-tokenize-train-eee-p0-workers-a0", 5_000_000, 5_180_000),
        # Parent itself — should be skipped (different depth).
        _job(parent, 0, 999_999),
    ]
    durations, cached = collect_perf_metrics.compute_stage_wall_seconds(jobs, parent)
    assert durations["normalize"] == pytest.approx(60.0)
    assert durations["minhash"] == pytest.approx(120.0)
    assert durations["fuzzy_dups"] == pytest.approx(500.0)  # 300 + 200
    assert durations["consolidate"] == pytest.approx(30.0)
    # tokenize-train + levanter-cache-copy both fold into tokenize.
    assert durations["tokenize"] == pytest.approx(200.0)  # 180 + 20
    # download didn't run — not in EXPECTED_STEPS, so doesn't appear in cached_steps.
    assert "download" not in durations
    assert cached == []


def test_compute_stage_wall_seconds_marks_missing_expected_steps_as_cached():
    """A pipeline step in EXPECTED_STEPS that produces no child job means cache hit."""
    parent = "/u/run-xyz"
    jobs = [_job(f"{parent}/zephyr-normalize-aaa-p0-a0", 1_000_000, 1_060_000)]
    durations, cached = collect_perf_metrics.compute_stage_wall_seconds(jobs, parent)
    assert durations["normalize"] == pytest.approx(60.0)
    # Everything else cache-hit.
    assert set(cached) == {"minhash", "fuzzy_dups", "consolidate", "tokenize"}
    for s in cached:
        assert durations[s] == 0.0


# --------------------------------------------------------------------------- #
# classify_task_failure / classify_failures
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# aggregate_job_tree (preemption / failure / task-state counts)
# --------------------------------------------------------------------------- #


def _fake_job_tree() -> list[dict]:
    """Realistic shape: parent (launcher) + coordinator + worker job."""
    return [
        {"job_id": "/u/run-abc", "preemption_count": 0, "failure_count": 0, "task_state_counts": {"succeeded": 1}},
        {
            "job_id": "/u/run-abc/zephyr-normalize-aaa-p0-a0",
            "preemption_count": 0,
            "failure_count": 0,
            "task_state_counts": {"succeeded": 1},
        },
        {
            "job_id": "/u/run-abc/zephyr-normalize-aaa-p0-a0/workers-a0",
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


# --------------------------------------------------------------------------- #
# build_report (end-to-end over synthetic inputs)
# --------------------------------------------------------------------------- #


def _fake_summary() -> dict:
    return {
        "job_id": "/u/run-abc",
        "state": "succeeded",
        "failure_count": 1,
        "preemption_count": 2,
        "task_count": 1,
        "completed_count": 1,
        "task_state_counts": {"succeeded": 1},
        "tasks": [
            {
                "task_id": "/u/run-abc/0",
                "state": "succeeded",
                "exit_code": 0,
                "duration_ms": 3_054_612,
                "memory_peak_mb": 14202,
                "error": "",
            },
        ],
    }


def _fake_stage_tree(parent: str) -> list[dict]:
    return [
        _job(f"{parent}/zephyr-normalize-a-p0-a0", 1_000_000, 1_060_000),
        _job(f"{parent}/zephyr-minhash-attrs-b-p0-a0", 2_000_000, 2_120_000),
        _job(f"{parent}/zephyr-fuzzy-dups-c-p0-a0", 3_000_000, 3_300_000),
        _job(f"{parent}/zephyr-consolidate-filter-d-p0-a0", 4_000_000, 4_030_000),
        _job(f"{parent}/zephyr-tokenize-train-e-p0-a0", 5_000_000, 5_180_000),
    ]


def test_build_report_combines_summary_tree_and_status():
    parent = "/u/run-abc"
    report = collect_perf_metrics.build_report(
        job_id=parent,
        summary=_fake_summary(),
        job_tree=_fake_job_tree() + _fake_stage_tree(parent),
        status={"status": "succeeded", "marin_prefix": "gs://marin-us-central1"},
        workflow_env={"run_id": "42", "run_attempt": "1", "workflow": "tier1", "commit_sha": "deadbeef"},
    )
    assert report.status == "succeeded"
    assert report.peak_worker_memory_mb == 14202
    assert report.wall_seconds_total == pytest.approx(3054.612)
    # Tree-derived stage durations.
    assert set(collect_perf_metrics.EXPECTED_STEPS).issubset(report.stage_wall_seconds)
    assert report.stage_wall_seconds["normalize"] == pytest.approx(60.0)
    # Counts come from the tree, not the parent-only summary's preemption_count=2.
    assert report.preemption_count == 3


def test_build_report_warns_when_all_steps_cache_hit():
    """A run where every expected step cache-hits is anomalous — flag it."""
    parent = "/u/run-abc"
    report = collect_perf_metrics.build_report(
        job_id=parent,
        summary=_fake_summary(),
        job_tree=_fake_job_tree(),  # no zephyr-* stage jobs
        status=None,
        workflow_env={"run_id": None, "run_attempt": None, "workflow": None, "commit_sha": None},
    )
    assert set(report.cached_steps) == set(collect_perf_metrics.EXPECTED_STEPS)
    assert any("all expected steps cache-hit" in w for w in report.warnings)


def test_build_report_falls_back_to_parent_summary_when_tree_unavailable():
    summary = {
        "preemption_count": 5,
        "failure_count": 2,
        "task_state_counts": {"succeeded": 1},
        "tasks": [],
    }
    report = collect_perf_metrics.build_report(
        job_id="/u/run-abc",
        summary=summary,
        job_tree=None,
        status=None,
        workflow_env={"run_id": None, "run_attempt": None, "workflow": None, "commit_sha": None},
    )
    assert report.preemption_count == 5
    assert report.failure_count == 2
    assert report.tree_job_count == 0
    assert any("iris job list --prefix" in w for w in report.warnings)
    assert any("iris job tree unavailable" in w for w in report.warnings)
