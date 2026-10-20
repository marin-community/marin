# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest

recovery = importlib.import_module(
    "experiments.evaluation.campaigns.eval-campaign-09-25-unlabeled.ablations.swebench_context_compaction.combine_recovery"
)


@pytest.fixture
def manifest():
    return {
        "task_descriptors": [{"path": name} for name in ("old", "new", "missing")],
        "retained_tasks": ["old"],
        "recovery_tasks": ["new", "missing"],
    }


def test_combined_score_preserves_missing_and_infrastructure_coverage(manifest):
    retained = [{"task_name": "old", "reward": 1.0, "scored": True}]
    recovered = [{"task_name": "new", "reward": 0.0, "scored": False}]
    result = recovery.combine(manifest, retained, recovered, [])
    assert result["mean_reward"] == 1.0
    assert result["scored_tasks"] == 1
    assert result["unscored_tasks"] == 1
    assert result["pending_tasks"] == ["missing"]
    assert not result["all_tasks_scored"]
    recovered = [
        {"task_name": "new", "reward": 0.0, "scored": True},
        {"task_name": "missing", "reward": 1.0, "scored": True},
    ]
    result = recovery.combine(manifest, retained, recovered, [])
    assert result["mean_reward"] == pytest.approx(2 / 3)
    assert result["all_tasks_scored"]


def test_combined_score_rejects_replayed_retained_task(manifest):
    retained = [{"task_name": "old", "reward": 1.0, "scored": True}]
    with pytest.raises(ValueError, match="unexpected or duplicate task"):
        recovery.combine(manifest, retained, retained, [])


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_outage_excludes_overlapping_successes_and_failures(manifest, reward):
    retained = [
        {
            "task_name": "old",
            "reward": 1.0,
            "scored": True,
            "source_uri": "s3://bucket/retained/trial/result.json",
            "agent_execution": {"started_at": "2026-10-20T21:20:00Z", "finished_at": "2026-10-20T21:50:00Z"},
        }
    ]
    recovered = [
        {
            "task_name": "new",
            "reward": reward,
            "scored": True,
            "source_uri": "s3://bucket/recovery/trial/result.json",
            "agent_execution": {"started_at": "2026-10-20T21:20:00Z", "finished_at": "2026-10-20T21:50:00Z"},
        },
        {
            "task_name": "missing",
            "reward": 0.0,
            "scored": True,
            "source_uri": "s3://bucket/recovery/other/result.json",
            "agent_execution": {"started_at": "2026-10-20T21:00:00Z", "finished_at": "2026-10-20T21:30:00Z"},
        },
    ]
    outages = [{"results_root": "s3://bucket/recovery", "started_at": "2026-10-20T21:30:00Z", "finished_at": None}]
    result = recovery.combine(manifest, retained, recovered, outages)
    assert result["scored_tasks"] == 2
    assert result["outage_excluded_tasks"] == 1
    assert result["mean_reward"] == 0.5
    assert not result["all_tasks_scored"]
    assert recovered[0]["scored"] is True
