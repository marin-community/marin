# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib
import json

import pytest
import yaml

recovery = importlib.import_module(
    "experiments.evaluation.campaigns.eval-campaign-09-25-unlabeled.ablations.swebench_context_compaction.prepare_recovery"
)
combined = importlib.import_module(
    "experiments.evaluation.campaigns.eval-campaign-09-25-unlabeled.ablations.swebench_context_compaction.combine_recovery"
)


def test_recovery_preserves_valid_results_but_retries_deadline_and_unfinished_tasks():
    successful = {
        "agent_execution": {"started_at": "2026-10-20T00:00:00Z", "finished_at": "2026-10-20T00:10:00Z"},
    }
    # A verifier can overwrite AgentTimeoutError, and can even write reward 1
    # after a runaway agent. Duration must still exclude that old-runtime result.
    late = {
        "agent_execution": {"started_at": "2026-10-20T00:00:00Z", "finished_at": "2026-10-20T00:30:01Z"},
        "verifier_result": {"rewards": {"reward": 1.0}},
    }
    evidence = [
        {"task_name": "valid", "recovery_reason": recovery.recovery_reason(successful, True, 1800)},
        {"task_name": "late", "recovery_reason": recovery.recovery_reason(late, True, 1800)},
    ]
    retained, requested = recovery.partition_tasks({"valid", "late", "unfinished"}, evidence)
    assert retained == ["valid"]
    assert requested == ["late", "unfinished"]


def test_recovery_retries_setup_failure_without_reclassifying_normal_agent_failures():
    setup = {"exception_info": {"exception_type": "NonZeroAgentExitCodeError"}}
    executed = {**setup, "agent_execution": {"started_at": "2026-10-20T00:00:00Z"}}
    assert recovery.recovery_reason(setup, True, 1800) == "agent_setup_failed_before_execution"
    assert recovery.recovery_reason(executed, True, 1800) is None
    assert recovery.recovery_reason({}, False, 1800) == "unscored:missing_verifier_result"


def test_recovery_rejects_duplicate_results_instead_of_double_counting():
    evidence = [{"task_name": "task", "recovery_reason": None}] * 2
    with pytest.raises(ValueError, match="duplicate source task"):
        recovery.partition_tasks({"task"}, evidence)


def test_followup_preserves_both_generations_and_valid_timeouts(tmp_path):
    tasks = ["old", "new", "timeout", "outage", "infra", "missing"]
    previous = {
        "task_descriptors": [{"path": f"tasks/{task}"} for task in tasks],
        "retained_tasks": ["old"],
        "harbor_commit": "tested-runtime",
    }
    rows = [
        {
            "task_name": task,
            "source_uri": f"s3://test/{generation}/{task}",
            "sha256": task,
            "scored": scored,
            "reward": reward,
            "error_type": error,
        }
        for task, generation, scored, reward, error in [
            ("old", "original", True, 1.0, None),
            ("new", "first-recovery", True, 1.0, None),
            ("timeout", "first-recovery", True, 0.0, "AgentTimeoutError"),
            ("outage", "first-recovery", False, 1.0, None),
            ("infra", "first-recovery", False, 0.0, "VerifierRuntimeError"),
        ]
    ]
    rows[3]["exclusion_reason"] = "inference_service_outage"
    manifest = recovery.followup_manifest(previous, {"trials": rows})
    recovery.write_recovery({"datasets": [{"name": "swebench"}]}, manifest, tmp_path)
    saved = json.loads((tmp_path / "manifest.json").read_text())
    config = yaml.safe_load((tmp_path / "harbor.yaml").read_text())
    assert saved["retained_tasks"] == ["new", "old", "timeout"]
    assert config["datasets"][0]["task_names"] == ["infra", "missing", "outage"]
    retained = [row for row in saved["observed_trials"] if row["recovery_reason"] is None]
    assert {row["source_uri"] for row in retained} == {
        "s3://test/original/old",
        "s3://test/first-recovery/new",
        "s3://test/first-recovery/timeout",
    }
    replayed = [{"task_name": task, "scored": True, "reward": 0.0} for task in saved["recovery_tasks"]]
    report = combined.combine(saved, retained, replayed, [])
    assert report["all_tasks_scored"]
    assert report["scored_tasks"] == 6
    assert report["mean_reward"] == pytest.approx(2 / 6)


def test_followup_rejects_a_subset_only_audit_that_loses_retained_results():
    previous = {
        "task_descriptors": [{"path": "tasks/old"}, {"path": "tasks/new"}],
        "retained_tasks": ["old"],
        "harbor_commit": "tested-runtime",
    }
    report = {"trials": [{"task_name": "new", "scored": True}]}
    with pytest.raises(ValueError, match="missing previously retained evidence"):
        recovery.followup_manifest(previous, report)


def test_native_recovery_retains_valid_timeouts_and_excludes_outage_wins_and_losses(tmp_path):
    tasks = ["timeout", "outage-win", "outage-loss", "missing"]
    evidence = [
        {
            "task_name": task,
            "source_uri": f"s3://test/run/{task}/result.json",
            "sha256": task,
            "scored": True,
            "reward": reward,
            "error_type": "AgentTimeoutError" if task == "timeout" else None,
            "agent_execution": {"started_at": start, "finished_at": finish},
        }
        for task, reward, start, finish in [
            ("timeout", 0.0, "2026-10-20T00:00:00Z", "2026-10-20T00:30:01Z"),
            ("outage-win", 1.0, "2026-10-20T00:31:00Z", "2026-10-20T00:40:00Z"),
            ("outage-loss", 0.0, "2026-10-20T00:32:00Z", "2026-10-20T00:40:00Z"),
        ]
    ]
    outages = [{"results_root": "s3://test/run", "started_at": "2026-10-20T00:31:00Z", "finished_at": None}]
    manifest = recovery.native_recovery_manifest(
        [{"path": f"tasks/{task}"} for task in tasks], evidence, outages, "fixed-runtime"
    )
    recovery.write_recovery({"datasets": [{"name": "swebench"}]}, manifest, tmp_path)
    saved = json.loads((tmp_path / "manifest.json").read_text())
    assert saved["retained_tasks"] == ["timeout"]
    assert saved["recovery_tasks"] == ["missing", "outage-loss", "outage-win"]
    assert saved["observed_trials"][2]["reward"] == 0.0
