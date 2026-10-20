# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest

recovery = importlib.import_module(
    "experiments.evaluation.campaigns.eval-campaign-09-25-unlabeled.ablations.swebench_context_compaction.prepare_recovery"
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
