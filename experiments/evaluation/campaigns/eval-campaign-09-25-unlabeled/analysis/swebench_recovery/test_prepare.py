# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path

import pytest

prepare = runpy.run_path(str(Path(__file__).with_name("prepare.py")))


def test_partition_preserves_model_failures_and_recovers_transport_failures():
    rows = [
        {"task_name": "pass", "reward": 1, "exception_type": None},
        {"task_name": "fail", "reward": 0, "exception_type": None},
        {"task_name": "timeout", "reward": 0, "exception_type": "AgentTimeoutError"},
        {
            "task_name": "transport",
            "reward": 0,
            "exception_type": "NonZeroAgentExitCodeError",
            "agent_logs": [{"terminal_error_classes": ["httpx.ReadTimeout"]}],
        },
        {
            "task_name": "retry",
            "reward": 1,
            "exception_type": "AgentTimeoutError",
            "agent_logs": [{"terminal_error_classes": [], "retry_error_classes": ["ReadTimeout"]}],
        },
        {
            "task_name": "verifier",
            "reward": None,
            "exception_type": "VerifierTimeoutError",
            "transport_failure_observed": True,
            "agent_logs": [{"terminal_error_classes": ["ContextWindowExceededError"]}],
        },
    ]
    names = {row["task_name"] for row in rows} | {"missing"}
    retained, recovery = prepare["partition"]({"trials": rows}, names)
    assert {row["task_name"] for row in retained} == {"pass", "fail", "timeout"}
    assert recovery == ["missing", "retry", "transport", "verifier"]


def test_transport_keyword_alone_cannot_discard_a_scored_trial():
    row = {
        "task_name": "ambiguous",
        "reward": 0,
        "exception_type": "AgentTimeoutError",
        "transport_failure_observed": True,
        "agent_logs": [{"terminal_error_classes": []}],
    }
    with pytest.raises(ValueError, match="manual review"):
        prepare["partition"]({"trials": [row]}, {"ambiguous"})


def test_review_retains_model_failure_only_for_exact_source_evidence():
    row = {
        "task_name": "model-broke-environment",
        "sha256": "result-digest",
        "reward": 0,
        "exception_type": "NonZeroAgentExitCodeError",
        "transport_failure_observed": True,
        "agent_logs": [{"sha256": "log-digest", "terminal_error_classes": ["FileNotFoundError"]}],
    }
    report = {"model": "model", "trials": [row]}
    review = {
        "model": "model",
        "task_name": row["task_name"],
        "result_sha256": "result-digest",
        "agent_log_sha256": "log-digest",
        "decision": "retain_scored_model_failure",
        "evidence": "The model moved the directory containing its agent dependencies.",
    }
    prepare["apply_reviews"](report, [review])
    retained, recovery = prepare["partition"](report, {row["task_name"]})
    assert retained == [row]
    assert recovery == []
    row["agent_logs"][0]["terminal_error_classes"] = ["httpx.ReadTimeout"]
    retained, recovery = prepare["partition"](report, {row["task_name"]})
    assert retained == []
    assert recovery == [row["task_name"]]
    with pytest.raises(ValueError, match="evidence differs"):
        prepare["apply_reviews"](report, [dict(review, result_sha256="another-trial")])


def test_review_excludes_scored_infrastructure_failure_only_for_exact_source_evidence():
    row = {
        "task_name": "broken-agent-install",
        "sha256": "result-digest",
        "reward": 0,
        "exception_type": "NonZeroAgentExitCodeError",
        "agent_logs": [{"sha256": "log-digest", "terminal_error_classes": ["ImportError"]}],
    }
    report = {"model": "model", "trials": [row]}
    review = {
        "model": "model",
        "task_name": row["task_name"],
        "result_sha256": "result-digest",
        "agent_log_sha256": "log-digest",
        "decision": "exclude_infrastructure_failure",
        "evidence": "Agent startup could not import a required package before the task began.",
    }
    prepare["apply_reviews"](report, [review])
    retained, recovery = prepare["partition"](report, {row["task_name"]})
    assert retained == []
    assert recovery == [row["task_name"]]
    with pytest.raises(ValueError, match="evidence differs"):
        prepare["apply_reviews"](report, [dict(review, agent_log_sha256="different-log")])
