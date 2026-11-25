# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("skyrl_gym.verification", reason="Run with the native SkyRL CPU environment")

from experiments.evaluation.micro210_finish import summarize_rows


def row(uid, score, *, passed=None, exception=None, treatment="grade"):
    return {
        "uid": str(uid),
        "data_source": "fixture",
        "score": score,
        "verification": {"status": "verified", "score": score, "passed": passed},
        "exception_type": exception,
        "error_treatment": treatment,
        "output_response": "retained model output",
        "server_error": None,
    }


def test_policy_scoring_keeps_timeout_and_context_verdicts_but_excludes_unnamed_errors():
    expected = {str(index): "fixture" for index in range(5)}
    result = summarize_rows(
        expected,
        [
            row(0, 0, exception="ValueError", treatment="mask"),
            row(1, 1, passed=True, exception="AgentTimeoutError"),
            row(2, 1, passed=True, exception="ContextLengthExceededError"),
            row(3, 0, passed=False, exception="ContextLengthExceededError"),
        ],
    )
    assert (result["scored"], result["assigned"], result["passed"]) == (3, 5, 2)
    assert result["pass_at_1"] == pytest.approx(2 / 3)
    assert result["coverage"] == 0.6
    assert result["coverage_eligible"] is False
    assert {item["exception_type"] for item in result["unavailable"]} == {"ValueError", "MissingTrial"}
    assert [item["uid"] for item in result["zero_score_ledger"]] == ["3"]
    with pytest.raises(ValueError, match="Infrastructure error unexpectedly carries a scored verdict"):
        summarize_rows(expected, [row(0, 0, exception="ValueError")])


def test_coverage_gate_uses_declared_trials_and_zero_cells_wait_for_agent_review():
    expected = {str(index): "fixture" for index in range(10)}
    result = summarize_rows(expected, [row(index, 0, passed=False) for index in range(9)])
    assert result["coverage_eligible"] is True
    assert result["per_benchmark"][0]["unavailable"] == 1
    assert result["zero_score_benchmarks"] == ["fixture"]
    assert len(result["zero_score_ledger"]) == 9
    assert result["seal_status"] == "AWAITING_AGENT_TRACE_REVIEW"
    with pytest.raises(ValueError, match="Unexpected or duplicate"):
        summarize_rows(expected, [row(0, 1), row(0, 1)])
