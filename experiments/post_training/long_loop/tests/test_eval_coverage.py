# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.post_training.long_loop.eval_runtime.audit_results import audit, require_evaluation_coverage


def test_same_total_cannot_hide_missing_benchmark_repetitions(tmp_path):
    directory = tmp_path / "exports/dumped_evals/eval_only"
    directory.mkdir(parents=True)
    row = {
        "data_source": "aime24",
        "verification": {"status": "verified", "score": 1},
        "response_token_ids": [1, 2],
        "stop_reason": "stop",
    }
    output = directory / "results.jsonl"
    output.write_text("\n".join(json.dumps(row) for _ in range(3)))
    expected = {"aime24": 2, "gpqa-diamond": 1}
    wrong = audit(str(tmp_path), expected_rollouts=3, expected_by_data_source=expected)
    assert wrong["completed_rollouts"] == 3
    assert not wrong["all_verified"]
    output.write_text("\n".join(json.dumps(item) for item in [row, row, {**row, "data_source": "gpqa-diamond"}]))
    corrected = audit(str(tmp_path), expected_rollouts=3, expected_by_data_source=expected)
    assert corrected["all_verified"]


def test_duplicate_attempt_cannot_replace_missing_identity(tmp_path):
    directory = tmp_path / "exports/dumped_evals/eval_only"
    directory.mkdir(parents=True)
    row = {
        "data_source": "dataset",
        "uid": "0",
        "verification": {"status": "verified", "score": 1},
        "response_token_ids": [1],
        "stop_reason": "stop",
    }
    output = directory / "results.jsonl"
    expected = [{"data_source": "dataset", "dataset_row_uid": str(i)} for i in range(2)]
    output.write_text("\n".join(json.dumps(row) for _ in range(2)))
    assert not audit(str(tmp_path), 2, {"dataset": 2}, expected)["all_verified"]
    output.write_text("\n".join(json.dumps({**row, "uid": str(i)}) for i in range(2)))
    assert audit(str(tmp_path), 2, {"dataset": 2}, expected)["all_verified"]


def _row(data_source, uid, *, status="verified", score=1, exception_type=None):
    return {
        "data_source": data_source,
        "uid": uid,
        "verification": {"status": status, "score": score},
        "response_token_ids": [1],
        "stop_reason": "stop",
        "exception_type": exception_type,
    }


def _expected_identities(count):
    return [{"data_source": "bench", "dataset_row_uid": str(i)} for i in range(count)]


def _write_rows(tmp_path, rows):
    directory = tmp_path / "exports/dumped_evals/eval_only"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows))


def test_policy_seal_accepts_infrastructure_absent_attempts(tmp_path):
    """#10197: unnamed/infrastructure unscored attempts count as absent; 96% valid seals."""
    rows = [_row("bench", str(i)) for i in range(423)]
    rows += [_row("bench", str(423 + i), status="error", score=0, exception_type="ValueError") for i in range(10)]
    rows.append(_row("bench", "433", status="error", score=0, exception_type="VerifierTimeoutError"))
    _write_rows(tmp_path, rows)
    evaluation = audit(str(tmp_path), expected_rollouts=440, expected_identities=_expected_identities(440))
    assert evaluation["verified_rollouts"] == 423
    require_evaluation_coverage(evaluation)
    receipt = evaluation["unscored_policy"]
    assert receipt["valid_scored"] == 423 and receipt["expected_attempts"] == 440
    assert receipt["infrastructure_absent"] == 17 and receipt["missing_attempts"] == 6
    assert receipt["gradeable_unscored"] == []
    assert receipt["valid_fraction"] == pytest.approx(423 / 440)


def test_policy_gate_refuses_gradeable_unscored_attempts(tmp_path):
    """#10197: Harbor AgentTimeout runs the verifier; it is not infrastructure-absent."""
    rows = [_row("bench", str(i)) for i in range(430)]
    rows.append(_row("bench", "430", status="error", score=0, exception_type="AgentTimeoutError"))
    _write_rows(tmp_path, rows)
    evaluation = audit(str(tmp_path), expected_rollouts=440, expected_identities=_expected_identities(440))
    with pytest.raises(ValueError, match="unscored"):
        require_evaluation_coverage(evaluation)
    assert evaluation["unscored_policy"]["gradeable_unscored"]


def test_policy_gate_refuses_below_seal_fraction(tmp_path):
    """#10197: the evaluation seals only at >=90% valid."""
    rows = [_row("bench", str(i)) for i in range(390)]
    _write_rows(tmp_path, rows)
    evaluation = audit(str(tmp_path), expected_rollouts=440, expected_identities=_expected_identities(440))
    with pytest.raises(ValueError, match="unscored"):
        require_evaluation_coverage(evaluation)
    assert evaluation["unscored_policy"]["valid_fraction"] < 0.90


def test_policy_gate_refuses_unexpected_identity(tmp_path):
    """A duplicated attempt identity cannot stand in for a missing one under the policy seal."""
    rows = [_row("bench", str(i)) for i in range(439)]
    rows.append(_row("bench", "0"))
    _write_rows(tmp_path, rows)
    evaluation = audit(str(tmp_path), expected_rollouts=440, expected_identities=_expected_identities(440))
    with pytest.raises(ValueError, match="unscored"):
        require_evaluation_coverage(evaluation)
    assert evaluation["identity_unexpected"]
