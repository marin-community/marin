# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from skyrl_gym.envs.nupa.verifier import NUPAVerifier
from skyrl_gym.verification import RolloutEvidence, VerificationStatus


@pytest.mark.parametrize(
    "answer,answer_format,response,prediction,exact,digit_match,format_valid",
    [
        ("123", "Integer", "Thus, the final answer is: 23", "23", 0, 2 / 3, 1),
        ("1.23e5", "ScientificNotation", r"\boxed{1.23e-5}", "1.23e5", 1, 1, 1),
        ("1.0", "Float", "1.00", "1.00", 0, 1, 1),
        ("123", "Integer", "I cannot answer", None, 0, 0, 0),
    ],
)
def test_nupa_json_ground_truth_retains_source_grades_and_skyrl_metric_names(
    answer, answer_format, response, prediction, exact, digit_match, format_valid
):
    verifier = NUPAVerifier(json.dumps({"answer": answer, "answer_format": answer_format}))
    result = verifier.verify(RolloutEvidence(response=response))
    assert result.status is VerificationStatus.VERIFIED
    assert result.score == exact
    assert result.passed is bool(exact)
    assert result.diagnostics["prediction"] == prediction
    assert result.diagnostics["exact_match"] == exact
    assert result.diagnostics["digit_match"] == digit_match
    assert result.diagnostics["format_valid"] == format_valid
    assert result.diagnostics["no_answer"] == 1 - format_valid
    assert "format_valid_rate" not in result.diagnostics
