# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from tasktrove_verify.adapters.harness_math_literal import grade_normalized_math
from tasktrove_verify.grade import Status


def strip_assignment(value):
    """Source's short-variable assignment normalization policy."""
    return value.removeprefix("x=")


@pytest.mark.parametrize("value", ["inf", "nan", "+infinity", "-inf"])
def test_assignment_cannot_hide_invalid_nonfinite_reference(value):
    result = grade_normalized_math("x=" + value, "x=" + value, strip_assignment)
    assert result.status is Status.INVALID_TASK
    assert result.reward == 0


@pytest.mark.parametrize("candidate", [None, "", "x=nan", "x=inf"])
def test_missing_or_normalized_nonfinite_candidate_scores_zero(candidate):
    result = grade_normalized_math("42", candidate, strip_assignment)
    assert result.status is Status.SCORED
    assert result.reward == 0


def test_normalized_assignment_uses_literal_case_sensitive_equality():
    assert grade_normalized_math("A", "x=A", strip_assignment).reward == 1
    assert grade_normalized_math("A", "x=a", strip_assignment).reward == 0
