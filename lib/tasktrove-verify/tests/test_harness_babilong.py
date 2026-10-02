# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Client reference normalization preserves source lower and literal whitespace semantics."""

import pytest
from tasktrove_verify.adapters.harness_babilong import grade_babilong_candidate
from tasktrove_verify.grade import InvalidTask


@pytest.mark.parametrize(
    "reference,candidate,score",
    [
        ("  kitchen  ", "The KITCHEN", 1.0),
        ("room", "roommate", 1.0),
        ("ß", "SS", 0.0),
        ("ki\ntchen", "ki\ntchen", 1.0),
        ("ki\ntchen", "ki tchen", 0.0),
        ("kitchen", "", 0.0),
    ],
)
def test_reference_normalization_and_substring_grading(reference, candidate, score):
    assert grade_babilong_candidate(reference, candidate).reward == score


@pytest.mark.parametrize(
    "reference,candidate", [("", "anything"), ("  ", "anything"), (None, "text"), (1, "1"), ("x", None)]
)
def test_vacuous_or_malformed_source_contract_is_unscored(reference, candidate):
    with pytest.raises(InvalidTask):
        grade_babilong_candidate(reference, candidate)
