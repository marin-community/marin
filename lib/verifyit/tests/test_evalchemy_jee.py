# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.adapters.evalchemy_jee import grade_jee_answer
from verifyit.grade import Status


@pytest.mark.parametrize(
    "reference,candidate,question_type,expected",
    [
        ("A,C,D", "DAC", "MCQ(multiple)", 1),
        ("A,C,D", "AC", "MCQ(multiple)", 0.5),
        ("A,C,D", "C", "MCQ(multiple)", 0.25),
        ("A,C,D", "AB", "MCQ(multiple)", 0),
        ("A", "a", "MCQ", 0),
        ("A", r"\text{A}", "MCQ", 1),
        ("A", "AE", "MCQ", 1),
        ("0", "0.01", "Numeric", 1),
        ("0", "0.0100001", "Numeric", 0),
        ("1000000000", "1000000000.05", "Numeric", 0),
        ("2", "2 is the answer", "Integer", 0),
        ("2", "nan", "Numeric", 0),
        ("2", "inf", "Numeric", 0),
    ],
)
def test_jee_preserves_subset_credit_literal_labels_and_absolute_numeric_rule(
    reference, candidate, question_type, expected
):
    result = grade_jee_answer(reference, candidate, question_type)
    assert result.status == Status.SCORED
    assert result.reward == expected


@pytest.mark.parametrize(
    "reference,candidate,question_type",
    [
        ("", "", "MCQ"),
        ("E", "E", "MCQ(multiple)"),
        ("AB", "AB", "MCQ"),
        ("nan", "nan", "Numeric"),
        (True, "1", "Integer"),
    ],
)
def test_jee_malformed_reference_never_scores(reference, candidate, question_type):
    result = grade_jee_answer(reference, candidate, question_type)
    assert result.status == Status.INVALID_TASK
    assert result.reward == 0
