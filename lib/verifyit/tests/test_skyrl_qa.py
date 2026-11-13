# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.adapters.skyrl_qa import grade_qa_exact, grade_qa_token_sets
from verifyit.grade import InvalidTask, Status


def test_qa_exact_keeps_literal_normalized_alternatives():
    assert grade_qa_exact("paris", ["london", "paris"]).reward == 1
    assert grade_qa_exact("PARIS", ["paris"]).reward == 0
    assert grade_qa_exact("paris ", ["paris"]).reward == 0
    assert grade_qa_exact("", [""]).reward == 1


@pytest.mark.parametrize("references", [[], None, ["paris", None]])
def test_qa_exact_invalid_alternatives_are_not_successful_zero_verdicts(references):
    with pytest.raises(InvalidTask):
        grade_qa_exact("paris", references)


@pytest.mark.parametrize(
    ("candidate", "reference", "expected"),
    [
        ({"red", "fox"}, {"fox", "jumps"}, 0.5),
        ({"red", "fox"}, {"fox"}, 2 / 3),
        ({"北", "京", "123"}, {"北", "京", "124"}, 2 / 3),
        ({"False"}, {"false"}, 0),
        (set(), set(), 0),
    ],
)
def test_qa_f1_preserves_fractional_set_overlap(candidate, reference, expected):
    result = grade_qa_token_sets(candidate, reference)
    assert result.status is Status.SCORED
    assert result.reward == pytest.approx(expected)


def test_qa_f1_rejects_duplicate_bearing_sequence_instead_of_awarding_inflated_overlap():
    with pytest.raises(InvalidTask, match="token sets"):
        grade_qa_token_sets(["fox", "fox"], {"fox"})
