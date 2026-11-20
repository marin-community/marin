# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.adapters.completion import Completion, CompletionStatus, answer_text
from verifyit.modes.grade_mcq import answer_letters, grade_mcq_candidate
from verifyit.spec import McqSpec


@pytest.mark.parametrize(
    "completion,expected",
    [
        (Completion("Answer: b", "The answer is A"), 1),
        (Completion("<think>Answer: A</think>Answer: b"), 1),
        (Completion("", "Answer: b"), 1),
        (Completion("", "Answer: b", CompletionStatus.TRUNCATED), 0),
    ],
)
def test_final_completion_choice_overrides_tentative_reasoning(completion, expected):
    letters = answer_letters(answer_text(completion))
    result = grade_mcq_candidate(McqSpec(expected="B"), letters[-1] if letters else "")
    assert result.reward == expected
