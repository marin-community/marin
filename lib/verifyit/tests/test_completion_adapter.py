# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.adapters.completion import Completion, CompletionStatus, answer_text, math_answer_text
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


@pytest.mark.parametrize(
    "completion,expected",
    [
        (Completion(r"\boxed{2}", r"\boxed{1}"), r"\boxed{2}"),
        (Completion("", r"\boxed{2}"), r"\boxed{2}"),
        (Completion("", r"\boxed{2}", CompletionStatus.TRUNCATED), ""),
        (Completion("", "The answer is 2"), ""),
        (Completion(r"<think>\boxed{1}</think>\boxed{2}"), r"\boxed{2}"),
    ],
)
def test_completed_reasoning_only_math_requires_box_and_final_content_wins(completion, expected):
    assert math_answer_text(completion) == expected
