# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Completion channel cases from Evalchemy's original MMLU-Pro grader tests."""

import pytest
from marin.evaluation.completion import completion_text
from verifyit.adapters.evalchemy_choices import extract_gpqa_answer, extract_mmlu_answer
from verifyit.adapters.evalchemy_math import grade_boxed_math


@pytest.mark.parametrize(
    "message,finish_reason,mmlu_answer,gpqa_answer",
    [
        ({"content": None, "reasoning_content": "B. This is a valid explanation."}, "stop", "B", "A"),
        ({"content": None, "reasoning_content": "B. This is a valid explanation."}, "length", None, "A"),
        ({"content": "Answer: C", "reasoning_content": "The answer is A"}, "length", "C", "C"),
        ({"content": "", "reasoning": "The answer is (d)."}, "stop", "D", "D"),
    ],
)
def test_source_completion_policy_preserves_grading(message, finish_reason, mmlu_answer, gpqa_answer):
    assert extract_mmlu_answer(completion_text(message, finish_reason, "final_or_complete_reasoning")) == mmlu_answer
    assert extract_gpqa_answer(completion_text(message, finish_reason, "combine")) == gpqa_answer


def test_combined_response_keeps_source_channel_separator():
    assert (
        completion_text({"content": "Final", "reasoning_content": "Reasoning"}, "stop", "combine")
        == "Reasoning\n\nFinal"
    )


@pytest.mark.parametrize(
    "message,finish_reason,score",
    [
        ({"content": "No final answer", "reasoning_content": r"\boxed{7}"}, "stop", 0),
        ({"content": "", "reasoning_content": r"\boxed{7}"}, "stop", 1),
        ({"content": "", "reasoning_content": r"\boxed{7}"}, "length", 0),
        ({"content": "", "reasoning_content": r"\boxed 7$"}, "stop", 0),
    ],
)
def test_boxed_math_preserves_final_channel_and_complete_reasoning_rules(message, finish_reason, score):
    text = completion_text(message, finish_reason, "final_or_complete_boxed_reasoning")
    assert grade_boxed_math("7", text).reward == score
