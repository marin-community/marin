# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.reasoning_markup import ReasoningViolation, reasoning_span_violation


@pytest.mark.parametrize(
    "text,violation",
    [
        ("The answer is 4.", None),
        ("<|start_think|>Add.<|end_think|>4", None),
        ("Add.<|end_think|>4", None),
        ("<|start_think|>Add.", ReasoningViolation.UNCLOSED),
        ("<|start_think|>A<|end_think|>B<|start_think|>C<|end_think|>4", ReasoningViolation.SECOND_SPAN),
        ("A<|end_think|>B<|start_think|>C<|end_think|>4", ReasoningViolation.SECOND_SPAN),
        ("<|start_think|>A<|end_think|>B<|end_think|>C", ReasoningViolation.STRAY_END),
        ("A<|end_think|>B<|end_think|>C", ReasoningViolation.STRAY_END),
        ("<think>A</think><|start_think|>B<|end_think|>4", ReasoningViolation.SECOND_SPAN),
        ("<think>A</thinking>B</think>4", ReasoningViolation.STRAY_END),
    ],
)
def test_reasoning_span_violation(text, violation):
    assert reasoning_span_violation(text, "stop") == violation


def test_a_length_stop_excuses_only_the_trailing_open_span():
    assert reasoning_span_violation("<think>Add the", "length") is None
    assert reasoning_span_violation("<think>A</think>B</think>C", "length") == ReasoningViolation.STRAY_END
    assert reasoning_span_violation("<think>A</think>B<think>C", "length") == ReasoningViolation.SECOND_SPAN
    assert reasoning_span_violation("<think>A</thinking>B", "length") == ReasoningViolation.STRAY_END
