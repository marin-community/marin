# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.reasoning_markup import ReasoningViolation, reasoning_span_violation


@pytest.mark.parametrize(
    "text,stop_reason,violation",
    [
        ("The answer is 4.", "stop", None),
        ("<|start_think|>Add.<|end_think|>4", "stop", None),
        ("Add.<|end_think|>4", "stop", None),
        ("<|start_think|>Add.", "stop", ReasoningViolation.UNCLOSED),
        ("<|start_think|>A<|end_think|>B<|start_think|>C<|end_think|>4", "stop", ReasoningViolation.SECOND_SPAN),
        ("A<|end_think|>B<|start_think|>C<|end_think|>4", "stop", ReasoningViolation.SECOND_SPAN),
        ("<|start_think|>A<|end_think|>B<|end_think|>C", "stop", ReasoningViolation.STRAY_END),
        ("A<|end_think|>B<|end_think|>C", "stop", ReasoningViolation.STRAY_END),
        ("<think>A</think><|start_think|>B<|end_think|>4", "stop", ReasoningViolation.SECOND_SPAN),
        ("<think>A</thinking>B</think>4", "stop", ReasoningViolation.STRAY_END),
        # A length stop excuses only the trailing open span.
        ("<think>Add the", "length", None),
        ("<think>A</think>B</think>C", "length", ReasoningViolation.STRAY_END),
        ("<think>A</think>B<think>C", "length", ReasoningViolation.SECOND_SPAN),
        ("<think>A</thinking>B", "length", ReasoningViolation.STRAY_END),
    ],
)
def test_reasoning_span_violation(text, stop_reason, violation):
    assert reasoning_span_violation(text, stop_reason) == violation
