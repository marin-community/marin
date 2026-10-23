# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from skyrl_gym.envs.nemotron_ultra.answer_extraction import ReasoningViolation, reasoning_span_violation


@pytest.mark.parametrize(
    "text,violation",
    [
        ("The answer is 4.", None),
        ("<|start_think|>Add.<|end_think|>4", None),
        ("Add.<|end_think|>4", None),
        ("<think>Add.</think>4", None),
        ("<|start_think|>Add.", ReasoningViolation.UNCLOSED),
        ("<|start_think|>A<|end_think|>B<|start_think|>C", ReasoningViolation.UNCLOSED),
        ("<|start_think|>A<|end_think|>B<|start_think|>C<|end_think|>4", ReasoningViolation.SECOND_SPAN),
        ("A<|end_think|>B<|start_think|>C<|end_think|>4", ReasoningViolation.SECOND_SPAN),
        ("<|start_think|>A<|end_think|>B<|end_think|>C", ReasoningViolation.STRAY_END),
        ("A<|end_think|>B<|end_think|>C", ReasoningViolation.STRAY_END),
        ("<think>A</think>B</think>C", ReasoningViolation.STRAY_END),
        ("<think>A</think><|start_think|>B<|end_think|>4", ReasoningViolation.SECOND_SPAN),
    ],
)
def test_reasoning_span_violation(text, violation):
    assert reasoning_span_violation(text) == violation

