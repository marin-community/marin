# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The reasoning_markup mode credits one closed span and names each malformed shape."""

import pytest
from verifyit.candidate import grade_candidate
from verifyit.reasoning_markup import ReasoningReply
from verifyit.spec import ReasoningMarkupSpec, parse_spec, render_spec


@pytest.mark.parametrize(
    "reply,stop_reason,reward,violation",
    [
        ("<|start_think|>plan<|end_think|>ls -la", "stop", 1.0, None),
        ("Plain reply without any reasoning markers.", "stop", 1.0, None),
        ("<|start_think|>plan<|end_think|>text<|end_think|>more", "stop", 0.0, "stray_end"),
        ("<|start_think|>one<|end_think|><|start_think|>two<|end_think|>", "stop", 0.0, "second_span"),
        ("<|start_think|>never closed", "stop", 0.0, "unclosed"),
        ("<think>plan</think>answer</think>", "stop", 0.0, "stray_end"),
        # The prompt opens no span, so a leading closer is stray.
        ("plan<|end_think|>ls", "tool_calls", 0.0, "stray_end"),
        # A length stop cuts the open span short; it is not a violation.
        ("<|start_think|>still planning", "length", 1.0, None),
    ],
)
def test_reasoning_markup_scores_well_formed_spans_and_names_violations(reply, stop_reason, reward, violation):
    spec = parse_spec(render_spec(ReasoningMarkupSpec()))
    verdict = grade_candidate(spec, ReasoningReply(reply, stop_reason), {})
    assert (verdict.reward, verdict.detail["violation"]) == (reward, violation)


def test_reasoning_markup_scores_an_empty_reply_zero_by_default():
    assert grade_candidate(ReasoningMarkupSpec(), ReasoningReply("", "stop"), {}).reward == 0.0
