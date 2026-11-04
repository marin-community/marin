# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The reasoning_markup mode credits one closed span and names each malformed shape."""

import pytest
from verifyit.candidate import grade_candidate
from verifyit.spec import ReasoningMarkupSpec, parse_spec, render_spec


@pytest.mark.parametrize(
    "reply,reward,violation",
    [
        ("<|start_think|>plan<|end_think|>ls -la", 1.0, None),
        ("Plain reply without any reasoning markers.", 1.0, None),
        ("<|start_think|>plan<|end_think|>text<|end_think|>more", 0.0, "stray_end"),
        ("<|start_think|>one<|end_think|><|start_think|>two<|end_think|>", 0.0, "second_span"),
        ("<|start_think|>never closed", 0.0, "unclosed"),
        ("<think>plan</think>answer</think>", 0.0, "stray_end"),
    ],
)
def test_reasoning_markup_scores_well_formed_spans_and_names_violations(reply, reward, violation):
    spec = parse_spec(render_spec(ReasoningMarkupSpec()))
    verdict = grade_candidate(spec, reply, {})
    assert (verdict.reward, verdict.detail["violation"]) == (reward, violation)


def test_reasoning_markup_scores_an_empty_reply_zero_by_default():
    assert grade_candidate(ReasoningMarkupSpec(), "", {}).reward == 0.0
