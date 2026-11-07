# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from verifyit.candidate import grade_candidate
from verifyit.reasoning_markup import ReasoningReply
from verifyit.spec import ReasoningMarkupSpec


def test_reasoning_markup_flags_a_leading_closer_because_the_prompt_opens_no_span():
    verdict = grade_candidate(ReasoningMarkupSpec(), ReasoningReply("plan<|end_think|>ls", "tool_calls"), {})
    assert (verdict.reward, verdict.detail["violation"]) == (0.0, "stray_end")


def test_reasoning_markup_scores_an_empty_reply_zero_by_default():
    assert grade_candidate(ReasoningMarkupSpec(), ReasoningReply("", "stop"), {}).reward == 0.0
