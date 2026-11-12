# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from verifyit.grade import Status
from verifyit.modes.grade_judge import JudgeConnection, grade_judge_candidate
from verifyit.spec import parse_spec, render_spec

from experiments.post_training.task_curation.judges import JUDGE_TIERS, JudgeTier
from experiments.post_training.task_curation.tests.review_server import JudgeReply


@pytest.mark.parametrize("tier", list(JudgeTier))
def test_serialized_tier_grades_reference_and_wrong_answers(review_server, tier):
    definition = JUDGE_TIERS[tier]
    spec = replace(definition.spec, question="What is two plus two?", references=("The complete answer is four.",))
    restored = parse_spec(render_spec(spec))
    connection = JudgeConnection(f"http://127.0.0.1:{review_server.server_port}/v1", "fixture-key")
    review_server.judge_replies = [JudgeReply("CORRECT"), JudgeReply("INCORRECT")]

    positive = grade_judge_candidate(restored, "Four.", connection=connection, runtime=None)
    negative = grade_judge_candidate(restored, "Seven.", connection=connection, runtime=None)

    assert (positive.status, positive.reward) == (Status.SCORED, 1.0)
    assert (negative.status, negative.reward) == (Status.SCORED, 0.0)
    for request in review_server.requests:
        assert request["model"] == spec.model
        assert request["temperature"] == 0.7
        assert request["max_completion_tokens"] == spec.max_completion_tokens
        assert [message["role"] for message in request["messages"]] == ["system", "user"]
        if spec.chat_template_kwargs:
            assert request["chat_template_kwargs"] == spec.chat_template_kwargs
        if spec.reasoning_effort:
            assert request["reasoning_effort"] == spec.reasoning_effort


def test_tier_parse_retry_repeats_the_same_turn_and_retains_failure(review_server):
    spec = replace(JUDGE_TIERS[JudgeTier.LIGHT].spec, question="Two plus two?", references=("Four.",))
    connection = JudgeConnection(f"http://127.0.0.1:{review_server.server_port}/v1", "fixture-key")
    review_server.judge_replies = [JudgeReply("Not a verdict"), JudgeReply("CORRECT")]

    result = grade_judge_candidate(spec, "Four.", connection=connection, runtime=None)

    assert result.reward == 1.0
    assert result.detail["previous_parse_attempts"] == [{"attempt": 1, "completion": "Not a verdict"}]
    assert len(review_server.requests) == 2
    assert review_server.requests[0]["messages"] == review_server.requests[1]["messages"]


def test_serialized_labels_context_limit_preserves_candidate_text(review_server):
    spec = replace(
        JUDGE_TIERS[JudgeTier.MEDIUM].spec,
        references=("reference",),
        system_prompt="",
        prompt_template="{reference}\n{context}\n{candidate}",
        context_char_limit=4,
    )
    restored = parse_spec(render_spec(spec))
    connection = JudgeConnection(f"http://127.0.0.1:{review_server.server_port}/v1", "fixture-key")
    review_server.judge_replies = [JudgeReply("CORRECT")]

    result = grade_judge_candidate(
        restored, "complete candidate", context="ééééexcluded", connection=connection, runtime=None
    )

    assert result.reward == 1.0
    assert review_server.requests[0]["messages"] == [{"role": "user", "content": "reference\néééé\ncomplete candidate"}]
