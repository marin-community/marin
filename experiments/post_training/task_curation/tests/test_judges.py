# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from taskcompendium.pipeline.reviewer_spec import JudgeEndpoint
from verifyit.grade import Status
from verifyit.modes.grade_judge import JudgeConnection, grade_judge_candidate
from verifyit.spec import parse_spec, render_spec

from experiments.post_training.task_curation import judges
from experiments.post_training.task_curation.judges import (
    JUDGE_TIERS,
    HostedJudgeEndpoint,
    JudgeTier,
    RelayJudgeEndpoint,
)
from experiments.post_training.task_curation.tests.review_server import JudgeReply

# The local Iris serving tests (LocalJudgeEndpoint lifecycle) run on the atlas
# lanes, whose marin.inference carries the serving API they drive; this catalog
# branch consumes only the tier specifications and the hosted/relay contracts.


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


@pytest.mark.parametrize("bare_model_list", [False, True])
def test_hosted_endpoint_requires_exact_model_before_grading(review_server, bare_model_list):
    review_server.bare_model_list = bare_model_list
    review_server.model_ids = ["MiniMaxAI/MiniMax-M3.1-Flash"]
    endpoint: JudgeEndpoint = HostedJudgeEndpoint(
        JUDGE_TIERS[JudgeTier.MEDIUM].spec.model, f"http://127.0.0.1:{review_server.server_port}/v1", "fixture-key"
    )
    with pytest.raises(ValueError, match="does not advertise"):
        with endpoint.connect():
            pytest.fail("A different MiniMax model cannot instantiate the MEDIUM tier")

    review_server.model_ids.append(JUDGE_TIERS[JudgeTier.MEDIUM].spec.model)
    review_server.judge_replies = [JudgeReply("CORRECT")]
    with endpoint.connect() as connection:
        spec = replace(JUDGE_TIERS[JudgeTier.MEDIUM].spec, question="Two plus two?", references=("Four.",))
        result = grade_judge_candidate(spec, "Four.", connection=connection, runtime=None)
    assert result.reward == 1
    assert connection.default_model == spec.model


def test_relay_endpoint_uses_registered_job_url_and_checks_model(review_server, monkeypatch):
    base_url = f"http://127.0.0.1:{review_server.server_port}/v1"
    discovered_jobs = []

    def resolve(relay_job):
        discovered_jobs.append(relay_job)
        return base_url

    monkeypatch.setattr(judges, "resolve_glm_base_url", resolve)
    endpoint: JudgeEndpoint = RelayJudgeEndpoint("/fixture/existing-relay", "fixture-reviewer-model", "fixture-key")
    review_server.model_ids = ["glm-5.3"]
    with pytest.raises(ValueError, match="does not advertise"):
        with endpoint.connect():
            pytest.fail("The relay must advertise the reviewer's pinned model")
    review_server.model_ids = ["fixture-reviewer-model"]
    with endpoint.connect() as connection:
        assert connection.base_url == base_url
        assert connection.default_model == "fixture-reviewer-model"
    assert discovered_jobs == ["/fixture/existing-relay"] * 2
