# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submission extraction and grading across the provider-state boundary."""

import pytest
from pydantic import TypeAdapter

from taskcompendium.grading import Outcome, exact_answer, structured_exact
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.submission import JsonAnswer, ProviderState, SubmissionConvention
from taskcompendium.verifier_registry import grade_answer


class MutableState:
    def __init__(self, state, second_state=None):
        self.state = state
        self.second_state = second_state
        self.reads = 0

    async def provider_state(self, provider):
        assert provider == "workplace"
        self.reads += 1
        return self.state if self.reads == 1 else self.second_state


def _task(expected):
    return TaskSpec(
        id="structured-state",
        context=ConversationInput(events=(TextMessage(role="user", content="Update the records."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.STATE,
        verifier=structured_exact(expected),
        source=Source(dataset="test", revision="1", row="0", importer_revision="1"),
    )


@pytest.mark.parametrize(
    "actual,reward",
    [
        ({"nested": {"right": [1, True, "x"], "left": None}}, 1.0),
        ({"nested": {"right": [True, True, "x"], "left": None}}, 0.0),
        ({"nested": {"right": [True, 1, "x"], "left": None}}, 0.0),
    ],
)
async def test_provider_state_uses_generic_type_strict_structured_grading(actual, reward):
    expected = {"nested": {"left": None, "right": [1, True, "x"]}}
    environment = MutableState(actual, second_state={"unexpected": "second read"})
    convention = TypeAdapter(SubmissionConvention).validate_json(
        ProviderState(id="state", provider="workplace").model_dump_json()
    )
    task = _task(expected)
    trace = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="Done.")))

    result = await grade_answer(task, convention, trace, environment)

    assert (result.status, result.reward) == (Outcome.GRADED, reward)
    assert environment.reads == 1


async def test_provider_state_failure_does_not_score_zero():
    task = _task({"files": []})
    trace = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="Done.")))

    with pytest.raises(TypeError, match="provider state"):
        await grade_answer(task, ProviderState(id="state", provider="workplace"), trace, object())

    with pytest.raises(TypeError, match="not JSON compatible"):
        await grade_answer(task, ProviderState(id="state", provider="workplace"), trace, MutableState({1: "bad key"}))


async def test_invalid_agent_submission_uses_explicit_zero_reward_policy():
    task = _task({}).model_copy(update={"answer_type": AnswerType.TEXT, "verifier": exact_answer("yes")})
    trace = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content='{"answer":')))

    result = await grade_answer(task, JsonAnswer(id="json"), trace, object())

    assert (result.status, result.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)
