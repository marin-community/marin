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
from taskcompendium.submission import GradingAttempt, ProviderState, SubmissionConvention
from taskcompendium.verifier_registry import grade_answer


class MutableState:
    def __init__(self, state, second_state=None):
        self.state = state
        self.second_state = second_state
        self.reads = 0

    def canonical_state(self):
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

    result = await grade_answer(task, convention, GradingAttempt(trace, {"workplace": environment}, object()))

    assert (result.status, result.reward) == (Outcome.GRADED, reward)
    assert environment.reads == 1


async def test_exact_answer_grades_string_provider_state():
    task = _task("complete").model_copy(update={"verifier": exact_answer("complete")})
    trace = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="Done.")))

    result = await grade_answer(
        task,
        ProviderState(id="state", provider="workplace"),
        GradingAttempt(trace, {"workplace": MutableState("complete")}, object()),
    )

    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)


async def test_provider_state_failure_does_not_score_zero():
    task = _task({"files": []})
    trace = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content="Done.")))

    with pytest.raises(KeyError, match="workplace"):
        await grade_answer(task, ProviderState(id="state", provider="workplace"), GradingAttempt(trace, {}, object()))

    with pytest.raises(TypeError, match="not JSON compatible"):
        await grade_answer(
            task,
            ProviderState(id="state", provider="workplace"),
            GradingAttempt(trace, {"workplace": MutableState({1: "bad key"})}, object()),
        )
