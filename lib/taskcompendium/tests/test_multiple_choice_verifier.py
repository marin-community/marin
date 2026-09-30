# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A multiple-choice answer verifier independent of a source importer."""

import pytest

from taskcompendium.grading import Outcome
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.submission import GradingAttempt, PlainText
from taskcompendium.verifier_registry import grade_answer
from taskcompendium.verifiers.multiple_choice import multiple_choice_answer


@pytest.mark.parametrize(
    "response,reward",
    [("B", 1.0), ("C", 0.0), ("E", 0.0)],
)
async def test_hand_authored_multiple_choice_answer(response, reward):
    specification = TaskSpec(
        id="hand-authored-mcq",
        context=ConversationInput(events=(TextMessage(role="user", content="Choose A, B, C, or D."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=multiple_choice_answer("B", 4),
        source=Source(dataset="hand-authored", revision="1", row="mcq", importer_revision="1"),
    )
    convention = PlainText(id="plain")

    result = await grade_answer(
        specification,
        convention,
        GradingAttempt(
            ConversationTrace(events=(*specification.context.events, TextMessage(role="assistant", content=response))),
            {},
            object(),
        ),
    )

    assert (result.status, result.reward) == (Outcome.GRADED, reward)
