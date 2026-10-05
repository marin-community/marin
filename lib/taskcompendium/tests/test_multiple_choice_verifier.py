# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A multiple-choice answer verifier independent of a source importer."""

import pytest

from taskcompendium.grading import grade_answer, multiple_choice_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    ConversationTrace,
    EnvironmentRequirements,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.submission import AnswerFormat, SubmissionConvention


@pytest.mark.parametrize(
    "response,reward",
    [("B", 1.0), ("C", 0.0), ("E", 0.0)],
)
def test_hand_authored_multiple_choice_answer(response, reward):
    specification = TaskSpec(
        id="hand-authored-mcq",
        context=ConversationInput(events=(TextMessage(role="user", content="Choose A, B, C, or D."),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        verifier=multiple_choice_answer("B", 4),
        source=Source(dataset="hand-authored", revision="1", row="mcq", importer_revision="1"),
    )
    convention = SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN)

    result = grade_answer(
        specification,
        convention,
        ConversationTrace(events=(*specification.context.events, TextMessage(role="assistant", content=response))),
    )

    assert (result.status, result.reward) == (Outcome.GRADED, reward)
