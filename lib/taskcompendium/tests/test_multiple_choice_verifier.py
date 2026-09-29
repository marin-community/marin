# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A multiple-choice answer verifier independent of a source importer."""

import pytest

from taskcompendium.grading import Outcome
from taskcompendium.models import AnswerType, Source, TaskRequirements, TaskSpec
from taskcompendium.submission import AnswerFormat, SubmissionConvention
from taskcompendium.verifier_registry import grade_answer
from taskcompendium.verifiers.multiple_choice import multiple_choice_answer


@pytest.mark.parametrize(
    "response,reward",
    [("B", 1.0), ("C", 0.0), ("E", 0.0)],
)
def test_hand_authored_multiple_choice_answer(response, reward):
    specification = TaskSpec(
        id="hand-authored-mcq",
        instructions="Choose A, B, C, or D.",
        verifier=multiple_choice_answer("B", 4),
        source=Source(dataset="hand-authored", revision="1", row="mcq", importer_revision="1"),
        requirements=TaskRequirements(),
        answer_type=AnswerType.TEXT,
    )
    convention = SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN)

    result = grade_answer(specification, convention, response, object())

    assert (result.status, result.reward) == (Outcome.GRADED, reward)
