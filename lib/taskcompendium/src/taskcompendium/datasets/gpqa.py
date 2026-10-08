# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPQA Diamond with reproducibly ordered choices and a private option key."""

import hashlib

from verifyit.spec import McqSpec

from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import (
    ImportFailureKind,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPolicy,
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    question = row.data.get("Question")
    choices = [
        row.data.get(key) for key in ("Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")
    ]
    if (
        not isinstance(question, str)
        or not question.strip()
        or not all(isinstance(choice, str) and choice.strip() for choice in choices)
    ):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_prompt_or_choices",
            detail="Question and all four choices are required",
        )
    texts = [str(choice).strip() for choice in choices]
    if len(set(texts)) != len(texts):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="duplicate_options",
            detail="Choice text repeats after boundary trimming",
        )
    # The hash order avoids always exposing the source's correct-first arrangement.
    ordered = sorted(enumerate(texts), key=lambda item: hashlib.sha256(f"{row.id}:{item[0]}".encode()).digest())
    expected = next(chr(65 + index) for index, (original, _) in enumerate(ordered) if original == 0)
    options = "\n".join(f"{chr(65 + index)}. {text}" for index, (_, text) in enumerate(ordered))
    prompt = f"{question.strip()}\n\n{options}\n\nChoose one option letter from A through D."
    return TaskSpec(
        id=row.id,
        context=ConversationInput(events=(TextMessage(role="user", content=prompt),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=verifyit_package(McqSpec(expected, options=len(texts))).grader,
        source=row.source,
    )


RUBRIC = ReviewRubric(
    id="science-mcqa",
    version="1",
    criteria=(
        "Assess every displayed option, units, assumptions, and scientific directionality before judging the key.",
        "Specialist background knowledge is allowed; an omitted experiment, figure, or passage is missing context.",
        "Flag ties, approximate synonyms, or an indefensible key. Technical difficulty alone is not a defect.",
    ),
)


def policy() -> TaskPolicy:
    return TaskPolicy(normalize=normalize, rubric=RUBRIC)
