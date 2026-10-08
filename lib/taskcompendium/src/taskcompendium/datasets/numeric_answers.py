# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Numeric-answer normalization and exact verifiers."""

from verifyit.numeric import numeric_literal
from verifyit.spec import NumericSpec

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


def normalize_aime24(row: RawRow) -> TaskSpec | ImportRejection:
    problem, answer = row.data.get("problem"), row.data.get("answer")
    if not isinstance(problem, str) or not problem.strip():
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT, reason="missing_prompt", detail="problem must be a nonempty string"
        )
    if not isinstance(answer, str) or not answer.strip().isdigit() or not 0 <= int(answer) <= 999:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="invalid_reference",
            detail="AIME answers must be integers from 0 through 999",
        )
    return TaskSpec(
        id=row.id,
        context=ConversationInput(events=(TextMessage(role="user", content=problem.strip()),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        answer_format=PlainText(),
        grader=verifyit_package(NumericSpec(answer.strip(), tolerance_abs=0.0, tolerance_rel=0.0)).grader,
        source=row.source,
    )


def normalize_svamp(row: RawRow) -> TaskSpec | ImportRejection:
    body, question, answer = (row.data.get(key) for key in ("Body", "Question", "Answer"))
    if not isinstance(body, str) or not body.strip() or not isinstance(question, str) or not question.strip():
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_prompt",
            detail="Body and Question must be nonempty strings",
        )
    if not isinstance(answer, str):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT, reason="invalid_reference", detail="Answer must be a numeric string"
        )
    try:
        numeric_literal(answer)
    except ValueError:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT, reason="invalid_reference", detail="Answer is not numeric"
        )
    return TaskSpec(
        id=row.id,
        context=ConversationInput(events=(TextMessage(role="user", content=f"{body.strip()} {question.strip()}"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        answer_format=PlainText(),
        grader=verifyit_package(NumericSpec(answer.strip(), tolerance_abs=0.0, tolerance_rel=0.0)).grader,
        source=row.source,
    )


AIME24_RUBRIC = ReviewRubric(
    id="competition-math",
    version="1",
    criteria=(
        "Preserve LaTeX, domains, quantifiers, and geometric assumptions. Do not demand decimal reformulation.",
        "The source expects one integer from 0 through 999; leading zeros do not change the integer.",
        "Check whether the premises specify a unique answer. Difficulty alone is not a quality defect.",
    ),
)


def aime24_policy() -> TaskPolicy:
    return TaskPolicy(normalize=normalize_aime24, rubric=AIME24_RUBRIC)


SVAMP_RUBRIC = ReviewRubric(
    id="arithmetic-word-problems",
    version="1",
    criteria=(
        "Identify the quantities and the operation the question actually requests. Check units and directionality.",
        "Ignore irrelevant quantities. Distracting numbers alone do not make a task ambiguous.",
        "Flag contradictions or missing quantities that prevent a unique numeric answer.",
    ),
)


def svamp_policy() -> TaskPolicy:
    return TaskPolicy(normalize=normalize_svamp, rubric=SVAMP_RUBRIC)
