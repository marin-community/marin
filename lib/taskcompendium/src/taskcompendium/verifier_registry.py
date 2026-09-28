# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve pinned verifier kinds without import-order-dependent registration."""

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import ValidationError

from taskcompendium.grading import EXACT_ANSWER_KIND, ExactAnswerVerifier, GradeResult, GradingAttempt, Verifier
from taskcompendium.models import TaskSpec, VerifierSpec
from taskcompendium.submission import SubmissionConvention
from taskcompendium.verifiers.tasktrove_mcqa import TASKTROVE_MCQA_KIND, TaskTroveMcqaVerifier

VERIFIERS: Mapping[str, type[Verifier]] = MappingProxyType(
    {EXACT_ANSWER_KIND: ExactAnswerVerifier, TASKTROVE_MCQA_KIND: TaskTroveMcqaVerifier}
)


def resolve_verifier(specification: VerifierSpec) -> Verifier:
    verifier_type = VERIFIERS.get(specification.kind)
    if verifier_type is None:
        raise ValueError(f"Unknown verifier kind: {specification.kind!r}")
    try:
        return verifier_type.model_validate_json(specification.parameters_json)
    except ValidationError as error:
        raise ValueError(f"Invalid {specification.kind!r} verifier parameters: {error}") from error


def validate_verifier(specification: VerifierSpec) -> None:
    resolve_verifier(specification)


def grade_answer(
    specification: TaskSpec, convention: SubmissionConvention, response: str | None, environment: object
) -> GradeResult:
    verifier = resolve_verifier(specification.verifier)
    return verifier.grade(GradingAttempt(convention, response, environment))
