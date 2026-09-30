# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve pinned verifier kinds without import-order-dependent registration."""

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import ValidationError

from taskcompendium.grading import (
    ExactAnswerVerifier,
    GradeResult,
    NumericAnswerVerifier,
    Outcome,
    StructuredExactVerifier,
    Verifier,
)
from taskcompendium.models import TaskSpec, VerifierKind, VerifierSpec
from taskcompendium.submission import (
    GradingAttempt,
    SubmissionConvention,
    SubmissionFailure,
    SubmissionFailurePolicy,
)
from taskcompendium.verifiers.multiple_choice import MultipleChoiceVerifier
from taskcompendium.verifiers.predicted_action import PredictedActionVerifier

VERIFIERS: Mapping[VerifierKind, type[Verifier]] = MappingProxyType(
    {
        VerifierKind.EXACT_ANSWER: ExactAnswerVerifier,
        VerifierKind.STRUCTURED_EXACT: StructuredExactVerifier,
        VerifierKind.PREDICTED_ACTION: PredictedActionVerifier,
        VerifierKind.NUMERIC_ANSWER: NumericAnswerVerifier,
        VerifierKind.MCQ_ANSWER: MultipleChoiceVerifier,
    }
)


def resolve_verifier(specification: VerifierSpec) -> Verifier:
    verifier_type = VERIFIERS.get(specification.kind)
    if verifier_type is None:
        raise ValueError(f"Unknown verifier kind: {specification.kind!r}")
    try:
        return verifier_type.model_validate_json(specification.parameters_json)
    except ValidationError as error:
        raise ValueError(f"Invalid {specification.kind.value!r} verifier parameters: {error}") from error


def validate_verifier(specification: VerifierSpec) -> None:
    resolve_verifier(specification)


async def grade_answer(
    specification: TaskSpec, convention: SubmissionConvention, attempt: GradingAttempt
) -> GradeResult:
    """Extract once, then grade the submitted value against a private verifier."""
    try:
        submission = await convention.extract(attempt)
    except SubmissionFailure as error:
        if convention.submission_failure_policy == SubmissionFailurePolicy.ZERO_REWARD:
            return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, str(error))
        raise ValueError(f"Unsupported submission-failure policy: {convention.submission_failure_policy}") from error
    verifier = resolve_verifier(specification.verifier)
    return await verifier.grade(submission, specification=specification.verifier, attempt=attempt)
