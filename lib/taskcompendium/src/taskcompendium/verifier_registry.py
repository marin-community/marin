# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve pinned verifier kinds without import-order-dependent registration."""

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import ValidationError

from taskcompendium.grading import (
    ExactAnswerVerifier,
    GradeResult,
    GradingAttempt,
    NumericAnswerVerifier,
    Outcome,
    Verifier,
)
from taskcompendium.models import (
    ConversationTrace,
    EnvironmentRequirements,
    TaskSpec,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.submission import FinalAction, SubmissionConvention
from taskcompendium.verifiers.multiple_choice import MultipleChoiceVerifier
from taskcompendium.verifiers.predicted_action import PredictedActionVerifier

VERIFIERS: Mapping[str, type[Verifier]] = MappingProxyType(
    {
        VerifierKind.EXACT_ANSWER: ExactAnswerVerifier,
        VerifierKind.PREDICTED_ACTION: PredictedActionVerifier,
        VerifierKind.NUMERIC_ANSWER: NumericAnswerVerifier,
        VerifierKind.MCQ_ANSWER: MultipleChoiceVerifier,
    }
)


def resolve_verifier(specification: VerifierSpec) -> Verifier:
    if specification.environment_requirements != EnvironmentRequirements():
        raise NotImplementedError("Pure verifiers cannot satisfy private environment requirements")
    verifier_type = VERIFIERS.get(specification.kind)
    if verifier_type is None:
        raise NotImplementedError(f"Unknown verifier kind: {specification.kind!r}")
    try:
        return verifier_type.model_validate_json(specification.parameters_json)
    except ValidationError as error:
        raise ValueError(f"Invalid {specification.kind!r} verifier parameters: {error}") from error


def validate_verifier(specification: VerifierSpec) -> None:
    resolve_verifier(specification)


def grade_answer(
    specification: TaskSpec, convention: SubmissionConvention, conversation: ConversationTrace, environment: object
) -> GradeResult:
    verifier = resolve_verifier(specification.verifier)
    if isinstance(convention, FinalAction):
        try:
            convention.validate_final_message(conversation.events[-1])
        except ValueError as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
    return verifier.grade(GradingAttempt(convention, conversation.events, environment))
