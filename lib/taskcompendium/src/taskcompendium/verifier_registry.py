# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve pinned verifier kinds without import-order-dependent registration."""

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import ValidationError

from taskcompendium.grading import ExactAnswerVerifier, GradeResult, GradingAttempt, NumericAnswerVerifier, Verifier
from taskcompendium.models import ConversationTrace, TaskSpec, VerifierKind, VerifierSpec
from taskcompendium.submission import SubmissionConvention
from taskcompendium.verifiers.multiple_choice import MultipleChoiceVerifier
from taskcompendium.verifiers.predicted_action import PredictedActionVerifier
from taskcompendium.verifiers.script import ScriptVerifier

VERIFIERS: Mapping[VerifierKind, type[Verifier]] = MappingProxyType(
    {
        VerifierKind.EXACT_ANSWER: ExactAnswerVerifier,
        VerifierKind.PREDICTED_ACTION: PredictedActionVerifier,
        VerifierKind.NUMERIC_ANSWER: NumericAnswerVerifier,
        VerifierKind.MCQ_ANSWER: MultipleChoiceVerifier,
    }
)


def resolve_verifier(specification: VerifierSpec) -> Verifier | ScriptVerifier:
    if specification.kind == VerifierKind.SCRIPT:
        try:
            return ScriptVerifier.model_validate_json(specification.parameters_json)
        except ValidationError as error:
            raise ValueError(f"Invalid 'script' verifier parameters: {error}") from error
    verifier_type = VERIFIERS.get(specification.kind)
    if verifier_type is None:
        raise ValueError(f"Unknown verifier kind: {specification.kind!r}")
    try:
        return verifier_type.model_validate_json(specification.parameters_json)
    except ValidationError as error:
        raise ValueError(f"Invalid {specification.kind.value!r} verifier parameters: {error}") from error


def validate_verifier(specification: VerifierSpec) -> None:
    resolve_verifier(specification)


def grade_answer(
    specification: TaskSpec, convention: SubmissionConvention, conversation: ConversationTrace, environment: object
) -> GradeResult:
    verifier = resolve_verifier(specification.verifier)
    if isinstance(verifier, ScriptVerifier):
        raise ValueError("Script verifiers require an isolated Harbor verifier runtime")
    return verifier.grade(GradingAttempt(convention, conversation.events, environment))
