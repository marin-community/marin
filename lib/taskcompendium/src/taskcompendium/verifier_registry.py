# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve pinned verifier kinds without import-order-dependent registration."""

from collections.abc import Mapping
from types import MappingProxyType

from pydantic import ValidationError

from taskcompendium.environment import ExternalVerifierSpec, ShellVerifierSpec
from taskcompendium.grading import (
    ExactAnswerVerifier,
    GradeResult,
    GradingAttempt,
    NumericAnswerVerifier,
    SkippedVerifier,
    Verifier,
)
from taskcompendium.models import ConversationTrace, StageVerifierSpec, TaskSpec, VerifierKind, VerifierSpec
from taskcompendium.submission import SubmissionConvention
from taskcompendium.verifiers.multiple_choice import MultipleChoiceVerifier
from taskcompendium.verifiers.predicted_action import PredictedActionVerifier

VERIFIERS: Mapping[VerifierKind, type[Verifier]] = MappingProxyType(
    {
        VerifierKind.EXACT_ANSWER: ExactAnswerVerifier,
        VerifierKind.PREDICTED_ACTION: PredictedActionVerifier,
        VerifierKind.NUMERIC_ANSWER: NumericAnswerVerifier,
        VerifierKind.MCQ_ANSWER: MultipleChoiceVerifier,
        VerifierKind.SKIPPED: SkippedVerifier,
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
    if specification.kind == VerifierKind.STAGED:
        StageVerifierSpec.model_validate_json(specification.parameters_json)
        return
    if specification.kind == VerifierKind.EXTERNAL:
        ExternalVerifierSpec.model_validate_json(specification.parameters_json)
        return
    if specification.kind == VerifierKind.SHELL:
        ShellVerifierSpec.model_validate_json(specification.parameters_json)
        return
    resolve_verifier(specification)


def validate_task_verifiers(task: TaskSpec) -> None:
    """Validate the final verifier and every private stage verifier."""
    validate_verifier(task.verifier)
    for stage in task.stages:
        validate_verifier(stage.verifier)


def grade_answer(
    specification: TaskSpec, convention: SubmissionConvention, conversation: ConversationTrace, environment: object
) -> GradeResult:
    verifier = resolve_verifier(specification.verifier)
    return verifier.grade(GradingAttempt(convention, conversation.events, environment))
