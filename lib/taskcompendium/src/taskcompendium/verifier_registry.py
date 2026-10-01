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
)
from taskcompendium.verifiers.mathematical import MathematicalAnswerVerifier
from taskcompendium.verifiers.multiple_choice import MultipleChoiceVerifier
from taskcompendium.verifiers.predicted_action import PredictedActionVerifier
from taskcompendium.verifiers.script import ScriptVerifier

VERIFIERS: Mapping[VerifierKind, type[Verifier]] = MappingProxyType(
    {
        VerifierKind.EXACT_ANSWER: ExactAnswerVerifier,
        VerifierKind.STRUCTURED_EXACT: StructuredExactVerifier,
        VerifierKind.PREDICTED_ACTION: PredictedActionVerifier,
        VerifierKind.NUMERIC_ANSWER: NumericAnswerVerifier,
        VerifierKind.MATHEMATICAL_ANSWER: MathematicalAnswerVerifier,
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


def validate_launch_parallel_tool_calls(specification: VerifierSpec, parallel_tool_calls: bool | None) -> None:
    """Reject a launch that cannot emit the task's expected final action."""
    verifier = resolve_verifier(specification)
    if (
        parallel_tool_calls is False
        and isinstance(verifier, PredictedActionVerifier)
        and len(verifier.expected_calls) > 1
    ):
        raise ValueError("Launch disables parallel calls required by the task")


async def grade_answer(
    specification: TaskSpec, convention: SubmissionConvention, attempt: GradingAttempt
) -> GradeResult:
    """Grade a task attempt, assigning zero reward to invalid agent submissions."""
    verifier = resolve_verifier(specification.verifier)
    if isinstance(verifier, ScriptVerifier):
        raise ValueError("Script verifiers require an isolated Harbor verifier runtime")
    try:
        submission = await convention.extract(attempt)
    except SubmissionFailure as error:
        return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, str(error))
    return await verifier.grade(submission, attempt=attempt)
