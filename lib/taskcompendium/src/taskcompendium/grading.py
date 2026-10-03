# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract TaskCompendium submission evidence for shared pure candidate graders."""

import json
from dataclasses import dataclass
from enum import StrEnum

from pydantic import JsonValue
from verifyit.candidate import (
    CandidateSpec,
    grade_text_candidate,
    supports_candidate_mode,
)
from verifyit.json_comparison import NumericTypePolicy
from verifyit.modes.grade_predicted_action import grade_predicted_action_candidate
from verifyit.modes.grade_structured_exact import grade_structured_exact_candidate
from verifyit.numeric import NumericCandidateError
from verifyit.spec import (
    ExactSpec,
    McqSpec,
    NumericSpec,
    PredictedActionSpec,
    Spec,
    StructuredExactSpec,
    mode_of,
    spec_to_table,
)
from verifyit.spec import FunctionCall as CandidateCall

from taskcompendium.grading_contract import (
    ActionSubmission,
    GradingAttempt,
    JsonSubmission,
    StateSubmission,
    Submission,
    SubmissionFailure,
    TextSubmission,
    resolve_verifier,
)
from taskcompendium.models import (
    AssistantToolCalls,
    EnvironmentRequirements,
    TaskSpec,
    VerifierSpec,
)
from taskcompendium.submission import Convention, submission_compatibility


class Outcome(StrEnum):
    GRADED = "graded"
    SUBMISSION_FAILURE = "submission_failure"
    INFRA_ERROR = "infra_error"


@dataclass(frozen=True)
class GradeResult:
    status: Outcome
    reward: float | None
    error: str | None = None


def validate_verifier(specification: VerifierSpec) -> None:
    resolve_verifier(specification)


def supports_verifier(specification: VerifierSpec) -> bool:
    if specification.environment_requirements != EnvironmentRequirements() or not supports_candidate_mode(
        specification.kind
    ):
        return False
    validate_verifier(specification)
    return True


def _grade_submission(verifier: CandidateSpec, submission: Submission) -> GradeResult:
    if isinstance(verifier, StructuredExactSpec):
        if not isinstance(submission, (JsonSubmission, StateSubmission)):
            raise TypeError("Structured exact verifier requires a JSON or state submission")
        return GradeResult(Outcome.GRADED, grade_structured_exact_candidate(verifier, submission.value).reward)
    if isinstance(verifier, PredictedActionSpec):
        if not isinstance(submission, ActionSubmission):
            raise TypeError("Predicted-action verifier requires an action submission")
        final = submission.message
        calls = (
            tuple(CandidateCall(call.name, call.arguments) for call in final.calls)
            if isinstance(final, AssistantToolCalls)
            else ()
        )
        return GradeResult(Outcome.GRADED, grade_predicted_action_candidate(verifier, calls).reward)
    if isinstance(verifier, ExactSpec) and isinstance(submission, (JsonSubmission, StateSubmission)):
        if not isinstance(submission.value, str):
            return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, "Text verifier requires a string JSON value")
        return GradeResult(Outcome.GRADED, grade_text_candidate(verifier, submission.value).reward)
    if not isinstance(submission, TextSubmission):
        raise TypeError("Text candidate verifier requires a text submission")
    if isinstance(verifier, McqSpec):
        letter = submission.value.strip()
        if len(letter) != 1 or not "A" <= letter.upper() <= "Z":
            return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, "MCQA response requires one option letter")
    return GradeResult(Outcome.GRADED, grade_text_candidate(verifier, submission.value).reward)


async def grade_answer(specification: TaskSpec, convention: Convention, attempt: GradingAttempt) -> GradeResult:
    """Acquire one submission and score it through the shared candidate contract."""
    verifier = resolve_verifier(specification.verifier)
    compatibility = submission_compatibility(specification, convention)
    if not compatibility.compatible:
        raise ValueError(f"Submission convention is incompatible: {compatibility.reasons}")
    try:
        submission = await convention.extract(attempt)
    except SubmissionFailure as error:
        return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, str(error))
    try:
        return _grade_submission(verifier, submission)
    except NumericCandidateError as error:
        return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, str(error))


def structured_exact(expected: JsonValue, *, numeric_types: NumericTypePolicy = NumericTypePolicy.VALUE) -> VerifierSpec:
    return verifier_descriptor(StructuredExactSpec(expected=expected, numeric_types=numeric_types))


def verifier_descriptor(spec: Spec) -> VerifierSpec:
    """Store a conversion-selected shared verifier contract in the private task slot."""
    parameters = spec_to_table(spec)
    parameters.pop("mode")
    descriptor = VerifierSpec(kind=mode_of(spec), parameters_json=json.dumps(parameters))
    validate_verifier(descriptor)
    return descriptor


def exact_answer(expected: str, ignore_case: bool = True, collapse_whitespace: bool = True) -> VerifierSpec:
    return verifier_descriptor(
        ExactSpec(expected=(expected,), ignore_case=ignore_case, ignore_whitespace=collapse_whitespace)
    )


def numeric_answer(expected: str, *, tolerance_abs: str, tolerance_rel: str) -> VerifierSpec:
    return verifier_descriptor(NumericSpec(expected=expected, tolerance_abs=tolerance_abs, tolerance_rel=tolerance_rel))
