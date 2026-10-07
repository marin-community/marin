# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bridge TaskCompendium submissions to verifyit's pure candidate graders.

This module checks task/convention compatibility, extracts one typed submission,
adapts it to verifyit inputs, and maps rewards and submission failures to
GradeResult. verifyit owns verifier-spec validation, numeric parsing, comparison
policies, and score calculation. Submission conventions own evidence extraction;
execution runtimes own provider decoding and workspace lifecycle.
"""

import json
import math

from pydantic import JsonValue
from verifyit.candidate import (
    CandidateSpec,
    grade_text_candidate,
    supports_candidate_mode,
)
from verifyit.grade import Reward, Status, scored
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
)
from verifyit.spec import FunctionCall as CandidateCall

from taskcompendium.grader import grader_package
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
from taskcompendium.grading_result import GradeResult, GradingFailure, Outcome
from taskcompendium.models import (
    AssistantToolCalls,
    EnvironmentRequirements,
    TaskSpec,
    VerifierSpec,
)
from taskcompendium.submission import SubmissionConvention, submission_compatibility


def validate_verifier(specification: VerifierSpec) -> None:
    resolve_verifier(specification)


def grade_result(verdict: Reward) -> GradeResult:
    """Normalize VerifyIT outcomes independently of the verifier runtime."""
    if (
        not isinstance(verdict.status, Status)
        or not isinstance(verdict.detail, dict)
        or isinstance(verdict.reward, bool)
        or not isinstance(verdict.reward, int | float)
        or not 0 <= verdict.reward <= 1
        or not math.isfinite(verdict.reward)
        or (verdict.status != Status.SCORED and verdict.reward != 0)
    ):
        raise ValueError("Invalid verifier verdict")
    if verdict.status == Status.SCORED:
        status = (
            Outcome.SUBMISSION_FAILURE if verdict.detail.get("reason") == "invalid_numeric_candidate" else Outcome.GRADED
        )
        return GradeResult(status, float(verdict.reward), verdict.detail.get("error"), verdict.detail)
    status = Outcome.INVALID_TASK if verdict.status == Status.INVALID_TASK else Outcome.INFRA_ERROR
    return GradeResult(status, None, verdict.detail.get("error"), verdict.detail)


def parse_grade_result(data: bytes) -> GradeResult:
    """Decode an isolated verifier verdict through the shared grading contract."""
    try:
        verdict = json.loads(data)
        if not isinstance(verdict, dict):
            raise ValueError("Verifier verdict must be an object")
        return grade_result(Reward(verdict["reward"], Status(verdict["status"]), verdict["detail"]))
    except (KeyError, TypeError, ValueError):
        return GradeResult(Outcome.INFRA_ERROR, None, "Invalid verifier verdict", failure=GradingFailure.INVALID_REWARD)


def _grade_submission(verifier: CandidateSpec, submission: Submission) -> GradeResult:
    match verifier, submission:
        case StructuredExactSpec(), JsonSubmission(value=value) | StateSubmission(value=value):
            return grade_result(grade_structured_exact_candidate(verifier, value))
        case PredictedActionSpec(), ActionSubmission(message=final):
            calls = (
                tuple(CandidateCall(call.name, call.arguments) for call in final.calls)
                if isinstance(final, AssistantToolCalls)
                else ()
            )
            return grade_result(grade_predicted_action_candidate(verifier, calls))
        case ExactSpec(), JsonSubmission(value=value) | StateSubmission(value=value):
            if not isinstance(value, str):
                return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, "Text verifier requires a string JSON value")
            return grade_result(grade_text_candidate(verifier, value))
        case ExactSpec() | NumericSpec() | McqSpec(), TextSubmission(value=value):
            return grade_result(grade_text_candidate(verifier, value))
        case StructuredExactSpec(), _:
            raise TypeError("Structured exact verifier requires a JSON or state submission")
        case PredictedActionSpec(), _:
            raise TypeError("Predicted-action verifier requires an action submission")
        case _:
            raise TypeError("Text candidate verifier requires a text submission")


def grade_answer(specification: TaskSpec, convention: SubmissionConvention, attempt: GradingAttempt) -> GradeResult:
    """Extract one submission and score it through the shared candidate contract."""
    if specification.verifier.environment_requirements != EnvironmentRequirements():
        raise NotImplementedError("Pure grading cannot satisfy private environment requirements")
    if not supports_candidate_mode(specification.verifier.kind):
        raise NotImplementedError("This verifier requires runtime grading")
    verifier = resolve_verifier(specification.verifier)
    assert isinstance(verifier, CandidateSpec)
    compatibility = submission_compatibility(specification, convention)
    if not compatibility.compatible:
        raise ValueError(f"Submission convention is incompatible: {compatibility.reasons}")
    try:
        submission = convention.extract(attempt)
    except SubmissionFailure as error:
        return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, str(error))
    try:
        return _grade_submission(verifier, submission)
    except NumericCandidateError as error:
        return grade_result(scored(0.0, reason="invalid_numeric_candidate", error=str(error)))


def structured_exact(expected: JsonValue, *, numeric_types: NumericTypePolicy = NumericTypePolicy.VALUE) -> VerifierSpec:
    return verifier_descriptor(StructuredExactSpec(expected=expected, numeric_types=numeric_types))


def verifier_descriptor(spec: Spec) -> VerifierSpec:
    """Store a conversion-selected shared verifier contract in the private task slot."""
    descriptor = grader_package(spec).verifier
    validate_verifier(descriptor)
    return descriptor


def exact_answer(expected: str, ignore_case: bool = True, collapse_whitespace: bool = True) -> VerifierSpec:
    return verifier_descriptor(
        ExactSpec(expected=(expected,), ignore_case=ignore_case, ignore_whitespace=collapse_whitespace)
    )


def numeric_answer(expected: str, *, tolerance_abs: float, tolerance_rel: float) -> VerifierSpec:
    return verifier_descriptor(NumericSpec(expected=expected, tolerance_abs=tolerance_abs, tolerance_rel=tolerance_rel))
