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

from pydantic import JsonValue
from verifyit.candidate import (
    CandidateSpec,
    grade_text_candidate,
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
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    AssistantToolCalls,
    TaskSpec,
    VerifierSpec,
)
from taskcompendium.submission import SubmissionConvention, submission_compatibility


def validate_verifier(specification: VerifierSpec) -> None:
    resolve_verifier(specification)


def _grade_submission(verifier: CandidateSpec, submission: Submission) -> GradeResult:
    match verifier, submission:
        case StructuredExactSpec(), JsonSubmission(value=value) | StateSubmission(value=value):
            return GradeResult(Outcome.GRADED, grade_structured_exact_candidate(verifier, value).reward)
        case PredictedActionSpec(), ActionSubmission(message=final):
            calls = (
                tuple(CandidateCall(call.name, call.arguments) for call in final.calls)
                if isinstance(final, AssistantToolCalls)
                else ()
            )
            return GradeResult(Outcome.GRADED, grade_predicted_action_candidate(verifier, calls).reward)
        case ExactSpec(), JsonSubmission(value=value) | StateSubmission(value=value):
            if not isinstance(value, str):
                return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, "Text verifier requires a string JSON value")
            return GradeResult(Outcome.GRADED, grade_text_candidate(verifier, value).reward)
        case ExactSpec() | NumericSpec() | McqSpec(), TextSubmission(value=value):
            return GradeResult(Outcome.GRADED, grade_text_candidate(verifier, value).reward)
        case StructuredExactSpec(), _:
            raise TypeError("Structured exact verifier requires a JSON or state submission")
        case PredictedActionSpec(), _:
            raise TypeError("Predicted-action verifier requires an action submission")
        case _:
            raise TypeError("Text candidate verifier requires a text submission")


def grade_answer(specification: TaskSpec, convention: SubmissionConvention, attempt: GradingAttempt) -> GradeResult:
    """Extract one submission and score it through the shared candidate contract."""
    verifier = resolve_verifier(specification.verifier)
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


def numeric_answer(expected: str, *, tolerance_abs: float, tolerance_rel: float) -> VerifierSpec:
    return verifier_descriptor(NumericSpec(expected=expected, tolerance_abs=tolerance_abs, tolerance_rel=tolerance_rel))
