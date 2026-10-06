# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bridge TaskCompendium submissions to verifyit's pure candidate graders.

This module validates verifier payloads by kind, checks task/convention
compatibility, extracts one typed submission, adapts it to verifyit inputs, and
maps rewards and submission failures to GradeResult. verifyit owns verifier-spec
validation, numeric parsing, comparison policies, and score calculation.
Submission conventions own evidence extraction; execution runtimes own provider
decoding, workspace lifecycle, and file-based grading.
"""

import json

from pydantic import JsonValue
from verifyit.candidate import (
    CandidateSpec,
    candidate_spec,
    grade_text_candidate,
    supports_candidate_mode,
)
from verifyit.grade import InvalidTask
from verifyit.json_comparison import NumericTypePolicy
from verifyit.json_objects import unique_object
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
    spec_from_table,
)
from verifyit.spec import FunctionCall as CandidateCall

from taskcompendium.environment import ExternalVerifierSpec, ShellVerifierSpec
from taskcompendium.grader import grader_package
from taskcompendium.grading_contract import (
    ActionSubmission,
    GradingAttempt,
    JsonSubmission,
    StateSubmission,
    Submission,
    SubmissionFailure,
    TextSubmission,
)
from taskcompendium.grading_contract import resolve_verifier as resolve_candidate_verifier
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    AssistantToolCalls,
    SkippedVerifierSpec,
    StageVerifierSpec,
    TaskSpec,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.submission import SubmissionConvention, submission_compatibility


def resolve_verifier(specification: VerifierSpec) -> Spec:
    """Read a candidate or file-based verifyit spec without acquiring runtime evidence."""
    try:
        parameters = json.loads(specification.parameters_json, object_pairs_hook=unique_object)
        if "mode" in parameters:
            raise ValueError("Verifier parameters must not override the mode")
        if supports_candidate_mode(specification.kind):
            return candidate_spec(specification.kind, parameters)
        return spec_from_table({"mode": specification.kind, **parameters})
    except (ValueError, InvalidTask) as error:
        raise ValueError(f"Invalid {specification.kind!r} verifier parameters: {error}") from error


def validate_verifier(specification: VerifierSpec) -> None:
    """Validate the payload for each supported verifier kind."""
    if specification.kind == VerifierKind.SHELL:
        verifier = ShellVerifierSpec.model_validate_json(specification.parameters_json)
        if verifier.artifacts and specification.environment is None:
            raise ValueError("Grading artifacts require a separate private environment")
    elif specification.kind == VerifierKind.EXTERNAL:
        ExternalVerifierSpec.model_validate_json(specification.parameters_json)
    elif specification.kind == VerifierKind.STAGED:
        StageVerifierSpec.model_validate_json(specification.parameters_json)
    elif specification.kind == VerifierKind.SKIPPED:
        SkippedVerifierSpec.model_validate_json(specification.parameters_json)
    else:
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
    verifier = resolve_candidate_verifier(specification.verifier)
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
    descriptor = grader_package(spec)
    validate_verifier(descriptor)
    return descriptor


def exact_answer(expected: str, ignore_case: bool = True, collapse_whitespace: bool = True) -> VerifierSpec:
    return verifier_descriptor(
        ExactSpec(expected=(expected,), ignore_case=ignore_case, ignore_whitespace=collapse_whitespace)
    )


def numeric_answer(expected: str, *, tolerance_abs: float, tolerance_rel: float) -> VerifierSpec:
    return verifier_descriptor(NumericSpec(expected=expected, tolerance_abs=tolerance_abs, tolerance_rel=tolerance_rel))


def skipped_verifier(reason: str) -> VerifierSpec:
    """Describe an explicit rollout-time grading omission."""
    return VerifierSpec(
        kind=VerifierKind.SKIPPED,
        parameters_json=SkippedVerifierSpec(reason=reason).model_dump_json(),
    )
