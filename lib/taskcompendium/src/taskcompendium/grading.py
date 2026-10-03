# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract TaskCompendium submission evidence for shared pure candidate graders."""

import json
from dataclasses import dataclass
from enum import StrEnum

from pydantic import JsonValue
from tasktrove_verify.candidate import (
    CandidateSpec,
    candidate_spec,
    grade_ejection_candidate,
    grade_text_candidate,
    supports_candidate_mode,
)
from tasktrove_verify.grade import InvalidTask
from tasktrove_verify.modes.grade_predicted_action import grade_predicted_action_candidate
from tasktrove_verify.modes.grade_structured_exact import grade_structured_exact_candidate
from tasktrove_verify.spec import (
    ExactSpec,
    McqSpec,
    NumericSpec,
    PredictedActionSpec,
    Spec,
    StructuredExactSpec,
    UnsolvableSpec,
    mode_of,
    spec_to_table,
)
from tasktrove_verify.spec import FunctionCall as CandidateCall

from taskcompendium.models import (
    AssistantToolCalls,
    EnvironmentRequirements,
    TaskSpec,
    VerifierSpec,
)
from taskcompendium.submission import (
    EJECT_CALL_NAME,
    ActionSubmission,
    GradingAttempt,
    StateSubmission,
    Submission,
    SubmissionConvention,
    SubmissionFailure,
    TextSubmission,
    extract_submission,
    is_ejection,
    submission_compatibility,
)


class Outcome(StrEnum):
    GRADED = "graded"
    SUBMISSION_FAILURE = "submission_failure"
    INFRA_ERROR = "infra_error"


@dataclass(frozen=True)
class GradeResult:
    status: Outcome
    reward: float | None
    error: str | None = None


def resolve_verifier(specification: VerifierSpec) -> CandidateSpec:
    """Read a shared verifier spec without any TaskCompendium registration step."""
    if specification.environment_requirements != EnvironmentRequirements():
        raise NotImplementedError("Pure verifiers cannot satisfy private environment requirements")
    try:
        return candidate_spec(specification.kind, json.loads(specification.parameters_json))
    except (ValueError, InvalidTask) as error:
        raise ValueError(f"Invalid {specification.kind!r} verifier parameters: {error}") from error


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
    if is_ejection(submission):
        return GradeResult(Outcome.GRADED, grade_ejection_candidate(verifier).reward)
    if isinstance(verifier, UnsolvableSpec):
        return GradeResult(Outcome.GRADED, 0.0)
    if isinstance(verifier, StructuredExactSpec):
        if not isinstance(submission, StateSubmission):
            raise TypeError("Structured exact verifier requires a state submission")
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
    if isinstance(verifier, ExactSpec) and isinstance(submission, StateSubmission) and isinstance(submission.value, str):
        return GradeResult(Outcome.GRADED, grade_text_candidate(verifier, submission.value).reward)
    if not isinstance(submission, TextSubmission):
        raise TypeError("Text candidate verifier requires a text submission")
    if isinstance(verifier, McqSpec):
        letter = submission.value.strip()
        if len(letter) != 1 or not "A" <= letter.upper() <= "Z":
            return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, "MCQA response requires one option letter")
    return GradeResult(Outcome.GRADED, grade_text_candidate(verifier, submission.value).reward)


async def grade_answer(
    specification: TaskSpec, convention: SubmissionConvention, attempt: GradingAttempt
) -> GradeResult:
    """Acquire one submission and score it through the shared candidate contract."""
    verifier = resolve_verifier(specification.verifier)
    if isinstance(verifier, UnsolvableSpec):
        compatibility = submission_compatibility(specification, convention)
        if not compatibility.compatible:
            raise ValueError(f"Submission convention is incompatible: {'; '.join(compatibility.reasons)}")
    try:
        # Source history is input, not a submission made during this interaction.
        for event in attempt.conversation.events[len(specification.context.events) : -1]:
            if isinstance(event, AssistantToolCalls) and any(call.name == EJECT_CALL_NAME for call in event.calls):
                raise SubmissionFailure("The interaction continued after eject_button")
        submission = await extract_submission(convention, attempt)
        if is_ejection(submission) and not any(tool.name == EJECT_CALL_NAME for tool in specification.final_tools):
            raise SubmissionFailure("Ejection was not advertised for this task")
    except SubmissionFailure as error:
        return GradeResult(Outcome.SUBMISSION_FAILURE, 0.0, str(error))
    return _grade_submission(verifier, submission)


def structured_exact(expected: JsonValue) -> VerifierSpec:
    return verifier_descriptor(StructuredExactSpec(expected=expected))


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


def numeric_answer(expected: float, tolerance_abs: float, tolerance_rel: float) -> VerifierSpec:
    return verifier_descriptor(NumericSpec(expected=expected, tolerance_abs=tolerance_abs, tolerance_rel=tolerance_rel))


def unsolvable() -> VerifierSpec:
    return verifier_descriptor(UnsolvableSpec())
