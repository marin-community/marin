# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract TaskCompendium submission evidence for shared pure candidate graders."""

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from verifyit.candidate import (
    CandidateSpec,
    candidate_spec,
    grade_text_candidate,
    supports_candidate_mode,
)
from verifyit.grade import InvalidTask
from verifyit.modes.grade_predicted_action import grade_predicted_action_candidate
from verifyit.spec import ExactSpec, McqSpec, NumericSpec, PredictedActionSpec, Spec, mode_of, spec_to_table
from verifyit.spec import FunctionCall as CandidateCall

from taskcompendium.environment import ExternalVerifierSpec, ShellVerifierSpec
from taskcompendium.models import (
    AssistantToolCalls,
    ConversationTrace,
    EnvironmentRequirements,
    SkippedVerifierSpec,
    StageVerifierSpec,
    TaskSpec,
    TextMessage,
    VerifierKind,
    VerifierSpec,
)
from taskcompendium.submission import AnswerFormat, FinalAction, Submission, extract_answer


class Outcome(StrEnum):
    GRADED = "graded"
    EXTRACTION_ERROR = "extraction_error"
    INFRA_ERROR = "infra_error"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


class GradingFailure(StrEnum):
    TIMEOUT = "timeout"
    MISSING_REWARD = "missing_reward"
    EMPTY_REWARD = "empty_reward"
    INVALID_REWARD = "invalid_reward"
    EXECUTION = "execution"


@dataclass(frozen=True)
class GradeResult:
    status: Outcome
    reward: float | None
    error: str | None = None
    passed: bool | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    failure: GradingFailure | None = None


def resolve_verifier(specification: VerifierSpec) -> CandidateSpec:
    """Validate and return the pure candidate grader for a verifier spec."""
    if specification.environment_requirements != EnvironmentRequirements():
        raise NotImplementedError("Pure verifiers cannot satisfy private environment requirements")
    try:
        return candidate_spec(specification.kind, json.loads(specification.parameters_json))
    except (ValueError, InvalidTask) as error:
        raise ValueError(f"Invalid {specification.kind!r} verifier parameters: {error}") from error


def validate_verifier(specification: VerifierSpec) -> None:
    """Validate the payload for each supported verifier kind."""
    if specification.kind == VerifierKind.SHELL:
        ShellVerifierSpec.model_validate_json(specification.parameters_json)
    elif specification.kind == VerifierKind.EXTERNAL:
        ExternalVerifierSpec.model_validate_json(specification.parameters_json)
    elif specification.kind == VerifierKind.STAGED:
        StageVerifierSpec.model_validate_json(specification.parameters_json)
    elif specification.kind == VerifierKind.SKIPPED:
        SkippedVerifierSpec.model_validate_json(specification.parameters_json)
    else:
        resolve_verifier(specification)


def supports_verifier(specification: VerifierSpec) -> bool:
    if specification.environment_requirements != EnvironmentRequirements() or not supports_candidate_mode(
        specification.kind
    ):
        return False
    validate_verifier(specification)
    return True


def grade_answer(specification: TaskSpec, convention: Submission, conversation: ConversationTrace) -> GradeResult:
    """Score terminal evidence and return its grading status and reward."""
    verifier = resolve_verifier(specification.verifier)
    final = conversation.events[-1]
    if isinstance(convention, FinalAction):
        try:
            convention.validate_final_message(final)
        except ValueError as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
    if isinstance(verifier, PredictedActionSpec):
        if convention.answer_format != AnswerFormat.FINAL_ACTION:
            return GradeResult(Outcome.INFRA_ERROR, None, "Incompatible final-action convention")
        if not isinstance(final, (TextMessage, AssistantToolCalls)):
            return GradeResult(Outcome.INFRA_ERROR, None, "Missing final assistant message")
        calls = (
            tuple(CandidateCall(call.name, call.arguments) for call in final.calls)
            if isinstance(final, AssistantToolCalls)
            else ()
        )
        return GradeResult(Outcome.GRADED, grade_predicted_action_candidate(verifier, calls).reward)
    try:
        candidate = extract_answer(final, convention)
    except (ValueError, TypeError) as error:
        return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
    if isinstance(verifier, McqSpec):
        letter = candidate.strip()
        if len(letter) != 1 or not "A" <= letter.upper() <= "Z":
            return GradeResult(Outcome.EXTRACTION_ERROR, None, "MCQA response requires one option letter")
    return GradeResult(Outcome.GRADED, grade_text_candidate(verifier, candidate).reward)


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


def skipped_verifier(reason: str) -> VerifierSpec:
    """Describe an explicit rollout-time grading omission."""
    return VerifierSpec(
        kind=VerifierKind.SKIPPED,
        parameters_json=SkippedVerifierSpec(reason=reason).model_dump_json(),
    )
