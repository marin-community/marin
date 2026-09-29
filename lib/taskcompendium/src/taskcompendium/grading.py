# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade submissions with typed private verifiers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from tasktrove_verify.grade import InvalidTask, numeric_tolerance
from tasktrove_verify.modes.grade_exact import grade_exact_candidate
from tasktrove_verify.modes.grade_math import grade_numeric_candidate
from tasktrove_verify.spec import ExactSpec, NumericSpec

from taskcompendium.models import ConversationEvent, VerifierKind, VerifierSpec
from taskcompendium.submission import AnswerFormat, SubmissionConvention, extract_answer

PROVIDER_STATE_PREFIX = "provider:"
RUNTIME_STATE_TARGET = "runtime"


class Outcome(StrEnum):
    GRADED = "graded"
    EXTRACTION_ERROR = "extraction_error"
    INFRA_ERROR = "infra_error"


@dataclass(frozen=True)
class GradeResult:
    status: Outcome
    reward: float | None
    error: str | None = None


@dataclass(frozen=True)
class GradingAttempt:
    """Submission evidence available to a verifier."""

    convention: SubmissionConvention
    conversation: tuple[ConversationEvent, ...]
    environment: object


@runtime_checkable
class StateGrader(Protocol):
    def grade_state(self, state_target: str, expected_state_json: str) -> float: ...


class Verifier(BaseModel, ABC):
    """Validated private configuration that grades one submission."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    @abstractmethod
    def grade(self, attempt: GradingAttempt) -> GradeResult:
        """Grade a submission using this verifier's configuration."""


class ExactAnswerVerifier(Verifier):
    """Compare a text answer using pinned normalization rules."""

    expected: str
    ignore_case: bool = True
    collapse_whitespace: bool = True

    @field_validator("expected")
    @classmethod
    def nonempty_expected(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("An exact answer is required")
        return value

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            candidate = extract_answer(attempt.conversation[-1], attempt.convention)
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        contract = ExactSpec(
            expected=(self.expected,), ignore_case=self.ignore_case, ignore_whitespace=self.collapse_whitespace
        )
        return GradeResult(Outcome.GRADED, grade_exact_candidate(contract, candidate).reward)


class NumericAnswerVerifier(Verifier):
    """Compare a submitted number with explicit absolute and relative tolerances."""

    expected: float
    tolerance_abs: float
    tolerance_rel: float

    @model_validator(mode="after")
    def validate_contract(self) -> "NumericAnswerVerifier":
        contract = NumericSpec(
            expected=self.expected, tolerance_abs=self.tolerance_abs, tolerance_rel=self.tolerance_rel
        )
        try:
            numeric_tolerance(contract)
        except InvalidTask as error:
            raise ValueError(f"Invalid numeric verifier contract: {error}") from error
        return self

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            candidate = extract_answer(attempt.conversation[-1], attempt.convention)
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        try:
            value = float(candidate.strip())
        except ValueError:
            return GradeResult(Outcome.GRADED, 0.0)
        contract = NumericSpec(
            expected=self.expected, tolerance_abs=self.tolerance_abs, tolerance_rel=self.tolerance_rel
        )
        return GradeResult(Outcome.GRADED, grade_numeric_candidate(contract, value).reward)


class StateMatchVerifier(Verifier):
    """Compare authoritative target state with independently constructed gold state."""

    state_target: str
    expected_state_json: str

    @field_validator("state_target")
    @classmethod
    def validate_state_target(cls, value: str) -> str:
        if value != RUNTIME_STATE_TARGET and (
            not value.startswith(PROVIDER_STATE_PREFIX) or not value.removeprefix(PROVIDER_STATE_PREFIX)
        ):
            raise ValueError("State target must be runtime or provider:<name>")
        return value

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        if attempt.convention.answer_format != AnswerFormat.STATE:
            raise ValueError("State grading requires a state submission convention")
        if not isinstance(attempt.environment, StateGrader):
            raise TypeError("State verifier requires a state grader")
        return GradeResult(
            Outcome.GRADED,
            float(attempt.environment.grade_state(self.state_target, self.expected_state_json)),
        )


def state_match(expected_state_json: str, state_target: str) -> VerifierSpec:
    """Construct a private state grader descriptor."""
    verifier = StateMatchVerifier(state_target=state_target, expected_state_json=expected_state_json)
    return VerifierSpec(kind=VerifierKind.STATE_MATCH, parameters_json=verifier.model_dump_json())


def exact_answer(expected: str, ignore_case: bool = True, collapse_whitespace: bool = True) -> VerifierSpec:
    """Construct a pinned exact-answer verifier descriptor."""
    verifier = ExactAnswerVerifier(expected=expected, ignore_case=ignore_case, collapse_whitespace=collapse_whitespace)
    return VerifierSpec(kind=VerifierKind.EXACT_ANSWER, parameters_json=verifier.model_dump_json())


def numeric_answer(expected: float, tolerance_abs: float, tolerance_rel: float) -> VerifierSpec:
    verifier = NumericAnswerVerifier(expected=expected, tolerance_abs=tolerance_abs, tolerance_rel=tolerance_rel)
    return VerifierSpec(kind=VerifierKind.NUMERIC_ANSWER, parameters_json=verifier.model_dump_json())
