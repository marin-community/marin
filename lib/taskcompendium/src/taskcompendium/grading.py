# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve private verifier kinds to typed grading handlers."""

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import entry_points
from typing import Generic, TypeVar, cast

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from taskcompendium.models import TaskSpec, VerifierSpec
from taskcompendium.submission import SubmissionConvention, extract_answer

ENTRY_POINT_GROUP = "taskcompendium.verifiers"
EXACT_ANSWER_KIND = "exact_answer"
PayloadT = TypeVar("PayloadT", bound=BaseModel)
WHITESPACE = re.compile(r"\s+")


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
    """Submission evidence available to a registered verifier."""

    convention: SubmissionConvention
    response: str | None
    environment: object


@dataclass(frozen=True)
class VerifierHandler(Generic[PayloadT]):
    payload_type: type[PayloadT]
    grade: Callable[[PayloadT, GradingAttempt], GradeResult]


def _handler(kind: str) -> VerifierHandler[BaseModel]:
    matches = [entry for entry in entry_points(group=ENTRY_POINT_GROUP) if entry.name == kind]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous verifier kind: {kind!r}")
    factory = cast(Callable[[], VerifierHandler[BaseModel]], matches[0].load())
    return factory()


def _resolved(verifier: VerifierSpec) -> tuple[VerifierHandler[BaseModel], BaseModel]:
    handler = _handler(verifier.kind)
    try:
        payload = handler.payload_type.model_validate(verifier.parameters)
    except ValidationError as error:
        raise ValueError(f"Invalid {verifier.kind!r} verifier parameters: {error}") from error
    return handler, payload


def validate_verifier(verifier: VerifierSpec) -> None:
    """Check a private verifier before export or launch, independent of submission convention."""
    _resolved(verifier)


def grade_attempt(specification: TaskSpec, attempt: GradingAttempt) -> GradeResult:
    """Dispatch an attempt by its private verifier kind."""
    handler, payload = _resolved(specification.verifier)
    return handler.grade(payload, attempt)


def grade_answer(
    specification: TaskSpec, convention: SubmissionConvention, response: str | None, environment: object
) -> GradeResult:
    """Extract and score a response through the verifier registry."""
    return grade_attempt(specification, GradingAttempt(convention, response, environment))


class ExactAnswerPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    expected: str
    ignore_case: bool = True
    collapse_whitespace: bool = True

    @field_validator("expected")
    @classmethod
    def nonempty_expected(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("An exact answer is required")
        return value


def exact_answer(expected: str, ignore_case: bool = True, collapse_whitespace: bool = True) -> VerifierSpec:
    """Construct a pinned exact-answer verifier descriptor."""
    payload = ExactAnswerPayload(expected=expected, ignore_case=ignore_case, collapse_whitespace=collapse_whitespace)
    return VerifierSpec(kind=EXACT_ANSWER_KIND, parameters=payload.model_dump())


def _normalize_exact(value: str, payload: ExactAnswerPayload) -> str:
    value = WHITESPACE.sub(" ", value).strip() if payload.collapse_whitespace else value.strip()
    return value.casefold() if payload.ignore_case else value


def _grade_exact(payload: ExactAnswerPayload, attempt: GradingAttempt) -> GradeResult:
    try:
        candidate = extract_answer(attempt.response, attempt.convention)
    except (ValueError, TypeError) as error:
        return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
    match = _normalize_exact(candidate, payload) == _normalize_exact(payload.expected, payload)
    return GradeResult(Outcome.GRADED, float(match))


def exact_answer_handler() -> VerifierHandler[ExactAnswerPayload]:
    return VerifierHandler(ExactAnswerPayload, _grade_exact)
