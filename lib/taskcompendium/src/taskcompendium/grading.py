# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve private verifier kinds to typed grading handlers."""

import dataclasses
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import msgspec
from tasktrove_verify.grade import Status, grade
from tasktrove_verify.spec import ExactSpec

from taskcompendium.models import TaskSpec, VerifierSpec
from taskcompendium.rendering import Rendering, extract_answer

ENTRY_POINT_GROUP = "taskcompendium.verifiers"
EXACT_ANSWER_KIND = "exact_answer"
PayloadT = TypeVar("PayloadT")


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

    rendering: Rendering
    response: str | None
    transcript: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class VerifierHandler(Generic[PayloadT]):
    payload_type: type[PayloadT]
    grade: Callable[[PayloadT, GradingAttempt], GradeResult]


def _handler(kind: str) -> VerifierHandler[Any]:
    matches = [entry for entry in entry_points(group=ENTRY_POINT_GROUP) if entry.name == kind]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous verifier kind: {kind!r}")
    factory = cast(Callable[[], VerifierHandler[Any]], matches[0].load())
    return factory()


def _resolved(verifier: VerifierSpec) -> tuple[VerifierHandler[Any], Any]:
    handler = _handler(verifier.kind)
    try:
        payload = msgspec.convert(verifier.parameters, type=handler.payload_type, strict=True)
    except (msgspec.ValidationError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid {verifier.kind!r} verifier parameters: {error}") from error
    return handler, payload


def validate_verifier(verifier: VerifierSpec) -> None:
    """Check a private verifier before export or launch, independent of rendering."""
    _resolved(verifier)


def grade_attempt(specification: TaskSpec, attempt: GradingAttempt) -> GradeResult:
    """Dispatch an attempt by its private verifier kind."""
    handler, payload = _resolved(specification.verifier)
    return handler.grade(payload, attempt)


def grade_answer(specification: TaskSpec, rendering: Rendering, response: str | None) -> GradeResult:
    """Grade a direct answer through the same open verifier registry."""
    return grade_attempt(specification, GradingAttempt(rendering, response))


class ExactAnswerPayload(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    expected: str
    ignore_case: bool = True
    ignore_whitespace: bool = True

    def __post_init__(self) -> None:
        if not self.expected.strip():
            raise ValueError("An exact answer is required")


def exact_answer(expected: str, ignore_case: bool = True, ignore_whitespace: bool = True) -> VerifierSpec:
    """Construct a pinned exact-answer verifier descriptor."""
    payload = ExactAnswerPayload(expected, ignore_case, ignore_whitespace)
    return VerifierSpec(EXACT_ANSWER_KIND, msgspec.to_builtins(payload))


def _grade_exact(payload: ExactAnswerPayload, attempt: GradingAttempt) -> GradeResult:
    try:
        candidate = extract_answer(attempt.response, attempt.rendering)
    except (ValueError, TypeError) as error:
        return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
    with tempfile.TemporaryDirectory(prefix="taskcompendium-answer-") as temporary:
        output = Path(temporary) / "answer.txt"
        output.write_text(candidate)
        contract = ExactSpec(
            expected=(payload.expected,),
            ignore_case=payload.ignore_case,
            ignore_whitespace=payload.ignore_whitespace,
        )
        result = grade(dataclasses.replace(contract, output=str(output)), output.parent, output.parent)
    if result.status != Status.SCORED:
        return GradeResult(Outcome.INFRA_ERROR, None, str(result.detail))
    return GradeResult(Outcome.GRADED, result.reward)


def exact_answer_handler() -> VerifierHandler[ExactAnswerPayload]:
    return VerifierHandler(ExactAnswerPayload, _grade_exact)
