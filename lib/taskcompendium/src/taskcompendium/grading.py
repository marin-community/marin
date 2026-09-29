# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade submissions with typed private verifiers."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, field_validator

from taskcompendium.models import VerifierKind, VerifierSpec
from taskcompendium.submission import SubmissionConvention, extract_answer

WHITESPACE = re.compile(r"\s+")


class Outcome(StrEnum):
    GRADED = "graded"
    EXTRACTION_ERROR = "extraction_error"
    INVALID_TASK = "invalid_task"
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
    response: str | None
    environment: object


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
            candidate = extract_answer(attempt.response, attempt.convention)
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        match = _normalize_exact(candidate, self) == _normalize_exact(self.expected, self)
        return GradeResult(Outcome.GRADED, float(match))


def exact_answer(expected: str, ignore_case: bool = True, collapse_whitespace: bool = True) -> VerifierSpec:
    """Construct a pinned exact-answer verifier descriptor."""
    verifier = ExactAnswerVerifier(expected=expected, ignore_case=ignore_case, collapse_whitespace=collapse_whitespace)
    return VerifierSpec(kind=VerifierKind.EXACT_ANSWER, parameters_json=verifier.model_dump_json())


def _normalize_exact(value: str, verifier: ExactAnswerVerifier) -> str:
    value = WHITESPACE.sub(" ", value).strip() if verifier.collapse_whitespace else value.strip()
    return value.casefold() if verifier.ignore_case else value
