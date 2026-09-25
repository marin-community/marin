# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a decoded answer without a filesystem-backed verifier."""

from dataclasses import dataclass
from enum import StrEnum

from taskcompendium.models import ExactAnswer, TaskSpec
from taskcompendium.rendering import Rendering, extract_answer


class Outcome(StrEnum):
    GRADED = "graded"
    EXTRACTION_ERROR = "extraction_error"
    INFRA_ERROR = "infra_error"


@dataclass(frozen=True)
class GradeResult:
    status: Outcome
    reward: float | None
    error: str | None = None


def _normalize_answer(value: str, contract: ExactAnswer) -> str:
    normalized = " ".join(value.split()) if contract.ignore_whitespace else value.strip()
    return normalized.casefold() if contract.ignore_case else normalized


def grade_answer(specification: TaskSpec, rendering: Rendering, response: str | None) -> GradeResult:
    """Extract and score a response while distinguishing invalid submissions from verifier failures."""
    try:
        candidate = extract_answer(response, rendering)
    except (ValueError, TypeError) as error:
        return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
    matches = _normalize_answer(candidate, specification.verifier) == _normalize_answer(
        specification.verifier.expected, specification.verifier
    )
    return GradeResult(Outcome.GRADED, float(matches))
