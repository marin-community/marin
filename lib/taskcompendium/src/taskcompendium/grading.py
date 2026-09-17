# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a decoded answer with the pinned TaskTrove exact-answer contract."""

import dataclasses
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tasktrove_verify.grade import Status, grade
from tasktrove_verify.spec import ExactSpec

from taskcompendium.models import TaskSpec
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


def grade_answer(specification: TaskSpec, rendering: Rendering, response: str | None) -> GradeResult:
    """Keep submission errors separate from failed correctness checks."""
    try:
        candidate = extract_answer(response, rendering)
    except (ValueError, TypeError) as error:
        return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
    with tempfile.TemporaryDirectory(prefix="taskcompendium-answer-") as temporary:
        output = Path(temporary) / "answer.txt"
        output.write_text(candidate)
        contract = ExactSpec(
            expected=(specification.verifier.expected,),
            ignore_case=specification.verifier.ignore_case,
            ignore_whitespace=specification.verifier.ignore_whitespace,
        )
        result = grade(dataclasses.replace(contract, output=str(output)), output.parent, output.parent)
    if result.status != Status.SCORED:
        return GradeResult(Outcome.INFRA_ERROR, None, str(result.detail))
    return GradeResult(Outcome.GRADED, result.reward)
