# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Private source rubric contracts awaiting an explicit semantic judge binding."""

from typing import Literal

from pydantic import JsonValue, model_validator

from taskcompendium.grading import GradeResult, Outcome
from taskcompendium.submission import extract_answer
from taskcompendium.verifiers.base import GradingAttempt, Verifier


class RubricJudgeVerifier(Verifier):
    """Preserve a reference-free source judge without inventing grading rewards."""

    mode: Literal["checklist", "holistic_numeric"]
    question: str
    criteria: tuple[str, ...]
    aggregation: dict[str, JsonValue]
    source_judge_data: dict[str, JsonValue]
    source_judge_toml: str

    @model_validator(mode="after")
    def validate_contract(self) -> "RubricJudgeVerifier":
        if not self.question.strip() or not self.criteria or any(not criterion.strip() for criterion in self.criteria):
            raise ValueError("A rubric judge requires its source question and nonempty criteria")
        return self

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            extract_answer(attempt.conversation[-1], attempt.convention)
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        return GradeResult(Outcome.INFRA_ERROR, None, "The source semantic rubric judge is not bound")
