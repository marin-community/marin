# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove open-QA exact gate with an explicitly unavailable semantic fallback."""

from pydantic import JsonValue, model_validator
from verifyit.modes.grade_judge import boxed_answer, normalize

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier
from taskcompendium.submission import extract_answer


class ReferenceAnswersVerifier(Verifier):
    """Keep exact successes without misgrading paraphrases as incorrect.

    A nonmatching response requires the source's semantic judge. Until that
    judge is bound, it returns an infrastructure error rather than a reward.
    """

    references: tuple[str, ...]
    question: str
    source_judge_data: dict[str, JsonValue]

    @model_validator(mode="after")
    def validate_references(self) -> "ReferenceAnswersVerifier":
        if not self.references or any(not reference.strip() for reference in self.references):
            raise ValueError("Open QA requires nonempty reference answers")
        if any(not normalize(reference) for reference in self.references):
            raise ValueError("A reference becomes empty under the source's normalization")
        return self

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            candidate = extract_answer(attempt.conversation[-1], attempt.convention)
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        if not candidate.strip():
            return GradeResult(Outcome.GRADED, 0.0)
        if normalize(boxed_answer(candidate)) in {normalize(reference) for reference in self.references}:
            return GradeResult(Outcome.GRADED, 1.0)
        return GradeResult(Outcome.INFRA_ERROR, None, "The source semantic reference judge is not bound")
