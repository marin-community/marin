# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed cleanup math scoring and the TaskTrove abstention exact gate."""

import re
import string
from typing import Literal

from verifyit.modes.grade_judge import boxed_answer
from verifyit.spec import MathSpec, MathType

from taskcompendium.grading import GradeResult, GradingAttempt, Outcome, Verifier
from taskcompendium.submission import extract_answer
from taskcompendium.verifiers.reasoning import grade_answer
from taskcompendium.verifiers.reference_answers import ReferenceAnswersVerifier

ABSTENTIONS = frozenset({"idk", "i dont know", "unknown", "unanswerable", "cannot answer"})


def abstention_normalized(text: str) -> str:
    """Port the source gate without the cleanup judge's extra LaTeX/Unicode folding."""
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(?:a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class MathAnswerVerifier(Verifier):
    """Reuse the cleanup comparator without claiming original scorer parity."""

    expected: str
    math_type: Literal["scalar", "equation", "interval", "set", "tuple", "list"]

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            candidate = extract_answer(attempt.conversation[-1], attempt.convention)
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        return grade_answer(MathSpec(expected=self.expected, math_type=MathType(self.math_type)), candidate)


class AbstentionAnswersVerifier(Verifier):
    """Reject abstention before the explicitly unbound semantic fallback."""

    reference: ReferenceAnswersVerifier
    abstention_token: str | None

    def grade(self, attempt: GradingAttempt) -> GradeResult:
        try:
            candidate = extract_answer(attempt.conversation[-1], attempt.convention)
        except (ValueError, TypeError) as error:
            return GradeResult(Outcome.EXTRACTION_ERROR, None, str(error))
        normalized = abstention_normalized(boxed_answer(candidate[: 64 * 1024]))
        # The source checks exact agreement before its abstention gate.
        if normalized in {abstention_normalized(answer) for answer in self.reference.references}:
            return GradeResult(Outcome.GRADED, 1.0)
        if not normalized:
            return GradeResult(Outcome.GRADED, 0.0)
        if normalized in ABSTENTIONS and not self.abstention_token:
            return GradeResult(Outcome.GRADED, 0.0)
        return GradeResult(Outcome.INFRA_ERROR, None, "The source semantic abstention judge is not bound")
