# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.test_time_scaling.config import ScoringMode
from marin.test_time_scaling.scorers import score_candidate_text


def test_math_boxed_scoring_extracts_normalized_answer():
    score = score_candidate_text(
        "Working carefully gives \\boxed{\\frac{1}{2}}",
        "\\boxed{\\frac{1}{2}}",
        ScoringMode.MATH_BOXED,
    )

    assert score.parse_valid is True
    assert score.extracted_answer == "1/2"
    assert score.is_correct is True


def test_math_boxed_scoring_marks_missing_boxed_answer_invalid():
    score = score_candidate_text(
        "The answer is 4.",
        "\\boxed{4}",
        ScoringMode.MATH_BOXED,
    )

    assert score.parse_valid is False
    assert score.extracted_answer is None
    assert score.is_correct is False
