# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import re

from marin.rl.environments.tinker_environments.math_grading import extract_boxed, grade_answer, normalize_answer
from marin.rl.math_utils import last_boxed_only_string
from marin.test_time_scaling.config import ScoringMode


@dataclass(frozen=True)
class CandidateScore:
    """Structured score fields added to each generated candidate."""

    extracted_answer: str | None
    parse_valid: bool | None
    is_correct: bool | None


_SIMPLE_FRAC_PATTERN = re.compile(r"^\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}$")


def _normalize_extracted_answer(answer: str) -> str:
    normalized = normalize_answer(answer)
    if normalized is None:
        return answer

    match = _SIMPLE_FRAC_PATTERN.fullmatch(normalized)
    if match is None:
        return normalized

    numerator, denominator = match.groups()
    return f"{numerator}/{denominator}"


def score_candidate_text(text: str, expected_answer: str | None, scoring_mode: ScoringMode) -> CandidateScore:
    """Score a generated candidate against the prompt's configured scoring mode."""

    if scoring_mode == ScoringMode.UNSCORED:
        return CandidateScore(extracted_answer=None, parse_valid=None, is_correct=None)

    if scoring_mode != ScoringMode.MATH_BOXED:
        raise ValueError(f"Unsupported scoring mode: {scoring_mode}")

    if "\\boxed" not in text:
        return CandidateScore(extracted_answer=None, parse_valid=False, is_correct=False if expected_answer else None)

    boxed = last_boxed_only_string(text)
    if boxed is None:
        return CandidateScore(extracted_answer=None, parse_valid=False, is_correct=False if expected_answer else None)

    extracted_answer_raw = extract_boxed(boxed)
    extracted_answer = _normalize_extracted_answer(extracted_answer_raw)
    is_correct = grade_answer(extracted_answer_raw, expected_answer) if expected_answer is not None else None
    return CandidateScore(extracted_answer=extracted_answer, parse_valid=True, is_correct=is_correct)
