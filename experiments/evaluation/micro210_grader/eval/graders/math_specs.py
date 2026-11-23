# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable item graders selected by sources through GraderSpec."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from importlib import import_module
from typing import Any

from eval.chat_benchmarks.GPQADiamond.testing_utils import get_multiple_choice_answer
from eval.contracts.eval_spec import EvalSpec, Grade, GraderSpec, JudgePlan
from eval.graders.answer_equivalence import math_answers_equivalent
from eval.graders.answer_extraction import (
    AnswerExtractionError,
    extract_boxed_answer,
    extract_final_boxed_answer,
    extraction_failure,
)
from eval.graders.judged import equivalence_judge
from eval.graders.mmlu import extract_answer as extract_mmlu_answer
from eval.graders.mmlu import grade_response as grade_mmlu_response


def _reference_option(spec: EvalSpec, name: str, default: str) -> str:
    return str(spec.grader.reference.get(name, default))


def _boxed_answer(response: Any, extractor: str) -> str:
    if extractor == "final_boxed":
        return extract_final_boxed_answer(response)
    if extractor == "hendrycks_math":
        from lm_eval.tasks.hendrycks_math.utils import last_boxed_only_string, remove_boxed

        try:
            return remove_boxed(last_boxed_only_string(str(response)))
        except (AssertionError, TypeError, ValueError):
            return ""
    raise ValueError(f"unknown boxed-answer extractor: {extractor}")


def _math_correct(answer: str | None, target: Any, comparator: str) -> bool:
    if comparator == "math_answers_equivalent":
        return math_answers_equivalent(answer, [str(target)])
    if comparator == "hendrycks_math":
        from lm_eval.tasks.hendrycks_math.utils import is_equiv

        return is_equiv(str(target), answer)
    raise ValueError(f"unknown math comparator: {comparator}")


def boxed_math(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    answer = _boxed_answer(responses[-1], _reference_option(spec, "extractor", "final_boxed"))
    correct = _math_correct(answer, spec.target, _reference_option(spec, "comparator", "math_answers_equivalent"))
    return Grade({"accuracy": float(correct)}, filtered_resps=[answer])


def letter_choice(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    extractor = get_multiple_choice_answer
    if _reference_option(spec, "extractor", "gpqa") == "hle":
        from eval.chat_benchmarks.HLE.testing_utils import get_multiple_choice_answer as extractor

    answer = extractor(str(responses[-1]))
    return Grade({"accuracy": float(answer == spec.target)}, filtered_resps=[answer])


def mmlu_choice(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    answer = extract_mmlu_answer(responses[-1])
    example = {"answer": spec.target, "model_answer": answer}
    grade_mmlu_response(example)
    return Grade(example["sample_metrics"], filtered_resps=[answer])


def nupa(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    scorer = import_module("eval.chat_benchmarks.NUPA-Loose.scorer")
    answer_format = spec.grader.reference["format"]
    score = scorer.score_prediction(str(responses[-1]), spec.target, answer_format)
    return Grade(
        {
            "exact_match": score.exact_match,
            "digit_match": score.digit_match,
            "dlength": score.dlength,
            "format_valid_rate": score.format_valid,
            "no_answer_rate": score.no_answer,
        },
        filtered_resps=[score.extracted_answer],
    )


def math_equivalence(spec: EvalSpec, responses: Sequence[Any]) -> Grade | JudgePlan:
    try:
        answer = extract_boxed_answer(responses[-1])
    except AnswerExtractionError as error:
        return Grade({"accuracy": 0.0}, {"answer_extraction_error": extraction_failure(error)}, filtered_resps=[""])
    correct = bool(answer.strip()) and math_answers_equivalent(answer, spec.grader.reference["answers"])
    if correct or not spec.grader.reference["judge"]:
        return Grade({"accuracy": float(correct)}, {"method": "minerva"}, filtered_resps=[answer])
    return replace(equivalence_judge(spec, [answer]), on_error=GraderSpec("judge_failed", {"accuracy": 0.0}))
