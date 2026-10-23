"""Reusable item graders selected by sources through GraderSpec."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from dataclasses import replace
import re
from typing import Any

from eval.completion_response import CompletionClassification, CompletionText
from eval.contracts.eval_spec import EvalSpec, Grade, GraderSpec, JudgePlan
from eval.graders.answer_equivalence import math_answers_equivalent
from eval.graders.answer_extraction import extract_final_boxed_answer, final_response_text
from eval.graders.mmlu import extract_answer as extract_mmlu_answer, grade_response as grade_mmlu_response
from eval.chat_benchmarks.GPQADiamond.testing_utils import get_multiple_choice_answer
from eval.lm_eval_tasks.gsm8k.utils import extract_gsm8k_flexible_answer
from eval.graders.answer_extraction import AnswerExtractionError, extract_boxed_answer, extraction_failure

from eval.graders.judged import equivalence_judge

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

def gsm8k(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    answer = extract_gsm8k_flexible_answer(str(responses[-1]))
    return Grade({"accuracy": float(answer == str(spec.target))}, filtered_resps=[answer])

def letter_choice(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    extractor = get_multiple_choice_answer
    if _reference_option(spec, "extractor", "gpqa") == "hle":
        from eval.chat_benchmarks.HLE.testing_utils import get_multiple_choice_answer as extractor

    response = responses[-1]
    if isinstance(response, CompletionText):
        completion = response.response
        if completion.content:
            response = completion.content
        elif completion.classification == CompletionClassification.REASONING_ONLY:
            response = completion.reasoning_content or ""
        else:
            response = ""
    answer = extractor(final_response_text(str(response)))
    return Grade({"accuracy": float(answer == spec.target)}, filtered_resps=[answer])

def mmlu_choice(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    answer = extract_mmlu_answer(responses[-1])
    example = {"answer": spec.target, "model_answer": answer}
    grade_mmlu_response(example)
    return Grade(example["sample_metrics"], filtered_resps=[answer])

def matharena(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    # Optional grader code is imported only after the HMMT source has loaded it.
    from eval.chat_benchmarks.HMMT.matharena.parser import check_answers, extract_answer, parse_answer

    target, _ = parse_answer(str(spec.target))
    text = str(responses[-1])
    list_answer = "," in str(spec.target)
    answer, _ = extract_answer(text, False, True, list_answer)
    # Expanding a compact expression such as 1 - 1/2025! only for audit can
    # exceed Python's integer-string limit after native grading succeeds.
    audit_answer, _ = extract_answer(text, False, False, list_answer)
    return Grade({"accuracy": float(check_answers(answer, target))}, filtered_resps=[str(audit_answer)])

def answer_number(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    text = str(responses[-1]).lower()
    answer = None
    for pattern in (r"answer:.*?(\d+)", r"answer is.*?(\d+)", r"boxed{(\d+)}"):
        matches = re.findall(pattern, text)
        if matches:
            answer = matches[-1]
            break
    correct = _math_correct(answer, spec.target, _reference_option(spec, "comparator", "math_answers_equivalent"))
    return Grade({"accuracy": float(correct)}, filtered_resps=[answer])

def jee(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    from eval.chat_benchmarks.JEEBench.utils import compute_score

    answer = _boxed_answer(responses[-1], "hendrycks_math")
    return Grade({"accuracy": float(compute_score(spec.target, answer, spec.grader.reference["type"]))},
                 filtered_resps=[answer])

def nupa(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    scorer = import_module("eval.chat_benchmarks.NUPA-Loose.scorer")
    answer_format = spec.grader.reference["format"]
    score = scorer.score_prediction(str(responses[-1]), spec.target, answer_format)
    return Grade({
        "exact_match": score.exact_match, "digit_match": score.digit_match, "dlength": score.dlength,
        "format_valid_rate": score.format_valid, "no_answer_rate": score.no_answer,
    }, filtered_resps=[scorer.extract_answer(str(responses[-1]), answer_format)])

def math_equivalence(spec: EvalSpec, responses: Sequence[Any]) -> Grade | JudgePlan:
    try:
        answer = extract_boxed_answer(responses[-1])
    except AnswerExtractionError as error:
        return Grade({"accuracy": 0.0}, {"answer_extraction_error": extraction_failure(error)}, filtered_resps=[""])
    correct = bool(answer.strip()) and math_answers_equivalent(answer, spec.grader.reference["answers"])
    if correct or not spec.grader.reference["judge"]:
        return Grade({"accuracy": float(correct)}, {"method": "minerva"}, filtered_resps=[answer])
    return replace(equivalence_judge(spec, [answer]), on_error=GraderSpec("judge_failed", {"accuracy": 0.0}))
