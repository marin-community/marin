"""Final-answer grading for GenomeQA, retaining response and failure evidence."""

import re
from collections.abc import Sequence
from typing import Any

from eval.completion_response import CompletionContentPolicy, CompletionText
from eval.contracts.eval_spec import EvalSpec, Grade
from eval.graders.answer_extraction import final_response_text

_ANSWER_HEADING = "### Answer"
_ANSWER = re.compile(r"\bAnswer\s*:\s*\**\s*\[?([A-Za-z])\]?(?![A-Za-z])", re.IGNORECASE)
_LETTER = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")


def genomeqa(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    response = responses[-1]
    selected = str(response)
    failure = None
    if isinstance(response, CompletionText):
        completion = response.response
        selected = completion.normalized_content(CompletionContentPolicy.FINAL_ONLY)
        if completion.finish_reason == "length":
            failure = "truncation"
        elif completion.raw_choice.get("message", {}).get("refusal") or completion.finish_reason == "content_filter":
            failure = "refusal"
    # An unterminated explicit thinking block contains no usable final answer.
    if any(
        selected.rfind(start) > selected.rfind(end)
        for start, end in (
            ("<think>", "</think>"),
            ("<|begin_think|>", "<|end_think|>"),
        )
    ):
        selected = ""
    selected = final_response_text(selected)
    has_heading = _ANSWER_HEADING in selected
    if has_heading:
        selected = selected.rsplit(_ANSWER_HEADING, 1)[1]
    explicit = {letter.upper() for letter in _ANSWER.findall(selected)}
    candidates = explicit if explicit else set(_LETTER.findall(selected)) if has_heading else set()
    answer = ""
    if failure is None:
        if len(candidates) != 1:
            failure = "conflicting_answers" if len(candidates) > 1 else "missing_answer"
        else:
            candidate = candidates.pop()
            if candidate not in spec.grader.reference["options"]:
                failure = "invalid_option"
            else:
                answer = candidate
    correct = float(failure is None and answer == spec.target)
    return Grade(
        {
            "accuracy": correct,
            "genomeqa_macro": correct,
            "unanswered": float(not answer),
            "truncation": float(failure == "truncation"),
            "refusal": float(failure == "refusal"),
            "parse_failure": float(failure in {"missing_answer", "conflicting_answers", "invalid_option"}),
        },
        audit={"genomeqa_parsing": {"selected_span": selected, "answer": answer, "failure": failure}},
        weights={"genomeqa_macro": spec.grader.reference["cell_weight"]},
        filtered_resps=[answer],
    )
