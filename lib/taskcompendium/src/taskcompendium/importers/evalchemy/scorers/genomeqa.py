# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GenomeQA final-answer parsing and equal-cell scoring."""

import re

from verifyit.adapters.completion import final_text
from verifyit.grade import Reward, scored

MACRO_METRIC = "genomeqa_macro"

_ANSWER_HEADING = "### Answer"
_CHOICE = r"[A-Za-z](?!\w)"
_ANSWER = re.compile(
    rf"\bAnswer\s*:\s*\**\s*[\[(]?({_CHOICE}(?:[\])]?\**\s*(?:or|and|/|,)\s*\**[\[(]?{_CHOICE})*)",
    re.IGNORECASE,
)
_LETTER = re.compile(r"(?<!\w)([A-Z])(?!\w)")


def grade_genomeqa(expected: str, options: str, cell_weight: float, response: str, stop_reason: str) -> Reward:
    selected = response
    failure = "refusal" if stop_reason == "content_filter" else None
    # An unterminated explicit thinking block contains no usable final answer.
    if any(
        selected.rfind(start) > selected.rfind(end)
        for start, end in (
            ("<think>", "</think>"),
            ("<|begin_think|>", "<|end_think|>"),
        )
    ):
        selected = ""
    selected = final_text(selected)
    has_heading = _ANSWER_HEADING in selected
    if has_heading:
        selected = selected.rsplit(_ANSWER_HEADING, 1)[1]
    explicit = {letter for declaration in _ANSWER.findall(selected) for letter in _LETTER.findall(declaration.upper())}
    candidates = explicit if explicit else set(_LETTER.findall(selected)) if has_heading else set()
    answer = ""
    if failure is None:
        if len(candidates) != 1:
            failure = "conflicting_answers" if len(candidates) > 1 else "missing_answer"
        else:
            candidate = candidates.pop()
            if candidate not in options:
                failure = "invalid_option"
            else:
                answer = candidate
    if not answer and stop_reason == "length":
        failure = "truncation"
    correct = float(failure is None and answer == expected)
    return scored(
        correct,
        metrics={
            "accuracy": correct,
            MACRO_METRIC: correct,
            "unanswered": float(not answer),
            "truncation": float(stop_reason == "length"),
            "refusal": float(failure == "refusal"),
            "parse_failure": float(failure in {"missing_answer", "conflicting_answers", "invalid_option"}),
        },
        metric_weights={MACRO_METRIC: cell_weight},
        genomeqa_parsing={"selected_span": selected, "answer": answer, "failure": failure},
    )
