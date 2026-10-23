"""Strict tagged-answer grading for the Marin LAB-Bench variant."""

import re
from collections.abc import Sequence
from typing import Any

from eval.completion_response import CompletionText
from eval.contracts.eval_spec import EvalSpec, Grade
from eval.graders.answer_extraction import final_response_text

ANSWER = re.compile(r"\[ANSWER\]\s*([A-Z])\s*\[/ANSWER\]")


def labbench(spec: EvalSpec, responses: Sequence[Any]) -> Grade:
    """Count malformed responses as attempted and incorrect, separately from abstention."""
    response = responses[-1]
    text = (response.response.content or "") if isinstance(response, CompletionText) else str(response)
    text = final_response_text(text)
    matches = ANSWER.findall(text)
    valid = (
        len(matches) == 1
        and text.count("[ANSWER]") == text.count("[/ANSWER]") == 1
        and matches[0] in spec.grader.reference["letters"]
    )
    answer = matches[0] if valid else ""
    abstained = valid and answer == spec.grader.reference["abstention"]
    correct = valid and answer == spec.target
    attempted = not abstained
    return Grade(
        metrics={
            "accuracy": float(correct),
            "precision": float(correct),
            "attempted_answer_coverage": float(attempted),
            "abstention_rate": float(abstained),
            "parse_failure_rate": float(not valid),
        },
        weights={"precision": float(attempted)},
        audit={"model_answer": answer, "abstained": abstained, "parse_failure": not valid},
        filtered_resps=[answer],
    )
