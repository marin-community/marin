"""Canonical MMLU-Pro answer extraction and per-example scoring."""

import re
from typing import Any, Dict, Optional

from eval.completion_response import CompletionClassification, CompletionText
from eval.contracts.sample_results import record_sample_metrics
from eval.graders.answer_extraction import final_response_text


def extract_answer(text: str) -> Optional[str]:
    if isinstance(text, CompletionText):
        completion = text.response
        if completion.content:
            text = completion.content
        elif completion.classification == CompletionClassification.REASONING_ONLY:
            text = completion.reasoning_content or ""
        else:
            return None
    text = final_response_text(text)
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    else:
        return extract_again(text)


def extract_again(text: str) -> Optional[str]:
    match = re.search(r"Answer:\s*([A-J])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    else:
        return extract_final(text)


def extract_final(text: str) -> Optional[str]:
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(0) if match else None


def grade_response(example: Dict[str, Any]) -> None:
    record_sample_metrics(example, accuracy=example["model_answer"] == example["answer"])
