# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPQA and MMLU-Pro extraction from Marin's Evalchemy fork.

GPQA's extraction was adapted there from QwenLM/Qwen2.5-Math (Apache-2.0).
These protocols deliberately differ: GPQA takes the last A-D anywhere, while
MMLU-Pro prioritizes the first answer marker before its case-sensitive fallback.
MedMCQA CoT reproduces lm-eval-harness's `medmcqa_cot` regex filter (MIT).
"""

import re

from verifyit.adapters.completion import final_text
from verifyit.grade import Reward, scored


def extract_gpqa_answer(response: str) -> str:
    matches = re.findall(r"\b(A|B|C|D)\b", response.upper())
    answer = matches[-1] if matches else response.strip().strip(".")
    return answer.rstrip(".").rstrip("/")


def extract_mmlu_answer(response: str) -> str | None:
    text = final_text(response)
    for pattern in (r"answer is \(?([A-J])\)?", r"Answer:\s*([A-J])"):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return match.group(0) if match else None


# lm-eval `medmcqa_cot`: the first match of this regex, stripped, is the answer.
MEDMCQA_COT_ANSWER = re.compile(r"Final Answer:\s*(?:\*\*)?\s*([A-D])")


def grade_medmcqa_cot(expected: str, response: str) -> Reward:
    match = MEDMCQA_COT_ANSWER.search(response)
    answer = match.group(1) if match else None
    return scored(float(answer == expected), extracted=answer, expected=expected)


def grade_gpqa_answer(expected: str, response: str) -> Reward:
    answer = extract_gpqa_answer(response)
    return scored(float(answer == expected), extracted=answer, expected=expected)


def grade_mmlu_answer(expected: str, response: str) -> Reward:
    answer = extract_mmlu_answer(response)
    return scored(float(answer == expected), extracted=answer, expected=expected)


def grade_bbh_cot(document: dict, response: str) -> Reward:
    matches = re.findall(r"(?<=the answer is )(.*)(?=.)", response)
    answer = matches[0] if matches else "[invalid]"
    value = float(answer == document["target"])
    return scored(value, metrics={"exact_match,get-answer": value}, extracted=answer)
