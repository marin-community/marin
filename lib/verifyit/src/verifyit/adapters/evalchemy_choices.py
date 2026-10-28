# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPQA and MMLU-Pro extraction from Marin's Evalchemy fork.

GPQA's extraction was adapted there from QwenLM/Qwen2.5-Math (Apache-2.0).
These protocols deliberately differ: GPQA takes the last A-D anywhere, while
MMLU-Pro prioritizes the first answer marker before its case-sensitive fallback.
"""

import re

from verifyit.grade import Reward, scored

REASONING_END_MARKERS = ("</think>", "<|end_think|>")


def extract_gpqa_answer(response: str) -> str:
    matches = re.findall(r"\b(A|B|C|D)\b", response.upper())
    answer = matches[-1] if matches else response.strip().strip(".")
    return answer.rstrip(".").rstrip("/")


def extract_mmlu_answer(response: str) -> str | None:
    end = max(
        (index + len(marker) for marker in REASONING_END_MARKERS if (index := response.rfind(marker)) >= 0),
        default=0,
    )
    text = response[end:]
    for pattern in (r"answer is \(?([A-J])\)?", r"Answer:\s*([A-J])"):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return match.group(0) if match else None


def grade_gpqa_answer(expected: str, response: str) -> Reward:
    answer = extract_gpqa_answer(response)
    return scored(float(answer == expected), extracted=answer, expected=expected)


def grade_mmlu_answer(expected: str, response: str) -> Reward:
    answer = extract_mmlu_answer(response)
    return scored(float(answer == expected), extracted=answer, expected=expected)
