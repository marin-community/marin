# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LongBench v2 answer extraction, matching upstream ``pred.py``.

https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/pred.py
"""

import re

from verifyit.grade import InvalidTask, Reward, scored

LONGBENCH_V2_CHOICES = ("A", "B", "C", "D")
_PARENTHESIZED = re.compile(r"The correct answer is \(([A-D])\)")
_BARE = re.compile(r"The correct answer is ([A-D])")
_REASONING_END = "</think>"


def extract_answer(response: str) -> str | None:
    """Return the stated choice letter as upstream does, ignoring Markdown emphasis.

    The first parenthesized ``(X)`` anywhere wins over an earlier bare ``is X``.
    """
    response = response.replace("*", "")
    match = _PARENTHESIZED.search(response) or _BARE.search(response)
    return match.group(1) if match else None


def grade_longbench_v2(expected: str, response: str) -> Reward:
    """Score exact choice-letter accuracy after the last inline ``</think>`` tag."""
    if expected not in LONGBENCH_V2_CHOICES:
        raise InvalidTask(f"LongBench v2 requires a gold letter in A-D, got {expected!r}")
    answer = extract_answer(response.rsplit(_REASONING_END, 1)[-1])
    return scored(
        float(answer == expected),
        metrics={"accuracy": float(answer == expected)},
        model_answer=answer,
        answer_extraction_errors=[] if answer is not None else ["failed_to_parse"],
    )
