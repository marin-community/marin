# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Legacy OlympiadBench answer preparation, adopted from Marin Evalchemy (MIT).

The legacy protocol extracts the first box, including boxes in reasoning, and
accepts any flattened reference. These rules differ from AIME's final-box path.
"""

import re
import threading
from collections.abc import Sequence

from math_verify import parse, verify
from sympy.core.relational import Relational
from verifyit.adapters.completion import END_OF_TURN
from verifyit.grade import Reward, invalid_task, scored

from .minerva import is_equiv, normalize_final_answer


def reference_answers(raw: str | int | float | Sequence[str]) -> tuple[str, ...]:
    """Flatten dollar-delimited alternatives without inventing joint-answer grading."""
    values = (str(raw),) if isinstance(raw, str | int | float) else raw
    answers = []
    for value in values:
        value = str(value)
        if "$" in value:
            groups = re.findall(r"\$([^$]*)\$", value)
            if groups:
                answers.extend(group.strip() for group in groups if group.strip())
                continue
        cleaned = value.strip()
        if len(cleaned) >= 2 and cleaned.startswith("$") and cleaned.endswith("$"):
            cleaned = cleaned[1:-1].strip()
            if len(cleaned) >= 2 and cleaned.startswith("$") and cleaned.endswith("$"):
                cleaned = cleaned[1:-1].strip()
        if cleaned:
            answers.append(cleaned)
    return tuple(answers)


def first_boxed_answer(response: str) -> str:
    """Extract the first source box before a task boundary, without removing reasoning."""
    ends = [index for stop in END_OF_TURN if (index := response.find(stop)) >= 0]
    response = response[: min(ends)] if ends else response
    starts = [index for marker in (r"\boxed", r"\fbox") if (index := response.find(marker)) >= 0]
    if not starts:
        return ""
    start = min(starts)
    later = [index for marker in (r"\boxed", r"\fbox") if (index := response.find(marker, start + 1)) >= 0]
    response = response[start : min(later)] if later else response[start:]
    if r"\boxed " in response:
        return response.split(r"\boxed ")[-1].split("$")[0]
    if not response.startswith(r"\boxed{"):
        return ""
    depth = 0
    for index, char in enumerate(response):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return response[len(r"\boxed{") : index]
    return ""


def grade_olympiad(expected: tuple[str, ...], response: str) -> Reward:
    """Score the first box against any reference under the source equivalence policy."""
    if not expected:
        return invalid_task("OlympiadBench requires reference answers")
    candidate = first_boxed_answer(response)
    correct = bool(candidate.strip()) and answers_equivalent(candidate, expected)
    return scored(float(correct), extracted=candidate, method="minerva")


def answers_equivalent(candidate: str, expected: tuple[str, ...]) -> bool:
    """Compare source-extracted content without applying extraction a second time."""
    timeout = 5 if threading.current_thread() is threading.main_thread() else 0
    prediction = parse(r"\boxed{" + candidate + "}", parsing_timeout=timeout)
    for answer in expected:
        reference = parse(r"\boxed{" + answer + "}", parsing_timeout=timeout)
        if reference and prediction:
            values = [[value for value in parsed if not isinstance(value, str)] for parsed in (reference, prediction)]
            relations = all(
                items and all(isinstance(value, Relational) and not value.free_symbols for value in items)
                for items in values
            )
            if not relations and verify(gold=reference, target=prediction, timeout_seconds=timeout):
                return True
        elif is_equiv(normalize_final_answer(candidate), normalize_final_answer(answer)):
            return True
    return False
