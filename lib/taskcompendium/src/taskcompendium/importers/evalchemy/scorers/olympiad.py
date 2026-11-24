# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OlympiadBench grading: the final answer's last box, as for AIME, against any flattened reference.

Reference flattening and equivalence are adopted from Marin Evalchemy (MIT), which extracted the first box instead.
"""

import re
import threading
from collections.abc import Sequence

from math_verify import parse, verify
from verifyit.grade import Reward, invalid_task, scored

from .math import final_boxed_answer
from .minerva import is_equiv, normalize_final_answer
from .relations import same_relations, symbol_free_relation


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


def grade_olympiad(expected: tuple[str, ...], response: str) -> Reward:
    """Score the final answer's last box against any reference under the source equivalence policy."""
    if not expected:
        return invalid_task("OlympiadBench requires reference answers")
    candidate = final_boxed_answer(response)
    correct = bool(candidate.strip()) and answers_equivalent(candidate, expected)
    return scored(float(correct), extracted=candidate, method="minerva")


def answers_equivalent(candidate: str, expected: tuple[str, ...]) -> bool:
    """Compare source-extracted content without applying extraction a second time."""
    timeout = 5 if threading.current_thread() is threading.main_thread() else 0
    prediction = parse(r"\boxed{" + candidate + "}", parsing_timeout=timeout)
    for answer in expected:
        reference = parse(r"\boxed{" + answer + "}", parsing_timeout=timeout)
        if reference and prediction:
            if symbol_free_relation(reference):
                if same_relations(prediction, reference):
                    return True
            elif verify(gold=reference, target=prediction, timeout_seconds=timeout):
                return True
        elif is_equiv(normalize_final_answer(candidate), normalize_final_answer(answer)):
            return True
    return False
