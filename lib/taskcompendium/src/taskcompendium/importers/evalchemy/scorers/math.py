# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evalchemy's final-box extraction and math-verify/Minerva comparison protocol.

Adopted from Marin's Evalchemy fork and lm-evaluation-harness v0.4.12 (MIT).
Keep source stops separate from the smaller set safe to send to a chat endpoint.
"""

import threading

from math_verify import parse, verify
from sympy.core.relational import Relational

from verifyit.adapters.completion import END_OF_TURN, final_text
from .amc import last_boxed_answer
from .minerva import is_equiv, normalize_final_answer
from verifyit.grade import Reward, invalid_task, scored

CHAT_REQUEST_STOPS = END_OF_TURN[:4]


def _symbol_free_relation(parsed: list) -> bool:
    values = [value for value in parsed if not isinstance(value, str)]
    return bool(values) and all(isinstance(value, Relational) and not value.free_symbols for value in values)


def final_boxed_answer(response: str) -> str:
    """Extract the source's last box in final content before a task boundary."""
    response = final_text(response)
    stops = [index for stop in END_OF_TURN if (index := response.find(stop)) >= 0]
    response = response[: min(stops)] if stops else response
    return last_boxed_answer(response)


def grade_boxed_math(expected: str, response: str) -> Reward:
    """Grade the complete final response under Evalchemy's boxed math protocol."""
    if not expected.strip():
        return invalid_task("Boxed math requires a nonempty reference")
    candidate = final_boxed_answer(response)
    if not candidate:
        return scored(0, extracted=candidate, expected=expected)
    timeout = 5 if threading.current_thread() is threading.main_thread() else 0
    reference = parse(r"\boxed{" + expected + "}", parsing_timeout=timeout)
    prediction = parse(r"\boxed{" + candidate + "}", parsing_timeout=timeout)
    if reference and prediction:
        relations = _symbol_free_relation(reference) and _symbol_free_relation(prediction)
        match = not relations and bool(verify(gold=reference, target=prediction, timeout_seconds=timeout))
    else:
        match = is_equiv(normalize_final_answer(candidate), normalize_final_answer(expected))
    return scored(float(match), extracted=candidate, expected=expected)
