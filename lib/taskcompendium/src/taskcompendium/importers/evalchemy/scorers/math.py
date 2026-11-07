# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evalchemy's final-box extraction and math-verify/Minerva comparison protocol.

Adopted from Marin's Evalchemy fork and lm-evaluation-harness v0.4.12 (MIT).
Keep source stops separate from the smaller set safe to send to a chat endpoint.
"""

import re
import threading

from math_verify import parse, verify
from sympy.core.relational import Relational
from verifyit.adapters.completion import END_OF_TURN, final_text
from verifyit.grade import Reward, invalid_task, scored

from .amc import last_boxed_answer
from .minerva import is_equiv, normalize_final_answer

CHAT_REQUEST_STOPS = END_OF_TURN[:4]


def normalize_numeric_answer(value: str) -> str:
    return re.sub(r",|\$|(?s:.*#### )|\.$", "", value).lower()


def grade_gsm8k_cot(document: dict, response: str) -> Reward:
    expected = document["answer"].rsplit("####", 1)[-1].strip()
    strict = re.findall(r"The answer is (\-?[0-9\.\,]+).", response)
    flexible = re.findall(r"(-?[$0-9.,]{2,})|(-?[0-9]+)", response)
    extracted = {
        "strict-match": strict[0] if strict else "[invalid]",
        "flexible-extract": next((value for value in flexible[-1] if value), "[invalid]") if flexible else "[invalid]",
    }
    metrics = {
        f"exact_match,{name}": float(normalize_numeric_answer(value) == normalize_numeric_answer(expected))
        for name, value in extracted.items()
    }
    return scored(metrics["exact_match,strict-match"], metrics=metrics, extracted=extracted)


def grade_minerva_completion(document: dict, response: str) -> Reward:
    matches = re.findall(
        r"Final Answer: The final answer is(.*?). I hope it is correct.", response + "I hope it is correct."
    )
    answer = normalize_final_answer(matches[0].strip() if matches else "[invalidanswer]")
    expected = normalize_final_answer(last_boxed_answer(document["solution"]))
    metrics = {
        "exact_match,none": float(is_equiv(answer, expected)),
        "math_verify,none": float(bool(verify(gold=parse(document["solution"]), target=parse(response)))),
    }
    return scored(metrics["exact_match,none"], metrics=metrics, extracted=answer)


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
