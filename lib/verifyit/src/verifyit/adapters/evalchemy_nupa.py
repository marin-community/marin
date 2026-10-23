# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""NUPA representation-sensitive component and aligned-digit exact grading."""

import json
from collections.abc import Callable

from verifyit.adapters.skyrl import grade_literal_candidate
from verifyit.grade import Aggregation, Reward, Status, aggregate_rewards, invalid_task

ALIGNMENT = {
    "Integer": (True,),
    "Float": (True, False),
    "Fraction": (True, True),
    "ScientificNotation": (True, False, True),
}


def grade_nupa_answer(
    candidate: object,
    expected: object,
    answer_format: str,
    *,
    extract_answer: Callable[[str | None, str], str | None],
    full_answer: Callable[[str, str], str | None],
    digit_parts: Callable[[str, str], tuple[str, ...]],
) -> Reward:
    """Keep trusted source extraction; grade full components and each aligned digit.

    The callbacks prepare representations, never compute correctness. All five
    benchmark metrics are retained; reward is the source exact-match metric.
    Source digit preparation omits signs, including scientific exponent signs.
    These are digit-component metrics, not mathematical numeric equivalence.
    """
    if not isinstance(answer_format, str) or answer_format not in ALIGNMENT:
        return invalid_task("unsupported NUPA answer format")
    if not isinstance(expected, str) or not expected or full_answer(expected, answer_format) is None:
        return invalid_task("NUPA requires a valid nonempty reference representation")
    reference = digit_parts(expected, answer_format)
    if len(reference) != len(ALIGNMENT[answer_format]) or any(not part or not part.isdigit() for part in reference):
        return invalid_task("malformed NUPA reference components")
    if candidate is not None and not isinstance(candidate, str):
        candidate = None
    extracted = extract_answer(candidate, answer_format)
    prediction = digit_parts(extracted or "", answer_format)
    if len(prediction) != len(reference) or any(not isinstance(part, str) for part in prediction):
        raise RuntimeError("source NUPA extraction returned malformed components")
    format_valid = extracted is not None
    exact = grade_literal_candidate(json.dumps(reference), json.dumps(prediction) if format_valid else "")
    if exact.status is not Status.SCORED:
        return exact
    digits = []
    for predicted, gold, from_right in zip(prediction, reference, ALIGNMENT[answer_format], strict=True):
        if from_right:
            predicted, gold = predicted[::-1], gold[::-1]
        digits.extend(grade_literal_candidate(wanted, actual) for actual, wanted in zip(predicted, gold, strict=False))
    gold_length = sum(map(len, reference))
    digit_match = aggregate_rewards(digits, expected_total=gold_length, policy=Aggregation.MEAN)
    if digit_match.status is not Status.SCORED:
        return digit_match
    metrics = {
        "exact_match": exact.reward,
        "digit_match": digit_match.reward,
        "dlength": float(abs(sum(map(len, prediction)) - gold_length)),
        "format_valid": float(format_valid),
        "no_answer": float(not format_valid),
    }
    return Reward(exact.reward, exact.status, {**exact.detail, "metrics": metrics, "extracted": extracted})
