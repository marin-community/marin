# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove puzzle answer contracts over VerifyIt's exact and math modes."""

from typing import Literal

from verifyit.spec import ExactSpec, MathSpec, MathType, Spec

PuzzleAnswerType = Literal["choice", "exact", "ordered_list", "number", "coords"]


def puzzle_spec(expected: str, answer_type: PuzzleAnswerType) -> Spec:
    """Select the existing comparator while preserving list order and symbolic coordinates."""
    if answer_type in {"number", "coords"}:
        return MathSpec(expected=expected, math_type=MathType.SCALAR)
    answers = (
        tuple(item.strip() for item in expected.split(",") if item.strip())
        if answer_type == "ordered_list"
        else (expected,)
    )
    return ExactSpec(expected=answers)
