# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repairs for Reasoning Gym 0.1.25 scorers that pay incorrect fixed replies."""

import ast
import math
import re
from collections import Counter
from collections.abc import Callable
from fractions import Fraction

from verifyit.grade import InvalidTask

MAX_EXPRESSION_LENGTH = 256
# A coin_flip reply within this fraction of the reference probability is correct: the question asks for no precision,
# and a reply rounded to four significant figures is within it.
COIN_FLIP_RELATIVE_DENOMINATOR = 1_000


def _puzzle24_value(node: ast.expr) -> tuple[Fraction, list[int]]:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return Fraction(node.value), [node.value]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        value, numbers = _puzzle24_value(node.operand)
        return (-value if isinstance(node.op, ast.USub) else value), numbers
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add | ast.Sub | ast.Mult | ast.Div):
        left, left_numbers = _puzzle24_value(node.left)
        right, right_numbers = _puzzle24_value(node.right)
        if isinstance(node.op, ast.Add):
            value = left + right
        elif isinstance(node.op, ast.Sub):
            value = left - right
        elif isinstance(node.op, ast.Mult):
            value = left * right
        else:
            value = left / right
        return value, left_numbers + right_numbers
    raise ValueError("puzzle24 answers must use integer literals and arithmetic operators")


def _score_puzzle24(answer: str, entry: dict) -> float:
    given_numbers = entry["metadata"].get("numbers")
    if (
        not isinstance(given_numbers, list)
        or len(given_numbers) != 4
        or any(type(number) is not int for number in given_numbers)
    ):
        raise InvalidTask("puzzle24 entry requires four integer numbers")
    if len(answer) > MAX_EXPRESSION_LENGTH:
        return 0.0
    try:
        value, used_numbers = _puzzle24_value(ast.parse(answer, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
        return 0.0
    return float(Counter(used_numbers) == Counter(given_numbers) and value == 24)


def _score_coin_flip(answer: str, entry: dict) -> float:
    try:
        expected = Fraction(entry["answer"].replace(",", ""))
    except (AttributeError, ValueError, ZeroDivisionError) as error:
        raise InvalidTask("coin_flip reference probability is malformed") from error
    if len(answer) > MAX_EXPRESSION_LENGTH:
        return 0.0
    try:
        candidate = Fraction(answer.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return 0.0
    if not 0 <= candidate <= 1:
        return 0.0
    return float(abs(candidate - expected) <= abs(expected) / COIN_FLIP_RELATIVE_DENOMINATOR)


def _formula_text(formula: str) -> str:
    """Ignore spaces and redundant outer parentheses, preserving operator and operand order."""
    formula = re.sub(r"\s+", "", formula)
    while formula.startswith("(") and formula.endswith(")"):
        depth = 0
        for position, character in enumerate(formula):
            depth += (character == "(") - (character == ")")
            if depth == 0 and position != len(formula) - 1:
                return formula
        if depth != 0:
            return formula
        formula = formula[1:-1]
    return formula


def score_reasoning_gym_reply(dataset: str, answer: str, entry: dict, upstream: Callable[[str, dict], float]) -> float:
    """Score with targeted fixes while retaining unaffected upstream scorers."""
    if dataset == "complex_arithmetic":
        # The pinned scorer's min(1, exp(-distance)) turns NaN distance into full reward.
        normalized = answer.replace(" ", "").lower().replace("i", "j").replace("+j", "+1j").replace("-j", "-1j")
        try:
            parsed = complex(normalized)
        except ValueError:
            return 0.0
        if not math.isfinite(parsed.real) or not math.isfinite(parsed.imag):
            return 0.0
        score = upstream(answer, entry)
        return score if math.isfinite(score) else 0.0
    if dataset == "coin_flip":
        return _score_coin_flip(answer, entry)
    if dataset == "game_of_life_halting":
        reference = entry["answer"]
        if not isinstance(reference, str) or reference.strip().casefold() not in {"true", "false"}:
            raise InvalidTask("game_of_life_halting reference answer must be True or False")
        expected = reference.strip().casefold()
        candidate = answer.strip().casefold()
        return float(candidate in {"true", "false"} and candidate == expected)
    if dataset == "propositional_logic":
        expected = entry["metadata"].get("example_answer")
        if not isinstance(expected, str) or not expected.strip():
            raise InvalidTask("propositional_logic entry requires an example answer")
        if len(answer) > MAX_EXPRESSION_LENGTH:
            return 0.0
        return float(bool(answer.strip()) and _formula_text(answer) == _formula_text(expected))
    if dataset == "puzzle24":
        return _score_puzzle24(answer, entry)
    return upstream(answer, entry)
