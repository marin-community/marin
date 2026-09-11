# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

r"""Grade mathematical answers, either symbolically or as a number with tolerance.

The candidate is the last ``\boxed{}`` expression, or the last non-empty line when the output has
none. Both sides are parsed as anchored LaTeX (``$...$``) so math-verify reads a whole expression
instead of the first bare number it finds, falling back to parsing the raw text with math-verify's
own anchors (``the answer is ...``) when the anchored parse yields nothing.

``math_type`` selects the comparison. Set and interval answers allow math-verify's set/relation
comparison, so an expected ``(2, \infty)`` accepts a candidate ``x > 2``. A list is an ordered
comma-separated sequence compared member by member, which accepts the brackets a model does or
does not write around it and keeps a reordered answer wrong. Everything else is one expression.

Expected text that math-verify cannot turn into an expression is a defective task, not a wrong
answer.
"""

import math
import re
import threading
from pathlib import Path

from tasktrove_verify.modes.extract import extract_boxed, last_line, strip_math_delimiters
from tasktrove_verify.grade import InvalidTask, Reward, read_output, scored
from tasktrove_verify.spec import MathSpec, MathType, NumericSpec

SET_TYPES = frozenset({MathType.SET, MathType.INTERVAL})

CLOSERS = {"(": ")", "[": "]", "{": "}"}
SIZE_COMMANDS = ("\\left", "\\right", "\\big", "\\Big", "\\bigg", "\\Bigg")
TIMEOUT = 5
"""Seconds math-verify may spend parsing or comparing one expression. Its timeout arms
``signal.alarm``, which only the main thread may do, so a worker thread runs without it."""
NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][-+]?\d+)?|[-+]?\.\d+(?:[eE][-+]?\d+)?")


def _timeout() -> int | None:
    return TIMEOUT if threading.current_thread() is threading.main_thread() else None


def _parse(text: str) -> list:
    """math-verify's parse of ``text`` as a LaTeX expression, else of the text as written."""
    from math_verify import parse

    timeout = _timeout()
    return parse(f"${strip_math_delimiters(text)}$", parsing_timeout=timeout) or parse(text, parsing_timeout=timeout)


def _verify(expected: object, candidate: object, allow_set_relation_comp: bool = False) -> bool:
    from math_verify import verify

    return verify(expected, candidate, allow_set_relation_comp=allow_set_relation_comp, timeout_seconds=_timeout())


def _is_expression(parsed: list) -> bool:
    """Whether math-verify recovered an expression rather than only the original string."""
    return any(not isinstance(item, str) for item in parsed)


def _split_members(text: str) -> list[str]:
    """The comma-separated members of a sequence, ignoring commas nested inside brackets.

    One optional layer of enclosing brackets is dropped, so ``[1, 2]`` and ``1, 2`` read alike.
    """
    value = strip_math_delimiters(text)
    for command in SIZE_COMMANDS:
        value = value.replace(command, "")
    value = value.strip()
    if len(value) > 1 and value[0] in CLOSERS and value[-1] == CLOSERS[value[0]]:
        value = value[1:-1]
    members: list[str] = []
    stack: list[str] = []
    start = 0
    for index, char in enumerate(value):
        if char in CLOSERS:
            stack.append(CLOSERS[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char == "," and not stack:
            members.append(value[start:index].strip())
            start = index + 1
    members.append(value[start:].strip())
    return members


def _parsed_members(text: str) -> list[list]:
    return [_parse(member) for member in _split_members(text)]


def _members_match(expected: list[list], candidate: list[list]) -> bool:
    if len(expected) != len(candidate):
        return False
    return all(
        bool(parsed_candidate) and _verify(parsed_expected, parsed_candidate)
        for parsed_expected, parsed_candidate in zip(expected, candidate, strict=True)
    )


def _grade_symbolic(spec: MathSpec, workspace: Path) -> Reward:
    is_list = spec.math_type is MathType.LIST
    expected = _parsed_members(spec.expected) if is_list else [_parse(spec.expected)]
    if not all(_is_expression(member) for member in expected):
        raise InvalidTask(f"math-verify cannot parse expected {spec.expected!r}")

    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    candidate = extract_boxed(text) or last_line(text) or ""
    parsed = _parsed_members(candidate) if is_list else [_parse(candidate)]
    if not any(parsed):
        return scored(0.0, reason="unparsable", extracted=candidate, expected=spec.expected)

    if is_list:
        match = _members_match(expected, parsed)
    else:
        match = _verify(expected[0], parsed[0], allow_set_relation_comp=spec.math_type in SET_TYPES)
    return scored(float(bool(match)), extracted=candidate, expected=spec.expected)


def _last_number(text: str) -> float | None:
    boxed = extract_boxed(text)
    sources = [boxed, text] if boxed else [text]
    for source in sources:
        matches = NUMBER.findall(source)
        if matches:
            return float(matches[-1].replace(",", ""))
    return None


def _grade_numeric(spec: NumericSpec, workspace: Path) -> Reward:
    if not math.isfinite(spec.expected):
        raise InvalidTask(f"numeric expected must be a finite number, got {spec.expected}")
    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    value = _last_number(text)
    if value is None:
        return scored(0.0, reason="no_number", expected=spec.expected)
    tolerance = max(spec.tolerance_abs, spec.tolerance_rel * abs(spec.expected))
    match = abs(value - spec.expected) <= tolerance
    return scored(float(match), extracted=value, expected=spec.expected, tolerance=tolerance)


def grade(spec: MathSpec | NumericSpec, tests_dir: Path, workspace: Path) -> Reward:
    if isinstance(spec, NumericSpec):
        return _grade_numeric(spec, workspace)
    return _grade_symbolic(spec, workspace)
