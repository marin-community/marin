# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

r"""Mode math: the candidate's final expression against ``expected``, compared by math-verify.

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

import threading
from pathlib import Path

from math_verify import parse, verify

from tasktrove_verify.modes.extract import extract_boxed, last_line, strip_math_delimiters
from tasktrove_verify.output import read_output
from tasktrove_verify.reward import InvalidTask, Reward, scored
from tasktrove_verify.spec import MathSpec, MathType

SET_TYPES = frozenset({MathType.SET, MathType.INTERVAL})

CLOSERS = {"(": ")", "[": "]", "{": "}"}
SIZE_COMMANDS = ("\\left", "\\right", "\\big", "\\Big", "\\bigg", "\\Bigg")
TIMEOUT = 5
"""Seconds math-verify may spend parsing or comparing one expression. Its timeout arms
``signal.alarm``, which only the main thread may do, so a worker thread runs without it."""


def _timeout() -> int | None:
    return TIMEOUT if threading.current_thread() is threading.main_thread() else None


def _parse(text: str) -> list:
    """math-verify's parse of ``text`` as a LaTeX expression, else of the text as written."""
    timeout = _timeout()
    return parse(f"${strip_math_delimiters(text)}$", parsing_timeout=timeout) or parse(text, parsing_timeout=timeout)


def _verify(expected: object, candidate: object, allow_set_relation_comp: bool = False) -> bool:
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


def grade(spec: MathSpec, tests_dir: Path, workspace: Path) -> Reward:
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
