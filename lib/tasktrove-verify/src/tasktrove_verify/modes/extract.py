# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

r"""Pulling a final answer out of free-form model output.

The answer-file modes share three ways of narrowing an output file to one candidate string: the
last ``\boxed{}`` expression, the last non-empty line, and whitespace normalization. Nothing here
inspects a spec or decides a reward; each mode composes these as its own extraction rule.
"""

import re

BOXED = r"\boxed{"

_WHITESPACE = re.compile(r"\s+")
# Longest-first: ``$$`` must be tried before ``$``.
_MATH_DELIMITERS = ((r"\[", r"\]"), (r"\(", r"\)"), ("$$", "$$"), ("$", "$"))


def extract_boxed(text: str) -> str | None:
    """The brace-balanced content of the last ``\\boxed{...}``, or ``None`` when there is none.

    An unterminated ``\\boxed{`` reads as no boxed expression.
    """
    start = text.rfind(BOXED)
    if start < 0:
        return None
    body = start + len(BOXED)
    depth = 0
    for index in range(body, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            if depth == 0:
                return text[body:index].strip()
            depth -= 1
    return None


def last_line(text: str) -> str | None:
    """The last non-empty line, stripped, or ``None`` when the text is blank."""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def collapse_whitespace(text: str) -> str:
    """Every run of whitespace becomes a single space, and the result is stripped."""
    return _WHITESPACE.sub(" ", text).strip()


def strip_math_delimiters(text: str) -> str:
    """Remove the LaTeX math delimiters that wrap the whole expression, however many layers deep.

    ``$...$`` and ``$$...$$`` are only stripped when the text holds exactly that one pair, so an
    expression such as ``$a$ + $b$`` is left alone.
    """
    value = text.strip()
    changed = True
    while changed:
        changed = False
        for left, right in _MATH_DELIMITERS:
            if len(value) <= len(left) + len(right):
                continue
            if not (value.startswith(left) and value.endswith(right)):
                continue
            if left == "$$" and value.count("$") != 4:
                continue
            if left == "$" and value.count("$") != 2:
                continue
            value = value[len(left) : -len(right)].strip()
            changed = True
            break
    return value
