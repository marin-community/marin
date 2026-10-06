# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact scalar literals and final-answer extraction for numeric grading."""

import re
from fractions import Fraction

from verifyit.modes.extract import BOXED, extract_boxed, last_line, strip_math_delimiters

MAX_NUMERIC_DIGITS = 4096
MAX_LITERAL_LENGTH = 2 * MAX_NUMERIC_DIGITS + 32
INTEGER = r"[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
DECIMAL = rf"(?:{INTEGER}(?:\.[0-9]*)?|[+-]?\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
LITERAL = re.compile(rf"(?:{INTEGER}\s*/\s*{INTEGER}|{DECIMAL})")
# Boundaries keep malformed numbers and numbers embedded in words from yielding partial literals.
NUMERIC_TOKEN = re.compile(rf"(?<![\w.,/]){LITERAL.pattern}(?![\w/]|[.,][\w.,])")


class NumericCandidateError(ValueError):
    """The submitted final answer does not contain one supported numeric literal."""


def numeric_literal(text: str) -> Fraction:
    """Parse an integer, decimal, scientific notation or integer fraction exactly.

    Thousands separators require groups of three digits. Literal components and
    expanded decimal powers are bounded before Fraction allocates large integers.
    """
    if not isinstance(text, str):
        raise ValueError("numeric values require literal strings")
    value = text.strip()
    if len(value) > MAX_LITERAL_LENGTH or LITERAL.fullmatch(value) is None:
        raise ValueError("numeric value requires one finite scalar literal")
    value = value.replace(",", "")
    if "/" in value:
        numerator, denominator = (part.strip() for part in value.split("/"))
        if any(len(part.lstrip("+-")) > MAX_NUMERIC_DIGITS for part in (numerator, denominator)):
            raise ValueError("numeric literal exceeds the digit limit")
        divisor = int(denominator)
        if divisor == 0:
            raise ValueError("numeric fraction denominator is zero")
        return Fraction(int(numerator), divisor)
    parts = re.split("[eE]", value)
    mantissa = parts[0]
    exponent_text = parts[1] if len(parts) == 2 else "0"
    if len(exponent_text.lstrip("+-")) > len(str(MAX_NUMERIC_DIGITS)):
        raise ValueError("numeric exponent exceeds the digit limit")
    exponent = int(exponent_text)
    digits = sum(character.isdigit() for character in mantissa)
    if digits + abs(exponent) > MAX_NUMERIC_DIGITS:
        raise ValueError("numeric literal exceeds the digit limit")
    return Fraction(value)


def extract_numeric_candidate(text: str) -> Fraction:
    """Read the last box or the sole numeric literal on the final nonempty line."""
    if BOXED in text:
        candidate = extract_boxed(text)
        if not candidate:
            raise NumericCandidateError("final numeric box is empty or malformed")
    else:
        matches = list(NUMERIC_TOKEN.finditer(last_line(text) or ""))
        if len(matches) != 1:
            raise NumericCandidateError("final numeric line requires exactly one literal")
        candidate = matches[0].group()
    candidate = strip_math_delimiters(candidate)
    try:
        return numeric_literal(candidate)
    except ValueError as error:
        raise NumericCandidateError(str(error)) from error
