# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

r"""Grade mathematical answers, either symbolically or as a number with tolerance.

The default anchored profile takes the last ``\boxed{}`` expression, or the last non-empty line when the output has
none. Both sides are parsed as anchored LaTeX (``$...$``) so math-verify reads a whole expression
instead of the first bare number it finds, falling back to parsing the raw text with math-verify's
own anchors (``the answer is ...``) when the anchored parse yields nothing. Other profiles retain
raw source extraction or require a boxed expression; their parser options remain task-controlled.

``math_type`` selects the comparison. Set and interval answers allow math-verify's set/relation
comparison, so an expected ``(2, \infty)`` accepts a candidate ``x > 2``. A list is an ordered
comma-separated sequence compared member by member, which accepts the brackets a model does or
does not write around it and keeps a reordered answer wrong. Everything else is one expression.

Expected text that math-verify cannot parse raises ``InvalidTask``.
"""

import math
import re
import threading
from enum import StrEnum
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifyit.grade import (
    InvalidTask,
    Reward,
    empty_output_policy,
    invalid_task,
    numeric_tolerance,
    read_output,
    scored,
)
from verifyit.modes.extract import BOXED, extract_boxed, last_line, strip_math_delimiters
from verifyit.numeric import NumericCandidateError, extract_numeric_candidate, numeric_literal
from verifyit.spec import MathProfile, MathSpec, MathType, NumericSpec

SET_TYPES = frozenset({MathType.SET, MathType.INTERVAL})

CLOSERS = {"(": ")", "[": "]", "{": "}"}
SIZE_COMMANDS = ("\\left", "\\right", "\\big", "\\Big", "\\bigg", "\\Bigg")
TIMEOUT = 5
"""Seconds math-verify may spend parsing or comparing one expression. Its timeout arms
``signal.alarm``, which only the main thread may do, so a worker thread runs without it."""


def _timeout() -> int:
    return TIMEOUT if threading.current_thread() is threading.main_thread() else 0


def _parse(text: str) -> list:
    """math-verify's parse of ``text`` as a LaTeX expression, else of the text as written."""
    from math_verify import parse  # noqa: PLC0415

    timeout = _timeout()
    return parse(f"${strip_math_delimiters(text)}$", parsing_timeout=timeout, raise_on_error=True) or parse(
        text, parsing_timeout=timeout, raise_on_error=True
    )


def _verify(expected: Any, candidate: Any, allow_set_relation_comp: bool = False) -> bool:
    from math_verify import verify  # noqa: PLC0415

    return verify(
        expected,
        candidate,
        allow_set_relation_comp=allow_set_relation_comp,
        timeout_seconds=_timeout(),
        raise_on_error=True,
    )


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


class MathMemberPolicy(StrEnum):
    FINITE = "finite"
    LITERAL_SYMBOLIC = "literal_symbolic"


def canonical_math_members(text: str, *, policy: MathMemberPolicy = MathMemberPolicy.FINITE) -> tuple[str, ...]:
    """Prepare finite exact constants for Exact multiset grading inside a bounded worker.

    Tokens encode canonical expressions as hex so commas inside symbolic constructors
    cannot become Exact member separators. The literal-symbolic policy keeps parsed
    variable expressions as raw text; undefined expressions never become literal tokens.
    """
    from latex2sympy2_extended import NormalizationConfig  # noqa: PLC0415
    from math_verify import parse  # noqa: PLC0415
    from math_verify.parser import LatexExtractionConfig  # noqa: PLC0415
    from sympy import Expr, Float, Rational, nan, oo, simplify, srepr, zoo  # noqa: PLC0415

    if not isinstance(policy, MathMemberPolicy):
        raise ValueError("unknown math member policy")
    strict_latex = LatexExtractionConfig(
        normalization_config=NormalizationConfig(
            basic_latex=True, units=False, malformed_operators=False, nits=False, boxed="none", equations=False
        )
    )
    if not isinstance(text, str) or not text.strip():
        raise ValueError("math members require nonempty text")
    tokens = []
    for member in _split_members(text):
        if re.fullmatch(
            r"[+-]?(?:nan|inf|infinity|\\(?:mathrm|text|operatorname)\{(?:nan|inf|infinity)\})",
            member,
            flags=re.IGNORECASE,
        ):
            raise ValueError("math member must be finite")
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", member):
            value = Rational(member)
        else:
            parsed = [
                value
                for value in parse(
                    f"${member}$", extraction_config=[strict_latex], parsing_timeout=_timeout(), raise_on_error=True
                )
                if not isinstance(value, str)
            ]
            if len(parsed) != 1:
                raise ValueError("math member is missing or ambiguous")
            value = parsed[0]
        canonical = simplify(value) if isinstance(value, Expr) else value
        if isinstance(canonical, Expr) and canonical.has(nan, zoo, oo, -oo):
            raise ValueError("math member must be finite")
        if isinstance(value, Expr) and value.free_symbols and policy is MathMemberPolicy.LITERAL_SYMBOLIC:
            tokens.append("literal:" + member.encode("utf-8").hex())
            continue
        if (
            not isinstance(value, Expr)
            or value.free_symbols
            or getattr(value, "is_finite", None) is not True
            or value.has(Float)
        ):
            raise ValueError("math member must be a finite exact constant")
        if canonical.free_symbols or getattr(canonical, "is_finite", None) is not True or canonical.has(Float):
            raise ValueError("math member cannot be normalized exactly")
        tokens.append(srepr(canonical).encode("utf-8").hex())
    return tuple(tokens)


def _parsed_members(text: str) -> list[list]:
    return [_parse(member) for member in _split_members(text)]


def _members_match(expected: list[list], candidate: list[list]) -> bool:
    if len(expected) != len(candidate):
        return False
    return all(
        bool(parsed_candidate) and _verify(parsed_expected, parsed_candidate)
        for parsed_expected, parsed_candidate in zip(expected, candidate, strict=True)
    )


def _additive_constant_match(expected: list, candidate: list) -> bool:
    """Compare finite scalar expressions modulo a finite additive constant."""
    from math_verify.utils import timeout  # noqa: PLC0415
    from sympy import Expr, exp, nan, oo, simplify, zoo  # noqa: PLC0415
    from sympy.matrices.expressions import MatrixExpr  # noqa: PLC0415

    @timeout(_timeout())
    def difference_of(gold, prediction):
        return simplify((gold - prediction).rewrite(exp))

    for gold in expected:
        for prediction in candidate:
            if not all(isinstance(value, Expr) and not isinstance(value, MatrixExpr) for value in (gold, prediction)):
                continue
            if any(value.has(nan, zoo, oo, -oo) for value in (gold, prediction)):
                continue
            difference = difference_of(gold, prediction)
            if not difference.free_symbols and difference.is_finite is True:
                return True
    return False


def grade_math_candidate(spec: MathSpec, candidate: str) -> Reward:
    """Score extracted math content; backend deadlines become infrastructure failures."""
    empty_output_policy(spec)
    if type(spec.allow_additive_constant) is not bool:
        raise InvalidTask("allow_additive_constant must be boolean")
    if not isinstance(spec.profile, MathProfile) or not isinstance(spec.math_type, MathType):
        raise InvalidTask("unknown math parsing profile or math_type")
    from math_verify.errors import TimeoutException  # noqa: PLC0415

    try:
        return _grade_math_candidate(spec, candidate)
    except TimeoutException as error:
        # This backend exception inherits BaseException, unlike Python's TimeoutError.
        raise RuntimeError("math verifier deadline exhausted") from error


def _grade_raw_math(spec: MathSpec, candidate: str) -> Reward:
    from math_verify import parse  # noqa: PLC0415
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig  # noqa: PLC0415
    from math_verify.utils import timeout  # noqa: PLC0415

    @timeout(_timeout())
    def strings_of(values):
        return [str(value) for value in values]

    if spec.math_type is not MathType.SCALAR:
        raise InvalidTask("raw math profile requires scalar math_type")
    if BOXED in candidate:
        boxed = extract_boxed(candidate)
        candidate = f"\\boxed{{{boxed}}}" if boxed else ""
    gold_text = f"\\boxed{{{spec.expected}}}"
    expected = parse(
        gold_text, extraction_config=[LatexExtractionConfig()], parsing_timeout=_timeout(), raise_on_error=True
    )
    if not _is_expression(expected):
        raise InvalidTask("raw math profile could not parse the reference")
    parsed = parse(
        candidate,
        extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
        parsing_timeout=_timeout(),
        raise_on_error=True,
    )
    match = bool(parsed) and _verify(expected, parsed)
    chosen = None
    if parsed:
        gold_strings = strings_of(expected)
        prediction_strings = strings_of(parsed)
        chosen = next(
            (value for value in prediction_strings if any(_verify(gold, value) for gold in gold_strings)),
            prediction_strings[0],
        )
    if not match and spec.allow_additive_constant:
        latex_prediction = parse(
            candidate, extraction_config=[LatexExtractionConfig()], parsing_timeout=_timeout(), raise_on_error=True
        )
        match = _additive_constant_match(expected, latex_prediction)
        if match:
            chosen = next(
                strings_of([value])[0] for value in latex_prediction if _additive_constant_match(expected, [value])
            )
    return scored(float(match), extracted=candidate, expected=spec.expected, parsed_candidate=chosen)


def _grade_math_candidate(spec: MathSpec, candidate: str) -> Reward:
    if spec.profile is MathProfile.RAW:
        return _grade_raw_math(spec, candidate)
    if spec.profile is MathProfile.BOXED:
        from math_verify import parse, verify  # noqa: PLC0415

        if spec.math_type is not MathType.SCALAR:
            raise InvalidTask("boxed math profile requires scalar math_type")
        expected = parse(f"\\boxed{{{spec.expected}}}", parsing_timeout=_timeout(), raise_on_error=True)
        if not expected:
            raise InvalidTask("math-verify cannot parse boxed reference")
        parsed = parse(f"\\boxed{{{candidate}}}", parsing_timeout=_timeout(), raise_on_error=True)
        if not parsed:
            return scored(0.0, reason="missing_parse", expected_parsed=bool(expected), candidate_parsed=bool(parsed))
        match = bool(verify(gold=expected, target=parsed, timeout_seconds=_timeout(), raise_on_error=True))
        if not match and spec.allow_additive_constant:
            match = _additive_constant_match(expected, parsed)
        return scored(float(match), extracted=candidate, expected=spec.expected)
    is_list = spec.math_type is MathType.LIST
    expected = _parsed_members(spec.expected) if is_list else [_parse(spec.expected)]
    if not all(_is_expression(member) for member in expected):
        raise InvalidTask(f"math-verify cannot parse expected {spec.expected!r}")

    parsed = _parsed_members(candidate) if is_list else [_parse(candidate)]
    if not any(parsed):
        return scored(0.0, reason="unparsable", extracted=candidate, expected=spec.expected)

    if is_list:
        match = _members_match(expected, parsed)
    else:
        match = _verify(expected[0], parsed[0], allow_set_relation_comp=spec.math_type in SET_TYPES)
    if not match and spec.allow_additive_constant and not is_list:
        match = _additive_constant_match(expected[0], parsed[0])
    return scored(float(bool(match)), extracted=candidate, expected=spec.expected)


def _grade_symbolic(spec: MathSpec, workspace: Path) -> Reward:
    text = read_output(spec, workspace)
    if text is None:
        # Validate the reference even when no candidate was submitted.
        grade_math_candidate(spec, "")
        return scored(0.0, reason="no_output")
    if spec.profile is MathProfile.RAW:
        return grade_math_candidate(spec, text)
    boxed = extract_boxed(text)
    candidate = (boxed or "") if BOXED in text else last_line(text) or ""
    return grade_math_candidate(spec, candidate)


def _numeric_rows(value: object) -> list[list[float]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("numeric rows must be a nonempty matrix")
    rows = []
    for row in value:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError("numeric rows must have nonempty columns")
        if any(isinstance(cell, bool) or not isinstance(cell, (int, float)) for cell in row):
            raise ValueError("matrix cells must be numeric")
        converted = [float(cell) for cell in row]
        if not all(math.isfinite(cell) for cell in converted):
            raise ValueError("matrix cells must be finite")
        if rows and len(converted) != len(rows[0]):
            raise ValueError("numeric rows must be rectangular")
        rows.append(converted)
    return rows


def grade_regression_candidate(expected: object, candidate: object, *, variance_floor: float) -> Reward:
    """Grade groups of finite prediction matrices by pooled, uniform-output R².

    Group counts and each group shape must match before rows are pooled. Within
    each group, rows are observations and columns are outputs. A variance at or below
    the task's explicit floor yields zero because normalized error is undefined.
    Trusted references are validated before candidate data; no sklearn dependency.
    """
    try:
        if isinstance(variance_floor, bool) or not math.isfinite(variance_floor) or variance_floor < 0:
            raise ValueError("variance floor must be finite and nonnegative")
        if not isinstance(expected, (list, tuple)) or not expected:
            raise ValueError("reference must contain nonempty groups")
        groups = [_numeric_rows(group) for group in expected]
        truth = _numeric_rows([row for group in groups for row in group])
        columns = list(zip(*truth, strict=True))
        means = [math.fsum(column) / len(column) for column in columns]
        variances = [
            math.fsum((x - mean) ** 2 for x in column) / len(column) for column, mean in zip(columns, means, strict=True)
        ]
        deviations = [
            math.fsum(abs(x - mean) for x in column) / len(column) for column, mean in zip(columns, means, strict=True)
        ]
    except (ValueError, TypeError, OverflowError) as error:
        return invalid_task(str(error))
    try:
        if not isinstance(candidate, (list, tuple)) or len(candidate) != len(groups):
            raise ValueError("prediction group count differs from reference")
        predicted_groups = [_numeric_rows(group) for group in candidate]
        for reference, prediction in zip(groups, predicted_groups, strict=True):
            if len(prediction) != len(reference) or len(prediction[0]) != len(reference[0]):
                raise ValueError("prediction group shape differs from reference")
        prediction = [row for group in predicted_groups for row in group]
        if any(variance <= variance_floor for variance in variances):
            return scored(0, reason="undefined_variance")
        errors = [
            [actual - target for actual, target in zip(predicted, column, strict=True)]
            for predicted, column in zip(zip(*prediction, strict=True), columns, strict=True)
        ]
        nmse = math.fsum(
            math.fsum(x * x for x in error) / len(error) / variance
            for error, variance in zip(errors, variances, strict=True)
        ) / len(columns)
        nmae = math.fsum(
            math.fsum(abs(x) for x in error) / len(error) / deviation
            for error, deviation in zip(errors, deviations, strict=True)
        ) / len(columns)
        r2 = 1.0 - nmse
        if not all(math.isfinite(value) for value in (nmse, nmae, r2)):
            raise ValueError("prediction metrics are nonfinite")
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        return scored(0, reason="invalid_predictions", error=str(error))
    return scored(max(0.0, r2), nmse=nmse, nmae=nmae, r2=r2)


def grade_numeric_candidate_float(expected: float, value: float | None, *, tolerance_abs: float) -> Reward:
    """Compare source-normalized binary floats with an absolute tolerance."""
    if not math.isfinite(expected):
        raise InvalidTask("numeric reference must be finite")
    if not math.isfinite(tolerance_abs) or tolerance_abs < 0:
        raise InvalidTask("numeric tolerance must be finite and nonnegative")
    if value is None or not math.isfinite(value):
        return scored(0.0, reason="nonfinite_candidate", extracted=None, expected=expected)
    return scored(float(abs(value - expected) <= tolerance_abs), extracted=value, expected=expected)


def grade_numeric_candidate(spec: NumericSpec, value: Fraction) -> Reward:
    """Compare an exact scalar against validated private literals and tolerances."""
    empty_output_policy(spec)
    tolerance = numeric_tolerance(spec)
    if not isinstance(value, Fraction):
        raise TypeError("numeric candidate must be an exact Fraction")
    expected = numeric_literal(spec.expected)
    match = abs(value - expected) <= tolerance
    return scored(
        float(match),
        extracted=str(value),
        expected=spec.expected,
        tolerance_abs=spec.tolerance_abs,
        tolerance_rel=spec.tolerance_rel,
    )


def _grade_numeric(spec: NumericSpec, workspace: Path) -> Reward:
    empty_output_policy(spec)
    numeric_tolerance(spec)
    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    try:
        value = extract_numeric_candidate(text)
    except NumericCandidateError as error:
        return scored(0.0, reason="invalid_numeric_candidate", error=str(error))
    return grade_numeric_candidate(spec, value)


def grade(spec: MathSpec | NumericSpec, tests_dir: Path, workspace: Path) -> Reward:
    if isinstance(spec, NumericSpec):
        return _grade_numeric(spec, workspace)
    return _grade_symbolic(spec, workspace)
