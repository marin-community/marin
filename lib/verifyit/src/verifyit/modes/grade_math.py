# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

r"""Grade mathematical answers, either symbolically or as a number with tolerance.

The default anchored profile takes the last ``\boxed{}`` expression, or the last non-empty line when the output has
none. References parse as a whole LaTeX expression. Candidates first use
anchored LaTeX (``$...$``), then math-verify's own anchors (``the answer is ...``) if that yields
nothing. Other profiles retain raw source extraction or require a boxed expression; their parser
options remain task-controlled.

``math_type`` selects the comparison. Set and interval answers allow math-verify's set/relation
comparison, so an expected ``(2, \infty)`` accepts a candidate ``x > 2``. A list is an ordered
comma-separated sequence compared member by member, which accepts the brackets a model does or
does not write around it and keeps a reordered answer wrong. Everything else is one expression.

Expected text that math-verify cannot parse raises ``InvalidTask``.
"""

import math
import re
import threading
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

from verifyit.compat import StrEnum
from verifyit.grade import (
    GradingInfraError,
    InvalidTask,
    Reward,
    empty_output_policy,
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
MATH_VERIFIER_DEADLINE = "math_verifier_deadline"
MATH_DEADLINE_REFERENCE_PARSE = "reference_parse"
MATH_DEADLINE_CANDIDATE_PARSE = "candidate_parse"
MATH_DEADLINE_COMPARISON = "comparison"
"""Which grading step ran out: the reference parse is a task property, while the
candidate parse and the comparison follow the reply."""
TIMEOUT = 5
"""Seconds math-verify may spend parsing or comparing one expression. Its timeout arms
``signal.alarm``, which only the main thread may do, so a worker thread runs without it."""


def _timeout() -> int:
    return TIMEOUT if threading.current_thread() is threading.main_thread() else 0


_DANGLING_OPERATOR = re.compile(r"(?:[<>=+\-*/^]|\\(?:leq?|geq?|lt|gt|neq?|cdot|times))\s*$")
TEXT_CONNECTOR_CONTENT = r"\s*(?:and|or)\s*"
TEXT_GROUP_COMMAND = r"\\(?:text|mbox|mathrm)"
REFERENCE_TEXT_GROUP = re.compile(TEXT_GROUP_COMMAND + r"\{(?!" + TEXT_CONNECTOR_CONTENT + r"\})")
LEADING_LIST_CONNECTOR = re.compile(r"^\\text\{" + TEXT_CONNECTOR_CONTENT + r"\}\s*")
# Unit normalization strips terminal text groups and their powers.
TERMINAL_UNIT_POWER = r"\^(?:[+-]?[0-9]|\{\s*[+-]?[0-9]+\s*\})"
UNIT_GROUP_CONTENT_PATTERN = r"[\s~]*[A-Za-z]+"
TERMINAL_UNIT_GROUP = TEXT_GROUP_COMMAND + r"\{" + UNIT_GROUP_CONTENT_PATTERN + r"\}"
TERMINAL_UNIT_SUFFIX = re.compile(
    rf"(?:{TERMINAL_UNIT_POWER})?(?:\s*(?:/|\\cdot|\\times)\s*{TERMINAL_UNIT_GROUP}(?:{TERMINAL_UNIT_POWER})?)*"
)
UNIT_GROUP_CONTENT = re.compile(UNIT_GROUP_CONTENT_PATTERN)
ORDINAL_SUFFIX = re.compile(r"(?<=\d)\^\{\\text\{(?:st|nd|rd|th)\}\}(?=\s*(?:" + TEXT_GROUP_COMMAND + r"\{|$))")


def _math_verify_parse(text: str, extraction_config: Sequence[object] | None = None, **kwargs: Any) -> list:
    """Parse math, retaining trailing variables that unit normalization removed."""
    from math_verify import parse  # noqa: PLC0415
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig  # noqa: PLC0415

    configs = list(extraction_config or [LatexExtractionConfig(), ExprExtractionConfig()])
    parsed = parse(text, extraction_config=configs, **kwargs)
    if not any(isinstance(item, str) and _DANGLING_OPERATOR.search(item) for item in parsed):
        return parsed
    unitless = [
        (
            replace(config, normalization_config=replace(config.normalization_config, units=False))
            if isinstance(config, LatexExtractionConfig)
            else config
        )
        for config in configs
    ]
    return parse(text, extraction_config=unitless, **kwargs)


def _parse(text: str, *, strip_units: bool = True) -> list:
    """Parse a candidate as anchored LaTeX, else as written."""
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig  # noqa: PLC0415

    timeout = _timeout()
    latex = LatexExtractionConfig()
    if not strip_units:
        latex = replace(latex, normalization_config=replace(latex.normalization_config, units=False))
    extraction_config = [latex, ExprExtractionConfig()]
    return _math_verify_parse(
        f"${strip_math_delimiters(text)}$",
        extraction_config=extraction_config,
        parsing_timeout=timeout,
        raise_on_error=True,
    ) or _math_verify_parse(text, extraction_config=extraction_config, parsing_timeout=timeout, raise_on_error=True)


def _parse_reference(text: str) -> tuple[list, bool]:
    """Parse a whole reference and return whether candidates should strip units."""
    from latex2sympy2_extended import latex2sympy, normalize_latex  # noqa: PLC0415
    from math_verify.grader import should_treat_as_complex  # noqa: PLC0415
    from math_verify.parser import LatexExtractionConfig  # noqa: PLC0415
    from math_verify.utils import timeout  # noqa: PLC0415

    # Normalization would otherwise leave a dangling exponent before an ordinal text group.
    value = ORDINAL_SUFFIX.sub("", strip_math_delimiters(text))
    text_groups = list(REFERENCE_TEXT_GROUP.finditer(value))
    strip_units = bool(text_groups)
    symbolic_text_group = False
    if re.search(r"(?<!\\)\$", value) or any(delimiter in value for delimiter in (r"\(", r"\)", r"\[", r"\]")):
        return [], strip_units
    for text_group in text_groups:
        # Unit normalization removes a terminal text-group suffix. It can
        # also swallow any intervening expression up to the final closing brace.
        _, first_close = _scan_brackets(value[text_group.end() - 1 :])
        if first_close is None:
            return [], strip_units
        group_contents = value[text_group.end() : text_group.end() + first_close - 1]
        suffix = value[text_group.end() + first_close :].strip()
        terminal_unit_suffix = TERMINAL_UNIT_SUFFIX.fullmatch(suffix)
        if suffix and terminal_unit_suffix and not UNIT_GROUP_CONTENT.fullmatch(group_contents):
            return [], strip_units
        if suffix and not terminal_unit_suffix:
            prefix = value[: text_group.start()].rstrip()
            upright_symbol = text_group.group() == r"\mathrm{" and len(group_contents) == 1 and group_contents.isalpha()
            no_gap_before_group = text_group.start() > 0 and not value[text_group.start() - 1].isspace()
            if not re.fullmatch(r"[^\s~]+", group_contents) or (
                prefix and prefix[-1] not in "([{=+-*/^" and not upright_symbol and not no_gap_before_group
            ):
                return [], strip_units
            # A text symbol starting a term or adjacent to one is part of the expression.
            symbolic_text_group = True
            strip_units = False
        if text_group.start() == 0 and len(group_contents.split()) > 4:
            return [], strip_units
    latex = LatexExtractionConfig(boxed_match_priority=-1)

    # math-verify can match an inner math fragment or retry only the last equation.
    @timeout(_timeout())
    def parse_whole() -> tuple[object, bool]:
        normalized = normalize_latex(value, config=latex.normalization_config)
        candidate_strip_units = not symbolic_text_group and (strip_units or normalized == value)
        if strip_units and _DANGLING_OPERATOR.search(normalized):
            # A terminal text symbol can be the operand that unit normalization removed.
            candidate_strip_units = False
        if not candidate_strip_units:
            # Other normalization changes, such as \dfrac -> \frac, do not imply a stripped unit.
            unitless = normalize_latex(value, config=replace(latex.normalization_config, units=False))
            candidate_strip_units = not symbolic_text_group and normalized == unitless
            if not candidate_strip_units:
                normalized = unitless
        parsed = latex2sympy(
            normalized,
            is_real=not should_treat_as_complex(normalized),
            convert_degrees=False,
            normalization_config=None,
        )
        return parsed, candidate_strip_units

    try:
        parsed, strip_units = parse_whole()
    except Exception as error:
        # latex2sympy reports syntax errors as bare Exception.
        raise InvalidTask(f"math-verify cannot parse expected {text!r}") from error
    return ([parsed] if parsed is not None else []), strip_units


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else [value]


def _is_relation_chain(value: Any) -> bool:
    from sympy import And  # noqa: PLC0415
    from sympy.core.relational import Relational  # noqa: PLC0415

    return isinstance(value, And) and all(isinstance(arg, Relational) for arg in value.args)


def _is_undefined(value: Any) -> bool:
    from sympy import Basic, S  # noqa: PLC0415

    return isinstance(value, Basic) and value.has(S.NaN, S.ComplexInfinity)


def _verify(expected: Any, candidate: Any, allow_set_relation_comp: bool = False) -> bool:
    """Whether any expected parse equals any candidate parse."""
    return any(
        _verify_pair(gold, target, allow_set_relation_comp)
        for gold in _as_list(expected)
        for target in _as_list(candidate)
    )


def _verify_pair(gold: Any, target: Any, allow_set_relation_comp: bool) -> bool:
    from math_verify import verify  # noqa: PLC0415
    from sympy import Eq, Ne  # noqa: PLC0415

    if (_is_relation_chain(gold) or _is_relation_chain(target)) and (
        not allow_set_relation_comp or _is_relation_chain(gold) == _is_relation_chain(target)
    ):
        # A chain a < b < c parses as a conjunction of relations, which SymPy
        # deduplicates (m < -n < n < -m keeps two: n < -m is m < -n) and orders.
        # math-verify zips two conjunctions in order with strict=True and raises
        # when their lengths differ (huggingface/Math-Verify#93). Two conjunctions
        # state the same constraints when each relation of one matches one of the other.
        gold_relations = gold.args if _is_relation_chain(gold) else (gold,)
        target_relations = target.args if _is_relation_chain(target) else (target,)
        return all(
            any(_verify_pair(g, t, allow_set_relation_comp) for t in target_relations) for g in gold_relations
        ) and all(any(_verify_pair(g, t, allow_set_relation_comp) for g in gold_relations) for t in target_relations)
    if isinstance(gold, Ne) or isinstance(target, Ne):
        # math-verify's relation table lacks Unequality. a != b holds exactly where
        # a = b fails, so two unequalities match when their equations do.
        if not (isinstance(gold, Ne) and isinstance(target, Ne)):
            return False
        gold, target = Eq(gold.lhs, gold.rhs, evaluate=False), Eq(target.lhs, target.rhs, evaluate=False)
    if _is_undefined(gold):
        raise InvalidTask(f"the reference parses to an undefined value: {gold}")
    if _is_undefined(target):
        # math-verify refuses to compare nan or zoo; a reference that is defined
        # cannot equal it.
        return False
    try:
        return verify(
            gold,
            target,
            allow_set_relation_comp=allow_set_relation_comp,
            timeout_seconds=_timeout(),
            raise_on_error=True,
        )
    except Exception as error:
        # A comparison failure not handled above: neither a verdict on the answer nor
        # a property of the task, so it surfaces instead of scoring silently.
        raise GradingInfraError(f"math-verify could not compare: {type(error).__name__}: {error}") from error


def _is_expression(parsed: list) -> bool:
    """Whether math-verify recovered an expression rather than only the original string."""
    return any(not isinstance(item, str) for item in parsed)


def _scan_brackets(value: str) -> tuple[list[tuple[int, int]], int | None]:
    """Walk ``value`` once, tracking bracket nesting.

    Returns the spans of the comma-separated top-level members — commas inside brackets
    never split — and the index where the bracket stack first empties again, which for a
    ``value[0]`` that opens a bracket is exactly where its pair closes.
    """
    spans: list[tuple[int, int]] = []
    stack: list[str] = []
    first_close: int | None = None
    start = 0
    for index, char in enumerate(value):
        if char in CLOSERS:
            stack.append(CLOSERS[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack and first_close is None:
                first_close = index
        elif char == "," and not stack:
            spans.append((start, index))
            start = index + 1
    spans.append((start, len(value)))
    return spans, first_close


def _split_members(text: str) -> list[str]:
    """The comma-separated members of a sequence, ignoring commas nested inside brackets.

    One optional layer of enclosing brackets is dropped, so ``[1, 2]`` and ``1, 2`` read alike,
    but only when the first bracket opens the whole text — its balanced close at the last
    character: ``[1, 2], [3, 4]`` keeps both bracketed members, and ``(1,2),(3,4)`` keeps both
    parenthesized pairs.
    """
    value = strip_math_delimiters(text)
    for command in SIZE_COMMANDS:
        value = value.replace(command, "")
    value = value.strip()
    spans, first_close = _scan_brackets(value)
    if len(value) > 1 and value[0] in CLOSERS and value[-1] == CLOSERS[value[0]] and first_close == len(value) - 1:
        # The first opener encloses the whole text: the members are its interior's top level.
        value = value[1:-1]
        spans = _scan_brackets(value)[0]
    return [value[start:end].strip() for start, end in spans]


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
                for value in _math_verify_parse(
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
            or not value.is_commutative
            or getattr(value, "is_finite", None) is not True
            or value.has(Float)
        ):
            raise ValueError("math member must be a finite exact constant")
        if canonical.free_symbols or getattr(canonical, "is_finite", None) is not True or canonical.has(Float):
            raise ValueError("math member cannot be normalized exactly")
        tokens.append(srepr(canonical).encode("utf-8").hex())
    return tuple(tokens)


def _candidate_parse(parse: Callable[[], list]) -> list:
    """A candidate's parse, empty when SymPy cannot build its literal."""
    from sympy import SympifyError  # noqa: PLC0415

    try:
        return parse()
    except (SympifyError, SyntaxError):
        return []


def _members_match(expected: list[list], candidate: list[list]) -> bool:
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


def _positive_symbol_match(expected: list, candidate: list, positive_symbols: tuple[str, ...]) -> bool:
    """Prove scalar equality after applying only the contract's positive assumptions."""
    from math_verify.utils import timeout  # noqa: PLC0415
    from sympy import Expr, Symbol, simplify  # noqa: PLC0415
    from sympy.matrices.expressions import MatrixExpr  # noqa: PLC0415

    @timeout(_timeout())
    def equal_under_assumptions(gold, target) -> bool:
        replacements = {
            symbol: Symbol(symbol.name, positive=True)
            for symbol in gold.free_symbols | target.free_symbols
            if symbol.name in positive_symbols
        }
        return simplify(gold.xreplace(replacements) - target.xreplace(replacements)) == 0

    return any(
        equal_under_assumptions(gold, target)
        for gold in expected
        for target in candidate
        if isinstance(gold, Expr)
        and isinstance(target, Expr)
        and not isinstance(gold, MatrixExpr)
        and not isinstance(target, MatrixExpr)
        and gold.is_commutative
        and target.is_commutative
        and not _is_undefined(gold)
        and not _is_undefined(target)
    )


def validate_math(spec: MathSpec) -> None:
    """Check the parsing policy; the reference itself is parsed only when grading."""
    empty_output_policy(spec)
    if type(spec.allow_additive_constant) is not bool:
        raise InvalidTask("allow_additive_constant must be boolean")
    if not isinstance(spec.profile, MathProfile) or not isinstance(spec.math_type, MathType):
        raise InvalidTask("unknown math parsing profile or math_type")
    if (
        not isinstance(spec.positive_symbols, tuple)
        or any(
            not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
            for name in spec.positive_symbols
        )
        or len(set(spec.positive_symbols)) != len(spec.positive_symbols)
    ):
        raise InvalidTask("positive_symbols must contain unique symbol names")
    if spec.positive_symbols and spec.math_type is not MathType.SCALAR:
        raise InvalidTask("positive_symbols requires scalar math_type")


def grade_math_candidate(spec: MathSpec, candidate: str) -> Reward:
    """Score extracted math content; backend deadlines become infrastructure failures."""
    validate_math(spec)
    from math_verify.errors import TimeoutException  # noqa: PLC0415

    try:
        return _grade_math_candidate(spec, candidate)
    except TimeoutException as error:
        # This backend exception inherits BaseException, unlike Python's TimeoutError.
        # Phase-tagged deadlines are raised by _deadline_phase below; this catch is the
        # fallback for any site outside a tagged phase.
        raise GradingInfraError("math verifier deadline exhausted", code=MATH_VERIFIER_DEADLINE) from error


@contextmanager
def _deadline_phase(phase: str):
    """Re-raise a math-verify deadline from one grading step with its phase tagged.

    The reference parse is a property of the task; the candidate parse and the
    comparison follow the reply.
    """
    from math_verify.errors import TimeoutException  # noqa: PLC0415

    try:
        yield
    except TimeoutException as error:
        raise GradingInfraError("math verifier deadline exhausted", code=MATH_VERIFIER_DEADLINE, phase=phase) from error


def _grade_raw_math(spec: MathSpec, candidate: str) -> Reward:
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
    with _deadline_phase(MATH_DEADLINE_REFERENCE_PARSE):
        expected = _math_verify_parse(
            gold_text, extraction_config=[LatexExtractionConfig()], parsing_timeout=_timeout(), raise_on_error=True
        )
    if not _is_expression(expected):
        raise InvalidTask("raw math profile could not parse the reference")
    with _deadline_phase(MATH_DEADLINE_CANDIDATE_PARSE):
        parsed = _math_verify_parse(
            candidate,
            extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()],
            parsing_timeout=_timeout(),
            raise_on_error=True,
        )
    with _deadline_phase(MATH_DEADLINE_COMPARISON):
        match = bool(parsed) and _verify(expected, parsed)
        if not match and spec.positive_symbols:
            match = _positive_symbol_match(expected, parsed, spec.positive_symbols)
        chosen = None
        if parsed:
            gold_strings = strings_of(expected)
            prediction_strings = strings_of(parsed)
            chosen = next(
                (value for value in prediction_strings if any(_verify(gold, value) for gold in gold_strings)),
                prediction_strings[0],
            )
        if not match and spec.allow_additive_constant:
            latex_prediction = _math_verify_parse(
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

        if spec.math_type is not MathType.SCALAR:
            raise InvalidTask("boxed math profile requires scalar math_type")
        with _deadline_phase(MATH_DEADLINE_REFERENCE_PARSE):
            expected = _math_verify_parse(f"\\boxed{{{spec.expected}}}", parsing_timeout=_timeout(), raise_on_error=True)
        if not expected:
            raise InvalidTask("math-verify cannot parse boxed reference")
        with _deadline_phase(MATH_DEADLINE_CANDIDATE_PARSE):
            parsed = _candidate_parse(
                lambda: _math_verify_parse(f"\\boxed{{{candidate}}}", parsing_timeout=_timeout(), raise_on_error=True)
            )
        if not parsed:
            return scored(0.0, reason="missing_parse", expected_parsed=bool(expected), candidate_parsed=bool(parsed))
        with _deadline_phase(MATH_DEADLINE_COMPARISON):
            match = bool(_verify(expected, parsed))
            if not match and spec.positive_symbols:
                match = _positive_symbol_match(expected, parsed, spec.positive_symbols)
            if not match and spec.allow_additive_constant:
                match = _additive_constant_match(expected, parsed)
        return scored(float(match), extracted=candidate, expected=spec.expected)
    is_list = spec.math_type is MathType.LIST
    reference_members = _split_members(spec.expected) if is_list else [spec.expected]
    if is_list:
        # A connector after a list comma normalizes to another leading comma.
        reference_members = [LEADING_LIST_CONNECTOR.sub("", member) for member in reference_members]
    with _deadline_phase(MATH_DEADLINE_REFERENCE_PARSE):
        references = [_parse_reference(member) for member in reference_members]
    expected = [parsed for parsed, _ in references]
    if not all(_is_expression(member) for member in expected):
        raise InvalidTask(f"math-verify cannot parse expected {spec.expected!r}")

    candidate_members = _split_members(candidate) if is_list else [candidate]
    if len(candidate_members) != len(expected):
        return scored(0.0, extracted=candidate, expected=spec.expected)
    with _deadline_phase(MATH_DEADLINE_CANDIDATE_PARSE):
        parsed = _candidate_parse(
            lambda: [
                _parse(member, strip_units=strip_units)
                for member, (_, strip_units) in zip(candidate_members, references, strict=True)
            ]
        )
    if not any(parsed):
        return scored(0.0, reason="unparsable", extracted=candidate, expected=spec.expected)

    with _deadline_phase(MATH_DEADLINE_COMPARISON):
        if is_list:
            match = _members_match(expected, parsed)
        else:
            match = _verify(expected[0], parsed[0], allow_set_relation_comp=spec.math_type in SET_TYPES)
            if not match and spec.positive_symbols:
                match = _positive_symbol_match(expected[0], parsed[0], spec.positive_symbols)
        if not match and spec.allow_additive_constant and not is_list:
            match = _additive_constant_match(expected[0], parsed[0])
    return scored(float(bool(match)), extracted=candidate, expected=spec.expected)


def math_answer(spec: MathSpec, text: str) -> str:
    """Keep raw responses whole; extract the last box or line for other profiles."""
    if spec.profile is MathProfile.RAW:
        return text
    if BOXED in text:
        return extract_boxed(text) or ""
    return last_line(text) or ""


def _grade_symbolic(spec: MathSpec, workspace: Path) -> Reward:
    text = read_output(spec, workspace)
    if text is None:
        # Validate the reference even when no candidate was submitted.
        grade_math_candidate(spec, "")
        return scored(0.0, reason="no_output")
    return grade_math_text_candidate(spec, text)


def grade_math_text_candidate(spec: MathSpec, text: str) -> Reward:
    """Extract math under the task's parsing profile and grade the result."""
    return grade_math_candidate(spec, math_answer(spec, text))


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


def validate_numeric(spec: NumericSpec) -> None:
    empty_output_policy(spec)
    numeric_tolerance(spec)


def grade_numeric_text_candidate(spec: NumericSpec, text: str) -> Reward:
    """Extract one exact numeric literal and apply the task's tolerances."""
    try:
        value = extract_numeric_candidate(text)
    except NumericCandidateError as error:
        return scored(0.0, reason="invalid_numeric_candidate", error=str(error))
    return grade_numeric_candidate(spec, value)


def _grade_numeric(spec: NumericSpec, workspace: Path) -> Reward:
    validate_numeric(spec)
    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    return grade_numeric_text_candidate(spec, text)


def grade(spec: MathSpec | NumericSpec, tests_dir: Path, workspace: Path) -> Reward:
    if isinstance(spec, NumericSpec):
        return _grade_numeric(spec, workspace)
    return _grade_symbolic(spec, workspace)
