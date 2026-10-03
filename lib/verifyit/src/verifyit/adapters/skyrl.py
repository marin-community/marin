# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Client extraction and canonicalization for SkyRL's existing exact grader."""

import ast
import json
import math
import re
import string
from decimal import Decimal, InvalidOperation
from fractions import Fraction

from verifyit.execution.worker import call_bounded
from verifyit.grade import Aggregation, InvalidTask, Reward, aggregate_rewards, scored
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.modes.grade_json_schema import grade_json_schema_candidate
from verifyit.modes.grade_math import MathMemberPolicy, canonical_math_members
from verifyit.spec import EmptyOutputPolicy, ExactSpec

RATIO = re.compile(r"(-?\d+):(-?\d+)")
GSM8K_FINAL = re.compile(r"#### (-?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?)")
GSM8K_STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def grade_literal_candidate(expected: str, candidate: str) -> Reward:
    """Compare literal strings through the existing exact mode's strict options."""
    spec = ExactSpec(
        expected=(expected,),
        ignore_case=False,
        ignore_whitespace=False,
        strip_outer_whitespace=False,
        empty_output=EmptyOutputPolicy.GRADE,
    )
    return grade_exact_candidate(spec, candidate)


def grade_gsm8k_strict(expected: str, response: str, *, format_score: float = 0.0) -> Reward:
    """Score the first source marker; the caller owns turn termination and feedback."""
    _gsm8k_expected(expected)
    match = GSM8K_STRICT.search(response)
    if match is None:
        return scored(0.0, reason="missing_answer_marker")
    answer = match.group(1).replace(",", "")
    result = grade_gsm8k_extracted(expected, answer)
    return scored(result.reward + (1.0 - result.reward) * format_score, **result.detail)


def _gsm8k_expected(expected: str) -> Decimal:
    try:
        value = Decimal(expected)
    except (InvalidOperation, ValueError, TypeError) as error:
        raise InvalidTask("GSM8K expected answer must be a finite decimal") from error
    if not value.is_finite():
        raise InvalidTask("GSM8K expected answer must be a finite decimal")
    return value


def grade_gsm8k_extracted(expected: str, candidate: str) -> Reward:
    """Validate numeric task data before literal strict/flexible comparison."""
    _gsm8k_expected(expected)
    return grade_literal_candidate(expected, candidate)


def _qa_normalize(answer: str) -> str:
    lower = answer.lower()
    unpunctuated = "".join(character for character in lower if character not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", unpunctuated).split())


def grade_search_em(targets: str | list[str], response: str) -> Reward:
    """Extract the last answer tag and compare normalized alternatives with exact."""
    alternatives = [targets] if isinstance(targets, str) else targets
    if (
        grade_json_schema_candidate({"type": "array", "minItems": 1, "items": {"type": "string"}}, alternatives).reward
        != 1
    ):
        raise InvalidTask("search requires nonempty string references")
    matches = list(ANSWER_TAG.finditer(response)) if isinstance(response, str) else []
    extracted = matches[-1].group(1).strip() if matches else None
    protocol = grade_json_schema_candidate({"type": "string"}, extracted)
    candidate = _qa_normalize(extracted) if extracted is not None else ""
    results = [grade_literal_candidate(_qa_normalize(target), candidate) for target in alternatives]
    alternatives_verdict = aggregate_rewards(results, expected_total=len(alternatives), policy=Aggregation.MAX)
    verdict = aggregate_rewards([protocol, alternatives_verdict], expected_total=2, policy=Aggregation.ALL)
    return Reward(verdict.reward, verdict.status, {**verdict.detail, "extracted": candidate})


def grade_rounded_candidate(expected: float, candidate: float | None) -> Reward:
    """Normalize banker-rounded data; schema and Exact own the comparisons."""
    schema = {"type": "number"}
    if grade_json_schema_candidate(schema, expected).reward != 1.0:
        raise InvalidTask("rounded expected answer must be finite")
    protocol = grade_json_schema_candidate(schema, candidate)
    try:
        normalized = "" if candidate is None else str(round(candidate))
    except (TypeError, ValueError, OverflowError):
        normalized = ""
    equality = grade_literal_candidate(str(round(expected)), normalized)
    return aggregate_rewards((protocol, equality), expected_total=2, policy=Aggregation.ALL)


def _typed_grid(value: object) -> object:
    """Frame only grid, row and scalar levels; preserve invalid cell types."""

    def scalar(item: object) -> dict:
        return {
            "kind": type(item).__name__,
            "value": item if isinstance(item, (str, int, float, bool)) or item is None else None,
        }

    if not isinstance(value, list):
        return scalar(value)
    return [[scalar(cell) for cell in row] if isinstance(row, list) else scalar(row) for row in value]


def grade_grid_candidate(expected: object, candidate: object) -> Reward:
    """Schema validates rectangular integer grids; Exact compares their data."""
    width = len(expected[0]) if isinstance(expected, list) and expected and isinstance(expected[0], list) else 0
    schema = {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "array",
            "minItems": max(width, 1),
            "maxItems": width,
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"const": "int"},
                    "value": {"type": "integer", "minimum": 0, "maximum": 9},
                },
                "required": ["kind", "value"],
                "additionalProperties": False,
            },
        },
    }
    reference = _typed_grid(expected)
    if grade_json_schema_candidate(schema, reference).reward != 1.0:
        raise InvalidTask("expected grid must be rectangular integer palette 0..9")
    response = _typed_grid(candidate)
    protocol = grade_json_schema_candidate(schema, response)
    equality = grade_literal_candidate(json.dumps(reference), json.dumps(response))
    return aggregate_rewards((protocol, equality), expected_total=2, policy=Aggregation.ALL)


def grade_aime_extracted(expected: str, candidate: str) -> Reward:
    """Grade trusted already-extracted answers; the caller must own the shared deadline."""
    expected_ratio = RATIO.fullmatch(expected) if isinstance(expected, str) else None
    candidate_ratio = RATIO.fullmatch(candidate) if isinstance(candidate, str) else None
    try:
        reference = canonical_math_members(
            "/".join(expected_ratio.groups()) if expected_ratio else expected, policy=MathMemberPolicy.LITERAL_SYMBOLIC
        )
    except ValueError as error:
        raise InvalidTask("AIME reference must define one valid exact answer") from error
    if len(reference) != 1:
        try:
            composite = ast.literal_eval(expected)
        except (SyntaxError, ValueError) as error:
            raise InvalidTask("AIME reference must define one valid exact answer") from error
        if not (
            expected.startswith("(")
            and expected.endswith(")")
            and isinstance(composite, tuple)
            and len(composite) >= 2
            and all(type(member) is int or (type(member) is float and math.isfinite(member)) for member in composite)
        ):
            raise InvalidTask("AIME reference must define one valid exact answer")
        # Source tuple answers compare their normalized spelling, not component values.
        return grade_literal_candidate(expected, candidate)
    try:
        tokens = canonical_math_members(
            "/".join(candidate_ratio.groups()) if candidate_ratio else candidate,
            policy=MathMemberPolicy.LITERAL_SYMBOLIC,
        )
    except ValueError:
        tokens = ()
    return grade_exact_candidate(ExactSpec(expected=reference, ordered=True), ",".join(tokens))


def grade_aime_candidate(expected: str, candidate: str) -> Reward:
    """Prepare one exact answer under a deadline, then delegate to Exact."""
    return call_bounded(grade_aime_extracted, expected, candidate, timeout=10)


def grade_gsm8k_final_line(expected: str, response: str) -> Reward:
    """Score a standalone final ``#### number`` line without float rounding."""
    lines = response.strip().splitlines()
    match = GSM8K_FINAL.fullmatch(lines[-1]) if lines else None
    if match is None:
        return scored(0.0, reason="missing_final_answer")
    candidate = match.group(1).replace(",", "")
    expected_value = _gsm8k_expected(expected)
    return grade_exact_candidate(ExactSpec(expected=(str(Fraction(expected_value)),)), str(Fraction(candidate)))
