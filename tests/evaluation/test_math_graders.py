# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for the self-contained MATH graders.

``data/math_grader_benchmark.jsonl`` carries the verdict lm-evaluation-harness
``d5e3391`` produced for each of its 100 problems, so parity is checked without
installing the harness

The single-case tests cover divergences a plausible fast path gets wrong, each
observed against the reference rather than invented.
"""

import json
from pathlib import Path

import pytest
from marin.evaluation.graders import hendrycks_math, minerva_math

from tests.test_utils import skip_if_module_missing

BENCHMARK = Path(__file__).parent / "data" / "math_grader_benchmark.jsonl"

# The sympy fallback path is an ANTLR-generated parser, so it needs both halves.
requires_sympy = skip_if_module_missing("sympy")
requires_antlr = skip_if_module_missing("antlr4")


@pytest.fixture(scope="module")
def benchmark():
    records = [json.loads(line) for line in BENCHMARK.read_text().splitlines()]
    assert len(records) == 100
    return records


def test_hendrycks_matches_harness_on_benchmark(benchmark):
    mismatched = [
        record
        for record in benchmark
        if hendrycks_math.grade(record["problem"], record["hendrycks_solution"], record["reference_answer"])
        != record["hendrycks_reference_grade"]
    ]
    assert mismatched == []


@requires_sympy
@requires_antlr
def test_minerva_matches_harness_on_benchmark(benchmark):
    mismatched = [
        record
        for record in benchmark
        if minerva_math.grade(record["problem"], record["minerva_solution"], record["reference_answer"])
        != record["minerva_reference_grade"]
    ]
    assert mismatched == []


def test_minerva_extracts_final_answer_template():
    solution = "Some work.\nFinal Answer: The final answer is $42$. I hope it is correct."
    assert minerva_math.extract_answer(solution) == "$42$"


def test_minerva_reports_invalid_when_template_absent():
    assert minerva_math.extract_answer("The answer is 42.") == minerva_math.INVALID_ANSWER


@pytest.mark.parametrize(
    ("candidate", "reference", "equivalent"),
    [
        # Bracketed comma lists are outside sympy's grammar, so the reference
        # scores them 0 even against an identical string.
        ("[2,5)", "[2,5)", False),
        ("(1,3)", "(1,3)", False),
        # A comma before three digits is a thousands separator, so this parses.
        ("(100,101)", "(100,101)", True),
        ("(100,101)", "100101", True),
        # The comma must be inside the brackets to prove a parse failure.
        ("(E),", "(E),", True),
        # The grammar does have a comma production for call arguments, so a
        # bracketed function call parses and must not be rejected on sight.
        ("(f(x,y))", "(f(x,y))", True),
        ("[f(x,y)]", "[f(x,y)]", True),
        ("(2f(x,y))", "(2f(x,y))", True),
        ("(x+f(a,b))", "(x+f(a,b))", True),
        # Decimals compare exactly against rationals.
        ("0.5", "\\frac{1}{2}", True),
        ("0.3", "\\frac{3}{10}", True),
        ("0.333", "\\frac{1}{3}", False),
        ("1.50", "1.5", True),
        ("-0", "0", True),
        ("2", "\\frac{4}{2}", True),
        # Leading zeros are a Python int-literal syntax error inside sympy.
        ("007", "7", False),
        ("\\frac{007}{2}", "\\frac{7}{2}", False),
        # sympy's grammar wants a digit before the point.
        (".5", ".5", False),
    ],
)
@requires_sympy
@requires_antlr
def test_minerva_equivalence_matches_reference(candidate, reference, equivalent):
    assert minerva_math.is_equiv(candidate, reference) is equivalent


@requires_sympy
@requires_antlr
def test_minerva_scores_relation_against_itself_as_unequal():
    """Parsing a relation yields an object that cannot be subtracted from itself."""
    assert minerva_math.is_equiv("-80\\leqg(x)\\leq82", "-80\\leqg(x)\\leq82") is False


def test_hendrycks_normalizes_equivalent_spellings():
    assert hendrycks_math.is_equiv("\\dfrac{1}{2}", "\\frac{1}{2}")
    assert hendrycks_math.is_equiv("0.5", "\\frac{1}{2}")
    assert hendrycks_math.is_equiv("\\left(3\\right)", "(3)")
    assert hendrycks_math.is_equiv("50\\%", "50")
    assert not hendrycks_math.is_equiv("\\frac{1}{2}", "\\frac{1}{3}")


def test_hendrycks_strips_percent_escapes_created_by_an_earlier_removal():
    """``str.replace`` is single-pass, so the reference's repeated ``\\%`` strip matters.

    Collapsing ``\\\\`` to ``\\`` leaves ``\\\\%%``; one pass removes the middle
    ``\\%`` and leaves a newly adjacent one that only a second pass catches.
    """
    assert hendrycks_math.strip_string(r"\\\%%") == ""
    assert hendrycks_math.is_equiv(r"\\\%%", r"\\%")


def test_hendrycks_falls_back_to_raw_equality_when_normalization_raises():
    """A non-integer ``a/b`` makes ``int()`` raise, dropping ``is_equiv`` to raw equality."""
    assert hendrycks_math.is_equiv("x/y", "x/y")
    assert not hendrycks_math.is_equiv("x/y", "x/y ")


def test_hendrycks_extracts_span_between_outermost_dollars():
    assert hendrycks_math.extract_answer(" The answer is $42$.") == "42"
    # Fewer than two delimiters means the whole completion is the answer.
    assert hendrycks_math.extract_answer("42") == "42"


def test_graders_need_only_problem_solution_and_reference():
    """The issue's contract: no dataset doc, no harness state."""
    assert (
        minerva_math.grade(
            "irrelevant problem text",
            "Final Answer: The final answer is $\\frac{1}{2}$. I hope it is correct.",
            "0.5",
        )
        == 1.0
    )
    assert hendrycks_math.grade("irrelevant problem text", " $0.5$", "\\frac{1}{2}") == 1.0
