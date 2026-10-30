"""Parity tests for the self-contained MATH graders in eval/graders.

lm-eval is a base dependency here, so every case is checked against the pinned
harness live rather than against recorded verdicts: the reference functions are
imported and run side by side with the port.

``data/math_grader_benchmark.jsonl`` holds 100 MATH test problems spanning all
seven subjects, each with a synthesized completion in the format its task's
prompt elicits, mixing verbatim, equivalently reformatted, wrong, and malformed
answers.

The single-case tests cover divergences a plausible fast path gets wrong, each
observed against the reference rather than invented.
"""

import json
import pathlib
import sys

import pytest
from lm_eval.tasks.hendrycks_math import utils as reference_hendrycks
from lm_eval.tasks.minerva_math import utils as reference_minerva

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.graders import hendrycks_math, minerva_math  # noqa: E402

BENCHMARK = pathlib.Path(__file__).parent / "data" / "math_grader_benchmark.jsonl"


@pytest.fixture(scope="module")
def benchmark():
    records = [json.loads(line) for line in BENCHMARK.read_text().splitlines()]
    assert len(records) == 100
    return records


def reference_hendrycks_grade(solution: str, reference_answer: str) -> float:
    """The harness ``process_results`` extraction and comparison, inlined."""
    indices = [pos for pos, char in enumerate(solution) if char == "$"]
    answer = solution if len(indices) <= 1 else solution[indices[0] + 1 : indices[-1]]
    return 1.0 if reference_hendrycks.is_equiv(answer, reference_answer) else 0.0


def reference_minerva_grade(solution: str, reference_answer: str) -> float:
    """The harness ``process_results`` exact_match metric, inlined."""
    candidate = reference_minerva.normalize_final_answer(reference_minerva.get_unnormalized_answer(solution))
    gold = reference_minerva.normalize_final_answer(reference_answer)
    return 1.0 if reference_minerva.is_equiv(candidate, gold) else 0.0


def test_hendrycks_matches_harness_on_benchmark(benchmark):
    mismatched = [
        record["reference_answer"]
        for record in benchmark
        if hendrycks_math.grade(record["problem"], record["hendrycks_solution"], record["reference_answer"])
        != reference_hendrycks_grade(record["hendrycks_solution"], record["reference_answer"])
    ]
    assert mismatched == []


def test_minerva_matches_harness_on_benchmark(benchmark):
    mismatched = [
        record["reference_answer"]
        for record in benchmark
        if minerva_math.grade(record["problem"], record["minerva_solution"], record["reference_answer"])
        != reference_minerva_grade(record["minerva_solution"], record["reference_answer"])
    ]
    assert mismatched == []


def test_hendrycks_normalization_matches_harness_on_benchmark_answers(benchmark):
    """strip_string is reproduced verbatim, so it must agree character for character."""
    for record in benchmark:
        answer = record["reference_answer"]
        assert hendrycks_math.strip_string(answer) == reference_hendrycks.strip_string(answer)


def test_minerva_normalization_matches_harness_on_benchmark_answers(benchmark):
    for record in benchmark:
        answer = record["reference_answer"]
        assert minerva_math.normalize_final_answer(answer) == reference_minerva.normalize_final_answer(answer)
        extracted = minerva_math.extract_answer(record["minerva_solution"])
        assert extracted == reference_minerva.get_unnormalized_answer(record["minerva_solution"])


@pytest.mark.parametrize(
    ("candidate", "reference"),
    [
        # Bracketed comma lists are outside sympy's grammar, so the reference
        # scores them 0 even against an identical string.
        ("[2,5)", "[2,5)"),
        ("(1,3)", "(1,3)"),
        # A comma before three digits is a thousands separator, so this parses.
        ("(100,101)", "(100,101)"),
        ("(100,101)", "100101"),
        # The comma must be inside the brackets to prove a parse failure.
        ("(E),", "(E),"),
        # The grammar does have a comma production for call arguments, so a
        # bracketed function call parses and must not be rejected on sight.
        ("(f(x,y))", "(f(x,y))"),
        ("[f(x,y)]", "[f(x,y)]"),
        ("(2f(x,y))", "(2f(x,y))"),
        ("(x+f(a,b))", "(x+f(a,b))"),
        # Decimals compare exactly against rationals.
        ("0.5", "\\frac{1}{2}"),
        ("0.3", "\\frac{3}{10}"),
        ("0.333", "\\frac{1}{3}"),
        ("1.50", "1.5"),
        ("-0", "0"),
        ("2", "\\frac{4}{2}"),
        # Leading zeros are a Python int-literal syntax error inside sympy.
        ("007", "7"),
        ("\\frac{007}{2}", "\\frac{7}{2}"),
        # sympy's grammar wants a digit before the point.
        (".5", ".5"),
        # A parsed relation cannot be subtracted from itself.
        ("-80\\leqg(x)\\leq82", "-80\\leqg(x)\\leq82"),
    ],
)
def test_minerva_equivalence_matches_reference(candidate, reference):
    assert minerva_math.is_equiv(candidate, reference) is bool(reference_minerva.is_equiv(candidate, reference))


def test_hendrycks_strips_percent_escapes_created_by_an_earlier_removal():
    """``str.replace`` is single-pass, so the reference's repeated ``\\%`` strip matters.

    Collapsing ``\\\\`` to ``\\`` leaves ``\\\\%%``; one pass removes the middle
    ``\\%`` and leaves a newly adjacent one that only a second pass catches.
    """
    assert hendrycks_math.strip_string(r"\\\%%") == reference_hendrycks.strip_string(r"\\\%%")
    assert hendrycks_math.is_equiv(r"\\\%%", r"\\%") == bool(reference_hendrycks.is_equiv(r"\\\%%", r"\\%"))


def test_hendrycks_falls_back_to_raw_equality_when_normalization_raises():
    """A non-integer ``a/b`` makes ``int()`` raise, dropping ``is_equiv`` to raw equality."""
    assert hendrycks_math.is_equiv("x/y", "x/y")
    assert not hendrycks_math.is_equiv("x/y", "x/y ")


def test_graders_need_only_problem_solution_and_reference():
    """No dataset doc, no harness state."""
    assert (
        minerva_math.grade(
            "irrelevant problem text",
            "Final Answer: The final answer is $\\frac{1}{2}$. I hope it is correct.",
            "0.5",
        )
        == 1.0
    )
    assert hendrycks_math.grade("irrelevant problem text", " $0.5$", "\\frac{1}{2}") == 1.0
