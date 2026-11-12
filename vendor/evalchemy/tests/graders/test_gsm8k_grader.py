"""Parity tests for the self-contained GSM8K grader in eval/graders.

lm-eval is a base dependency here, so every case is checked against the pinned
harness live rather than against recorded verdicts: the reference ``RegexFilter``
and ``exact_match_hf_evaluate`` are imported and run side by side with the port.

``data/gsm8k_grader_benchmark.jsonl`` holds 100 GSM8K test problems, each with a
synthesized completion in the format the 5-shot prompt elicits -- reasoning
closing with a ``#### <number>`` line -- mixing verbatim, equivalently
reformatted, wrong, and malformed answers.

The single-case tests cover behavior a plausible rewrite gets wrong, each
observed against the reference rather than invented.
"""

import json
import pathlib
import sys

import pytest
from lm_eval.api.metrics import exact_match_hf_evaluate
from lm_eval.filters.extraction import RegexFilter

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval.graders import gsm8k  # noqa: E402

BENCHMARK = pathlib.Path(__file__).parent / "data" / "gsm8k_grader_benchmark.jsonl"

# Mirrors the filter_list and metric_list of lm_eval/tasks/gsm8k/gsm8k.yaml.
REGEXES_TO_IGNORE = [",", "\\$", "(?s).*#### ", "\\.$"]
REFERENCE_STRICT = RegexFilter(regex_pattern=r"#### (\-?[0-9\.\,]+)", group_select=0)
REFERENCE_FLEXIBLE = RegexFilter(regex_pattern=r"(-?[$0-9.,]{2,})|(-?[0-9]+)", group_select=-1)


@pytest.fixture(scope="module")
def benchmark():
    records = [json.loads(line) for line in BENCHMARK.read_text().splitlines()]
    assert len(records) == 100
    return records


def reference_grade(regex_filter: RegexFilter, solution: str, reference_answer: str) -> float:
    """The harness filter chain and exact_match metric, inlined."""
    prediction = regex_filter.apply([[solution]], [{}])[0][0]
    result = exact_match_hf_evaluate(
        predictions=[prediction],
        references=[reference_answer],
        regexes_to_ignore=REGEXES_TO_IGNORE,
        ignore_case=True,
        ignore_punctuation=False,
    )
    return float(result["exact_match"])


def test_strict_match_matches_harness_on_benchmark(benchmark):
    mismatched = [
        record["reference_answer"]
        for record in benchmark
        if gsm8k.grade(record["problem"], record["solution"], record["reference_answer"])
        != reference_grade(REFERENCE_STRICT, record["solution"], record["reference_answer"])
    ]
    assert mismatched == []


def test_flexible_extract_matches_harness_on_benchmark(benchmark):
    mismatched = [
        record["reference_answer"]
        for record in benchmark
        if gsm8k.grade_flexible_extract(record["problem"], record["solution"], record["reference_answer"])
        != reference_grade(REFERENCE_FLEXIBLE, record["solution"], record["reference_answer"])
    ]
    assert mismatched == []


def test_extraction_matches_harness_on_benchmark(benchmark):
    """Both filters are reproduced exactly, not just their downstream verdict."""
    for record in benchmark:
        solution = record["solution"]
        assert gsm8k.extract_strict_match(solution) == REFERENCE_STRICT.apply([[solution]], [{}])[0][0]
        assert gsm8k.extract_flexible(solution) == REFERENCE_FLEXIBLE.apply([[solution]], [{}])[0][0]


def test_strict_match_falls_back_when_marker_absent():
    """The reference emits a sentinel rather than guessing, so the grade is 0."""
    assert gsm8k.extract_strict_match("The answer is 18.") == gsm8k.FALLBACK


def test_strict_match_rejects_a_currency_prefixed_answer():
    """``#### (\\-?[0-9\\.\\,]+)`` cannot start on ``$``, so the whole line is missed."""
    assert gsm8k.extract_strict_match("#### $18") == gsm8k.FALLBACK
    assert gsm8k.grade("", "#### $18", "#### 18") == 0.0


def test_flexible_extract_takes_the_last_number():
    """group_select is -1, so a trailing restatement wins over earlier arithmetic."""
    assert gsm8k.extract_flexible("3 + 2 = 5, so he has 7 left.") == "7"


def test_flexible_extract_ignores_numbers_in_a_later_question():
    solution = "The answer is 12.\n[Question] Configure an ETL job for 8088 rows."

    assert gsm8k.extract_flexible(solution) == "12."


def test_flexible_extract_recovers_an_answer_strict_match_misses():
    assert gsm8k.grade("", "The answer is 18", "#### 18") == 0.0
    assert gsm8k.grade_flexible_extract("", "The answer is 18", "#### 18") == 1.0


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        # regexes_to_ignore drops commas, dollar signs, and one trailing period.
        ("1,234", "#### 1234"),
        ("$18", "#### 18"),
        ("18.", "#### 18"),
        # Only the text after the last "#### " marker survives.
        ("18", "reasoning\n#### 18"),
        ("18", "18"),
        ("18", "#### 19"),
        # The period rule is anchored, so an interior one stays.
        ("18.5", "#### 18.5"),
        ("18.5", "#### 185"),
        # "$" also matches before a trailing newline, so rstrip(".") would differ.
        ("18.\n", "#### 18"),
    ],
)
def test_exact_match_matches_harness_normalization(prediction, reference):
    expected = exact_match_hf_evaluate(
        predictions=[prediction],
        references=[reference],
        regexes_to_ignore=REGEXES_TO_IGNORE,
        ignore_case=True,
        ignore_punctuation=False,
    )
    assert gsm8k.exact_match(prediction, reference) is bool(expected["exact_match"])


def test_grader_needs_only_problem_solution_and_reference():
    """No dataset doc, no harness state."""
    assert gsm8k.grade("irrelevant problem text", "reasoning\n#### 42", "#### 42") == 1.0
    assert gsm8k.grade("irrelevant problem text", "reasoning\n#### 42", "42") == 1.0
