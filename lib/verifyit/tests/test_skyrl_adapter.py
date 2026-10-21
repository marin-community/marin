# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.adapters.skyrl import (
    grade_aime_candidate,
    grade_aime_extracted,
    grade_grid_candidate,
    grade_gsm8k_final_line,
    grade_gsm8k_strict,
    grade_literal_candidate,
    grade_rounded_candidate,
    grade_search_em,
)
from verifyit.grade import InvalidTask, Status


@pytest.mark.parametrize(
    ("expected", "candidate", "reward"),
    [
        ("Evelyn", "Evelyn", 1.0),
        ("Evelyn", "E*y*n*v*l*e", 0.0),
        ("p-q", "p-q", 1.0),
        ("p-q", "-q+p", 0.0),
        ("2x-8", "2x-8", 1.0),
        ("20:7", "40:14", 1.0),
        ("20:7", "7:20", 0.0),
        ("0.5", r"\frac{1}{2}", 1.0),
        (r"\frac{14}{3}", r"\dfrac{14}{3}", 1.0),
        (r"\frac{3}{\pi}", r"\dfrac{3}{\pi}", 1.0),
        (r"\tfrac{3}{\pi}", r"\frac{3}{\pi}", 1.0),
        (r"\frac{3}{\pi}", r"\dfrac{4}{\pi}", 0.0),
        (r"-\frac{1}{4}", "-1/4", 1.0),
        ("-1/4", r"-\dfrac{1}{4}", 1.0),
        (r"-\frac{-1}{4}", "1/4", 1.0),
        (r"-\frac{1}{4}", "1/4", 0.0),
        ("-1/4", r"--\frac{1}{4}", 0.0),
        (r"-\frac{1}{-4}", "1/4", 1.0),
        (r"-\frac{-1}{-4}", "-1/4", 1.0),
        (r"\frac{3}{\pi}", r"\dfracfoo{3}{\pi}", 0.0),
        ("045", "45", 1.0),
        ("9007199254740992", "9007199254740993", 0.0),
        ("9007199254740992.5", "18014398509481985/2", 1.0),
        ("20:7", "20:0", 0.0),
        ("1000", "1e3", 1.0),
        ("1000", "1_000", 0.0),
        ("0.5", "1.0:2.0", 0.0),
    ],
)
def test_aime_normalized_answer_exact_rational_equivalence(expected, candidate, reward):
    verdict = grade_aime_candidate(expected, candidate)
    assert (verdict.reward, verdict.status) == (reward, Status.SCORED)


@pytest.mark.parametrize(
    ("expected", "response", "reward"),
    [
        ("1234.0", "Reasoning\n#### 1,234", 1.0),
        ("1234", "#### 12,34", 0.0),
        ("42", "#### 42\nMore prose", 0.0),
        ("42", "#### 42\n#### 41", 0.0),
        ("42", "#### 41\n#### 42", 1.0),
        ("9007199254740992", "#### 9007199254740993", 0.0),
        ("0.5", "#### 0.50", 1.0),
        ("42", "The answer is #### 42", 0.0),
    ],
)
def test_gsm8k_requires_exact_value_on_standalone_final_line(expected, response, reward):
    verdict = grade_gsm8k_final_line(expected, response)
    assert (verdict.reward, verdict.status) == (reward, Status.SCORED)


@pytest.mark.parametrize(
    ("expected", "response", "reward"),
    [
        ("42", "#### 42\n#### 41", 1.0),
        ("42", "#### 41\n#### 42", 0.04),
        ("42", "no marker 42", 0.0),
        ("1234", "#### 12,34", 1.0),
        ("42", "#### 42.0", 0.04),
        ("42", "#### .", 0.04),
    ],
)
def test_gsm8k_strict_retains_first_marker_literal_equality_and_turn_shaping(expected, response, reward):
    verdict = grade_gsm8k_strict(expected, response, format_score=0.2 / 5)
    assert (verdict.reward, verdict.status) == (reward, Status.SCORED)


@pytest.mark.parametrize(
    ("targets", "response", "reward"),
    [
        (["New York", "NYC"], "<answer>The NYC!</answer>", 1.0),
        ("cat", "<answer>cat</answer><answer>dog</answer>", 0.0),
        ("cat", "<answer>dog</answer><answer>A cat.</answer>", 1.0),
        ("cat", "cat", 0.0),
        ("cat", "<answer>the catfish</answer>", 0.0),
        ("", "<answer>The!!!</answer>", 1.0),
        ("é", "<answer>É</answer>", 1.0),
        ("ss", "<answer>ß</answer>", 0.0),
        ("", "no answer tag", 0.0),
        ("cat", None, 0.0),
    ],
)
def test_search_qa_normalization_alternatives_and_last_tag(targets, response, reward):
    verdict = grade_search_em(targets, response)
    assert (verdict.reward, verdict.status) == (reward, Status.SCORED)


@pytest.mark.parametrize(("candidate", "reward"), [(" 42", 0.0), ("42 ", 0.0), ("42\n", 0.0), ("42", 1.0)])
def test_literal_encoding_keeps_outer_whitespace_significant(candidate, reward):
    assert grade_literal_candidate("42", candidate).reward == reward


@pytest.mark.parametrize(
    ("expected", "candidate", "reward"),
    [(2.5, 1.5, 1.0), (3.5, 2.5, 0.0), (3.5, 4.0, 1.0), (2.0, None, 0.0), (2.0, float("inf"), 0.0)],
)
def test_rounded_answers_use_bankers_rounding_and_reject_nonfinite_candidates(expected, candidate, reward):
    assert grade_rounded_candidate(expected, candidate).reward == reward


@pytest.mark.parametrize(
    ("candidate", "reward"),
    [
        ([[1, 2], [3, 4]], 1.0),
        ([[1, 2, 3, 4]], 0.0),
        ([[True, 2], [3, 4]], 0.0),
        ([[1.0, 2], [3, 4]], 0.0),
        ([[3, 4], [1, 2]], 0.0),
        ([[1, 2], [3]], 0.0),
        ([[1, 2], [3, 10]], 0.0),
        ([[1, 2], [3, float("nan")]], 0.0),
        (None, 0.0),
    ],
)
def test_grid_comparison_preserves_rows_and_rejects_boolean_and_float_cells(candidate, reward):
    verdict = grade_grid_candidate([[1, 2], [3, 4]], candidate)
    assert verdict.reward == reward


@pytest.mark.parametrize("reference", [[[True]], [[1.0]], [], [[]], [[1], [2, 3]], [[10]]])
def test_invalid_grid_reference_is_not_candidate_zero(reference):
    with pytest.raises(InvalidTask, match="expected grid"):
        grade_grid_candidate(reference, [[1]])


@pytest.mark.parametrize("expected", [".", "nan", "inf"])
def test_gsm8k_malformed_reference_cannot_award_literal_or_format_credit(expected):
    with pytest.raises(InvalidTask, match="finite decimal"):
        grade_gsm8k_strict(expected, f"#### {expected}", format_score=0.04)


@pytest.mark.parametrize(
    "expected",
    [
        None,
        "",
        "nan",
        "inf",
        r"\frac{1}{0}",
        r"x/0",
        r"\frac{x}{x-x}",
        r"\mathrm{NaN}",
        r"\text{Inf}",
        r"\dfracfoo{3}{\pi}",
        "1,2",
    ],
)
@pytest.mark.parametrize("candidate", ["", "42", "same_reference"])
def test_aime_invalid_reference_precedes_candidate_scoring(expected, candidate):
    with pytest.raises(InvalidTask, match="one valid exact answer"):
        grade_aime_candidate(expected, expected if candidate == "same_reference" else candidate)


@pytest.mark.parametrize("candidate", [r"\frac{1}{0}", "nan", "inf", "42,43", "43,42"])
def test_aime_undefined_or_multiple_candidates_cannot_earn_credit(candidate):
    assert grade_aime_candidate("42", candidate).reward == 0.0


def test_deep_grid_candidate_is_schema_zero_and_cannot_mask_invalid_reference():
    candidate = []
    for _ in range(1200):
        candidate = [candidate]
    assert grade_grid_candidate([[1]], candidate).reward == 0.0
    with pytest.raises(InvalidTask, match="expected grid"):
        grade_grid_candidate([[True]], candidate)


@pytest.mark.parametrize("targets", [[], None, ["cat", None]])
@pytest.mark.parametrize("response", ["<answer>cat</answer>", "no answer tag"])
def test_search_invalid_reference_precedes_candidate_format_or_early_match(targets, response):
    with pytest.raises(InvalidTask):
        grade_search_em(targets, response)


@pytest.mark.parametrize("candidate,reward", [("(18,-24)", 1.0), ("(-24,18)", 0.0), ("(18.0,-24)", 0.0)])
def test_aime_flat_tuple_keeps_literal_source_spelling(candidate, reward):
    assert grade_aime_extracted("(18,-24)", candidate).reward == reward


@pytest.mark.parametrize(
    "reference",
    [
        "(18,-24]",
        "(18,-24),",
        "((18,-24),3)",
        "[18,-24]",
        "(18,)",
        "(True,2)",
        "(nan,2)",
        "(1e9999,2)",
        "(1e309,2)",
        "(-1e309,2)",
    ],
)
def test_aime_unsupported_tuple_reference_cannot_match_itself(reference):
    with pytest.raises(InvalidTask):
        grade_aime_extracted(reference, reference)
