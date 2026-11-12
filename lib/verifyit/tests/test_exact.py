# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from verifyit.grade import InvalidTask, Status
from verifyit.grade import grade as dispatch
from verifyit.modes import grade_exact
from verifyit.spec import EmptyOutputPolicy, ExactSpec


def _answer(workspace: Path, text: str) -> None:
    (workspace / "answer.txt").write_text(text)


@pytest.mark.parametrize(
    "spec, text, reward",
    [
        (ExactSpec(expected=("Paris",)), "  Paris \n", 1.0),
        (ExactSpec(expected=("Paris",)), "paris\n", 1.0),
        (ExactSpec(expected=("Paris",), ignore_case=False), "paris\n", 0.0),
        (ExactSpec(expected=("Paris",)), "Lyon\n", 0.0),
        (ExactSpec(expected=("hello world",)), "hello    world\n", 1.0),
        (ExactSpec(expected=("hello world",), ignore_whitespace=False), "hello    world\n", 0.0),
        # A trailing newline never decides the reward, even with whitespace significant.
        (ExactSpec(expected=("hello world",), ignore_whitespace=False), "hello world\n", 1.0),
        # A boxed answer is compared on its own, so surrounding prose does not spoil it.
        (ExactSpec(expected=("Paris",)), "The capital is \\boxed{Paris} of course.\n", 1.0),
        (ExactSpec(expected=("Paris",)), "The capital is \\boxed{Lyon} of course.\n", 0.0),
        (ExactSpec(expected=("Paris",)), "The capital is Paris of course.\n", 0.0),
    ],
)
def test_exact_single_expected_compares_the_whole_candidate(tmp_path, spec, text, reward):
    _answer(tmp_path, text)
    assert grade_exact.grade(spec, tmp_path, tmp_path).reward == reward


@pytest.mark.parametrize(
    "spec, text, reward",
    [
        (ExactSpec(expected=("red", "green", "blue")), "red, green, blue\n", 1.0),
        (ExactSpec(expected=("red", "green", "blue")), "red\ngreen\nblue\n", 1.0),
        (ExactSpec(expected=("red", "green", "blue")), "blue, green, red\n", 0.0),
        (ExactSpec(expected=("red", "green", "blue"), ordered=False), "blue, green, red\n", 1.0),
        (ExactSpec(expected=("red", "green", "blue")), "red, green\n", 0.0),
        (ExactSpec(expected=("red", "green", "blue")), "red, green, blue, yellow\n", 0.0),
        # A multiset counts repeats: two "a" are not the same answer as one.
        (ExactSpec(expected=("a", "a", "b"), ordered=False), "a, b, a\n", 1.0),
        (ExactSpec(expected=("a", "a", "b"), ordered=False), "a, b, b\n", 0.0),
        (ExactSpec(expected=("red", "green")), "\\boxed{red, green}\n", 1.0),
    ],
)
def test_exact_several_expected_compares_the_candidate_as_a_list(tmp_path, spec, text, reward):
    _answer(tmp_path, text)
    assert grade_exact.grade(spec, tmp_path, tmp_path).reward == reward


def test_exact_empty_output_scores_zero_with_no_output(tmp_path):
    _answer(tmp_path, "\n \n")
    result = grade_exact.grade(ExactSpec(expected=("Paris",)), tmp_path, tmp_path)
    assert (result.status, result.reward, result.detail["reason"]) == (Status.SCORED, 0.0, "no_output")


def test_exact_reward_detail_carries_the_extracted_candidate(tmp_path):
    _answer(tmp_path, "The capital is \\boxed{Lyon}.\n")
    detail = grade_exact.grade(ExactSpec(expected=("Paris",)), tmp_path, tmp_path).detail
    assert detail == {"extracted": "Lyon", "expected": ["Paris"]}


def test_exact_without_an_expected_string_is_an_invalid_task(tmp_path):
    _answer(tmp_path, "Paris\n")
    assert dispatch(ExactSpec(expected=()), tmp_path, tmp_path).status == Status.INVALID_TASK


@pytest.mark.parametrize(
    "candidate, expected, reward", [("42 ", "42", 0), (" 42", "42", 0), ("42\n", "42", 0), (" 42", " 42", 1)]
)
def test_exact_literal_boundary_preserves_whitespace(candidate, expected, reward):
    spec = ExactSpec((expected,), ignore_case=False, ignore_whitespace=False, strip_outer_whitespace=False)
    assert grade_exact.grade_exact_candidate(spec, candidate).reward == reward


def test_invalid_direct_normalization_flag_cannot_award_correct_answer():
    with pytest.raises(InvalidTask, match="booleans"):
        grade_exact.grade_exact_candidate(ExactSpec(("2",), ignore_case="false"), "2")


@pytest.mark.parametrize(
    "candidate,score",
    [("The capital is Paris.", 1.0), ("Parisian", 1.0), ("Lyon", 0.0), ("", 0.0)],
)
def test_explicit_substring_contract_grades_containment(candidate, score):
    spec = ExactSpec(("Paris",), substring=True)
    assert grade_exact.grade_exact_candidate(spec, candidate).reward == score


@pytest.mark.parametrize("expected", [("",), ("  \n",), ("Paris", "Lyon")])
def test_substring_vacuous_or_multi_reference_task_invalid_before_missing_output(tmp_path, expected):
    spec = ExactSpec(expected, substring=True)
    with pytest.raises(InvalidTask, match="one nonempty"):
        grade_exact.grade(spec, tmp_path, tmp_path)
    result = dispatch(spec, tmp_path, tmp_path)
    assert result.status == Status.INVALID_TASK
    assert result.reward == 0.0


def test_substring_boolean_flag_is_strict():
    with pytest.raises(InvalidTask, match="booleans"):
        grade_exact.grade_exact_candidate(ExactSpec(("Paris",), substring=1), "Paris")


def test_source_lower_semantics_are_separate_from_casefold():
    spec = ExactSpec(("ß".lower(),), ignore_case=False, ignore_whitespace=False, substring=True)
    assert grade_exact.grade_exact_candidate(spec, "SS".lower()).reward == 0.0
    assert grade_exact.grade_exact_candidate(spec, "Straße".lower()).reward == 1.0


@pytest.mark.parametrize("expected", ["idk", b"idk", {"i": 1, "d": 1, "k": 1}])
def test_direct_reference_container_cannot_award_character_list_credit(tmp_path, expected):
    spec = ExactSpec(expected=expected)
    with pytest.raises(InvalidTask, match="expected string"):
        grade_exact.grade_exact_candidate(spec, "i,d,k")
    _answer(tmp_path, "i,d,k")
    result = dispatch(spec, tmp_path, tmp_path)
    assert result.status == Status.INVALID_TASK
    assert result.reward == 0.0


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_explicit_empty_equality_requires_grade_policy_and_a_present_file(tmp_path, text):
    zero = ExactSpec(expected=("",))
    configured = ExactSpec(expected=("",), empty_output=EmptyOutputPolicy.GRADE)
    assert grade_exact.grade_exact_candidate(zero, text).reward == 0
    assert grade_exact.grade_exact_candidate(configured, text).reward == 1
    _answer(tmp_path, text)
    assert grade_exact.grade(zero, tmp_path, tmp_path).reward == 0
    assert grade_exact.grade(configured, tmp_path, tmp_path).reward == 1
    (tmp_path / "answer.txt").unlink()
    assert grade_exact.grade(configured, tmp_path, tmp_path).reward == 0
    with pytest.raises(InvalidTask):
        grade_exact.grade_exact_candidate(ExactSpec(expected=(), empty_output=EmptyOutputPolicy.GRADE), text)


@pytest.mark.parametrize("multiplicity,reward", [("set", 1.0), ("multiset", 0.8)])
def test_prepared_collection_f1_controls_duplicate_credit(multiplicity, reward):
    verdict = grade_exact.grade_collection_f1(
        ["fox", "red"],
        ["fox", "fox", "red"],
        multiplicity=multiplicity,
        empty_reference="invalid",
        round_digits=None,
    )
    assert verdict.status is Status.SCORED
    assert verdict.reward == reward


@pytest.mark.parametrize("overlap,expected", [(1, 0.0), (29, 0.14), (109, 0.55), (125, 0.62)])
def test_collection_decimal_rounding_uses_scaled_ties_to_even(overlap, expected):
    reference = [str(i) for i in range(200)]
    candidate = reference[:overlap] + [f"other-{i}" for i in range(200 - overlap)]
    assert (
        grade_exact.grade_collection_f1(
            reference, candidate, multiplicity="multiset", empty_reference="invalid", round_digits=2
        ).reward
        == expected
    )


def test_collection_reference_policy_is_validated_before_empty_candidate():
    options = {"multiplicity": "multiset", "round_digits": None}
    assert grade_exact.grade_collection_f1([], [], empty_reference="zero", **options).reward == 0
    with pytest.raises(InvalidTask, match="reference must not be empty"):
        grade_exact.grade_collection_f1([], [], empty_reference="invalid", **options)
    with pytest.raises(InvalidTask):
        grade_exact.grade_collection_f1([float("nan")], [], empty_reference="zero", **options)
    assert grade_exact.grade_collection_f1(["one"], [float("nan")], empty_reference="invalid", **options).reward == 0


def test_collection_limits_cannot_award_identical_oversized_inputs():
    reference = ["one"]
    candidate = ["one"] * (grade_exact.MAX_COLLECTION_ITEMS + 1)
    options = {"multiplicity": "set", "empty_reference": "invalid", "round_digits": None}
    assert grade_exact.grade_collection_f1(reference, candidate, **options).reward == 0
    with pytest.raises(InvalidTask, match="too many items"):
        grade_exact.grade_collection_f1(candidate, candidate, **options)
    with pytest.raises(InvalidTask, match="rounding"):
        grade_exact.grade_collection_f1(reference, reference, **{**options, "round_digits": float("inf")})


def test_precision_zero_interval_distinguishes_disjoint_from_invalid_candidates():
    options = {"multiplicity": "set", "empty_reference": "zero", "minimum_percent": -2, "maximum_percent": 2}
    for reference in (["abc"], []):
        assert grade_exact.grade_collection_precision_interval(reference, ["xyz"], **options).reward == 1
        for candidate in ([], [float("nan")], ["xyz"] * (grade_exact.MAX_COLLECTION_ITEMS + 1)):
            result = grade_exact.grade_collection_precision_interval(reference, candidate, **options)
            assert result.status is Status.SCORED
            assert result.reward == 0
    with pytest.raises(InvalidTask):
        grade_exact.grade_collection_precision_interval([None], [], **options)
    with pytest.raises(InvalidTask):
        grade_exact.grade_collection_precision_interval([], [], **{**options, "empty_reference": "invalid"})
    with pytest.raises(InvalidTask):
        grade_exact.grade_collection_precision_interval(["abc"], ["xyz"], **{**options, "maximum_percent": float("nan")})


@pytest.mark.parametrize(
    "multiplicity, minimum, maximum, expected", [("set", 49, 51, 1), ("multiset", 49, 51, 0), ("multiset", 32, 34, 1)]
)
def test_precision_interval_counts_candidate_occurrences(multiplicity, minimum, maximum, expected):
    result = grade_exact.grade_collection_precision_interval(
        ["abc"],
        ["abc", "abc", "xyz"],
        multiplicity=multiplicity,
        empty_reference="invalid",
        minimum_percent=minimum,
        maximum_percent=maximum,
    )
    assert result.reward == expected


def test_precision_interval_preserves_percent_scaling_boundary():
    reference = [str(i) for i in range(29)]
    candidate = [str(i) for i in range(50)]
    options = {"multiplicity": "set", "empty_reference": "invalid"}
    # The source contract scales the quotient: 29 / 50 * 100 is just below 58.
    assert (
        grade_exact.grade_collection_precision_interval(
            reference, candidate, minimum_percent=58, maximum_percent=62, **options
        ).reward
        == 0
    )
    assert (
        grade_exact.grade_collection_precision_interval(
            reference, candidate, minimum_percent=56, maximum_percent=60, **options
        ).reward
        == 1
    )


def test_precision_finite_integer_bound_does_not_overflow_float_conversion():
    options = {"minimum_percent": 0, "maximum_percent": 10**400, "multiplicity": "set", "empty_reference": "invalid"}
    assert grade_exact.grade_collection_precision_interval(["a"], ["a"], **options).reward == 1
    assert grade_exact.grade_collection_precision_interval(["a"], [], **options).reward == 0


@pytest.mark.parametrize("candidate,reward", [("DAC", 1), ("AC", 0.5), ("CC", 0.25), ("AB", 0), ("", 0)])
def test_collection_subset_preserves_complete_partial_and_extra_item_rewards(candidate, reward):
    result = grade_exact.grade_collection_subset(tuple("ACD"), tuple(candidate), item_credit=0.25)
    assert result.status == Status.SCORED
    assert result.reward == reward


def test_collection_subset_invalid_reference_precedes_empty_candidate():
    with pytest.raises(InvalidTask):
        grade_exact.grade_collection_subset(("",), (), item_credit=0.25)
    with pytest.raises(InvalidTask):
        grade_exact.grade_collection_subset(tuple("ABCDE"), (), item_credit=0.5)
    with pytest.raises(InvalidTask):
        grade_exact.grade_collection_subset(("A", "B"), (), item_credit=10**400)
