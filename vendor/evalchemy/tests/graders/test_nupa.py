from importlib import import_module

import pytest

from eval.contracts.eval_spec import EvalSpec, GraderSpec, SourceRef
from eval.graders.math_specs import nupa

scorer = import_module("eval.chat_benchmarks.NUPA-Loose.scorer")


@pytest.mark.parametrize(
    ("prediction", "answer_format", "expected"),
    [
        ("The answer is 9.9, because 9.9 is larger.", scorer.FLOAT, "9.9"),
        ("I think the answer is 9.9", scorer.FLOAT, None),
        ("So the answer is 123", scorer.INTEGER, "123"),
        ("5.04E+04", scorer.SCIENTIFIC, "5.04e04"),
    ],
)
def test_extract_answer_follows_direct_answer_protocol(prediction, answer_format, expected):
    assert scorer.extract_answer(prediction, answer_format) == expected


def test_stored_step92_answers_are_extracted_without_changing_exact_match():
    fraction_output = (
        "First, I need to calculate the result of dividing 56 by 84616. This can be written as the fraction "
        "\\(\\frac{56}{84616}\\).\n\n"
        "Next, I'll simplify this fraction by finding the greatest common divisor (GCD) of 56 and 84616.\n\n"
        "The GCD of 56 and 84616 is 56.\n\n"
        "Now, I'll divide both the numerator and the denominator by 56:\n\n"
        "\\(\\frac{56 \\div 56}{84616 \\div 56} = \\frac{1}{1511}\\)\n\n"
        "The simplified fraction is \\(\\boxed{1/1511}\\). "
    )
    rounded_float_output = (
        "The result is calculated as follows:  \n" "426105174.011715486 + 19.07129229308 = 426105193.08300778 "
    )

    assert scorer.extract_answer(fraction_output, scorer.FRACTION) == "1/1511"
    assert scorer.score_prediction(fraction_output, "1/1511", scorer.FRACTION).exact_match == 1.0
    assert scorer.extract_answer(rounded_float_output, scorer.FLOAT) == "426105193.08300778"
    assert scorer.score_prediction(rounded_float_output, "426105193.08300777908", scorer.FLOAT).exact_match == 0.0


@pytest.mark.parametrize(
    ("prediction", "answer_format", "expected"),
    [
        (r"<|start_think|>Maybe \boxed{9}<|end_think|>\(\boxed{1/1511}\).", scorer.FRACTION, "1/1511"),
        (r"<|start_think|>Maybe 9<|end_think|>\(1/1511\).", scorer.FRACTION, "1/1511"),
        ("<|start_think|>Maybe 9<|end_think|>42", scorer.INTEGER, "42"),
        (r"<|start_think|>\boxed{9}", scorer.INTEGER, None),
        (r"The answer is \boxed{9.90}.", scorer.FLOAT, "9.90"),
        ("So the answer is: 83564549", scorer.INTEGER, "83564549"),
        ("Therefore, the final answer is 2.547762432164e18.", scorer.SCIENTIFIC, "2.547762432164e18"),
        (r"\boxed{1,234}", scorer.INTEGER, None),
        (r"\boxed{9.90}", scorer.INTEGER, None),
        ("Result: 9.9\nI need to check that", scorer.FLOAT, None),
    ],
)
def test_extract_answer_uses_final_content(prediction, answer_format, expected):
    assert scorer.extract_answer(prediction, answer_format) == expected


@pytest.mark.parametrize(
    ("prediction", "answer_format", "expected"),
    [
        ("The calculation ends here.42", scorer.INTEGER, "42"),
        ("The calculation ends here.486.612", scorer.FLOAT, "486.612"),
        ("The calculation ends here.43/20", scorer.FRACTION, "43/20"),
        ("The calculation ends here.5.09786549e52", scorer.SCIENTIFIC, "5.09786549e52"),
    ],
)
def test_extract_answer_accepts_typed_terminal_suffix(prediction, answer_format, expected):
    assert scorer.extract_answer(prediction, answer_format) == expected


@pytest.mark.parametrize(
    ("prediction", "gold", "answer_format"),
    [
        ("Thus the final answer is 1818", "18", scorer.INTEGER),
        ("Thus the final answer is 486.612486.612", "486.612", scorer.FLOAT),
        ("Thus the final answer is 43/2043/20", "43/20", scorer.FRACTION),
        ("Thus the final answer is 5.09786549e525.09786549e52", "5.09786549e52", scorer.SCIENTIFIC),
    ],
)
def test_score_prediction_accepts_exact_repeated_terminal_reference(prediction, gold, answer_format):
    score = scorer.score_prediction(prediction, gold, answer_format)

    assert score.extracted_answer == gold
    assert score.exact_match == 1.0


@pytest.mark.parametrize(
    ("prediction", "gold"),
    [("Thus the final answer is 1818", "19"), ("Thus the final answer is 11818", "18")],
)
def test_score_prediction_does_not_credit_repeated_nonreference_answer(prediction, gold):
    score = scorer.score_prediction(prediction, gold, scorer.INTEGER)

    assert score.exact_match == 0.0


def test_nupa_grade_records_the_answer_used_for_scoring():
    spec = EvalSpec(
        task="NUPA5K-Loose",
        source=SourceRef("fixture"),
        source_id="1",
        ordinal=0,
        requests=(),
        grader=GraderSpec("nupa", {"format": scorer.INTEGER}),
        target="18",
    )

    grade = nupa(spec, ["Thus the final answer is 1818"])

    assert grade.metrics["exact_match"] == 1.0
    assert grade.filtered_resps == ["18"]


def test_score_prediction_preserves_numeric_representation():
    exact = scorer.score_prediction("9.9", "9.9", scorer.FLOAT)
    wrong_digits = scorer.score_prediction("9.11", "9.9", scorer.FLOAT)
    extra_zero = scorer.score_prediction("9.90", "9.9", scorer.FLOAT)
    comma = scorer.score_prediction("1,234", "1234", scorer.INTEGER)

    assert exact.exact_match == 1.0
    assert exact.digit_match == 1.0
    assert exact.dlength == 0.0
    assert wrong_digits.exact_match == 0.0
    assert wrong_digits.digit_match == 0.5
    assert wrong_digits.dlength == 1.0
    assert extra_zero.exact_match == 0.0
    assert comma.exact_match == 0.0


def test_fraction_and_scientific_components_are_scored_independently():
    fraction = scorer.score_prediction("123/456", "12/3456", scorer.FRACTION)
    scientific = scorer.score_prediction("5.04e4", "5.04e4", scorer.SCIENTIFIC)

    assert fraction.dlength == 0.0
    assert scientific.exact_match == 1.0


def test_invalid_format_records_no_answer():
    score = scorer.score_prediction("one half", "1/2", scorer.FRACTION)

    assert score.exact_match == 0.0
    assert score.digit_match == 0.0
    assert score.format_valid == 0.0
    assert score.no_answer == 1.0


@pytest.mark.parametrize(
    ("digit", "max_digit", "expected"),
    [
        (4, 20, "S"),
        (8, 20, "M"),
        (14, 20, "L"),
        (15, 20, "XL"),
        (10, 100, "S"),
        (20, 100, "M"),
        (60, 100, "L"),
        (61, 100, "XL"),
    ],
)
def test_length_bucket_matches_the_paper_intervals(digit, max_digit, expected):
    assert scorer.length_bucket(digit, max_digit=max_digit) == expected
