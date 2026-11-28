# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("math_verify", reason="math mode needs the `math` extra")

import latex2sympy2_extended
import math_verify
import sympy
from math_verify.errors import TimeoutException
from verifyit.execution.worker import call_bounded
from verifyit.grade import GradingInfraError, InvalidTask, Status, run, write_reward
from verifyit.grade import grade as dispatch
from verifyit.modes import grade_math
from verifyit.modes.grade_exact import grade_exact_candidate
from verifyit.spec import ExactSpec, MathProfile, MathSpec, MathType, render_spec


def _answer(workspace: Path, text: str) -> None:
    (workspace / "answer.txt").write_text(text)


@pytest.mark.parametrize(
    "expected, text, reward",
    [
        ("0.5", "So the area is one half.\n\\boxed{\\frac{1}{2}}\n", 1.0),
        ("0.5", "\\boxed{1/2}", 1.0),
        ("0.5", "\\boxed{0.5}", 1.0),
        ("0.5", "\\boxed{2}", 0.0),
        ("2\\sqrt{3}", "\\boxed{2\\sqrt 3}", 1.0),
        ("2\\sqrt{3}", "\\boxed{3\\sqrt{2}}", 0.0),
        ("x^2+1", "\\boxed{1 + x^2}", 1.0),
        # No boxed expression: the last line is the answer.
        ("42", "First I add the parts.\nThe answer is 42\n", 1.0),
        ("42", "First I add the parts.\nThe answer is 41\n", 0.0),
        # The last box wins, as a model that revises itself boxes twice.
        ("42", "\\boxed{7}\nthat was wrong, actually \\boxed{42}\n", 1.0),
        # A malformed final box cannot expose an earlier answer to the parser.
        ("42", "\\boxed{42}\nthat was wrong, actually \\boxed{\n", 0.0),
        ("42", "\\boxed{42}\nthat was wrong, actually \\boxed{}\n", 0.0),
        ("42", "$\\boxed{42}$", 1.0),
        # An unreadable candidate is a scored wrong answer.
        ("42", "\\boxed{???}", 0.0),
        ("42", "I have no idea how to do this problem.\n", 0.0),
    ],
)
def test_math_scalar_answers_are_compared_symbolically(tmp_path, expected, text, reward):
    _answer(tmp_path, text)
    assert grade_math.grade(MathSpec(expected=expected), tmp_path, tmp_path).reward == reward


@pytest.mark.parametrize("power", [(19, 9), (17, 6), (5, 4)])
def test_math_positive_symbols_accept_equivalent_ratio_powers(power):
    numerator, denominator = power
    exponent = rf"\frac{{{numerator}}}{{{denominator}}}"
    reference = rf"\frac{{a_{{1}} \left(\frac{{a_{{3}}}}{{a_{{2}}}}\right)^{{{exponent}}}}}{{a_{{3}}}}"
    regrouped = rf"\frac{{a_{{1}} a_{{3}}^{{{exponent}}}}}{{a_{{2}}^{{{exponent}}} a_{{3}}}}"
    spec = MathSpec(expected=reference, positive_symbols=("a_1", "a_2", "a_3"))

    assert grade_math.grade_math_candidate(spec, regrouped).reward == 1.0
    wrong_exponent = regrouped.replace(exponent, rf"\frac{{{numerator + 1}}}{{{denominator}}}")
    assert grade_math.grade_math_candidate(spec, wrong_exponent).reward == 0.0
    assert grade_math.grade_math_candidate(MathSpec(expected=reference), regrouped).reward == 0.0


def test_math_positive_symbols_apply_only_to_explicitly_named_symbols():
    assert grade_math.grade_math_candidate(MathSpec(expected=r"\sqrt{x^2}"), "x").reward == 0.0
    assert grade_math.grade_math_candidate(MathSpec(expected=r"\sqrt{x^2}", positive_symbols=("x",)), "x").reward == 1.0
    assert (
        grade_math.grade_math_candidate(MathSpec(expected=r"\sqrt{x^2}+y", positive_symbols=("y",)), "x+y").reward == 0.0
    )


def test_math_positive_symbols_survive_spec_file_grading(tmp_path):
    spec = MathSpec(expected=r"\sqrt{x^2}", positive_symbols=("x",))
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(spec))
    _answer(tmp_path, "x")

    assert run(spec_path, tmp_path).reward == 1.0


@pytest.mark.parametrize(
    "expected, math_type, text, reward",
    [
        ("\\{1,2,3\\}", MathType.SET, "\\boxed{\\{3,2,1\\}}", 1.0),
        ("\\{1,2,3\\}", MathType.SET, "\\boxed{\\{1,2\\}}", 0.0),
        ("(1,3]", MathType.INTERVAL, "\\boxed{(1,3]}", 1.0),
        ("(1,3]", MathType.INTERVAL, "\\boxed{[1,3]}", 0.0),
        # The set/relation comparison the interval and set types enable.
        ("(2,\\infty)", MathType.INTERVAL, "\\boxed{x > 2}", 1.0),
        ("(2,\\infty)", MathType.INTERVAL, "\\boxed{x < 2}", 0.0),
        ("y = 2x + 1", MathType.EQUATION, "\\boxed{y=2x+1}", 1.0),
        ("y = 2x + 1", MathType.EQUATION, "\\boxed{y=2x+2}", 0.0),
        ("(1, 2)", MathType.TUPLE, "\\boxed{(1,2)}", 1.0),
        ("(1, 2)", MathType.TUPLE, "\\boxed{(2,1)}", 0.0),
        ("[1, 2, 3]", MathType.LIST, "\\boxed{[1,2,3]}", 1.0),
        # A list is ordered, and the brackets around it are optional.
        ("[1, 2, 3]", MathType.LIST, "\\boxed{3, 2, 1}", 0.0),
        ("[1, 2, 3]", MathType.LIST, "\\boxed{1, 2, 3}", 1.0),
        ("[1, 2, 3]", MathType.LIST, "\\boxed{\\left[1, 2, 3\\right]}", 1.0),
        ("[1, 2, 3]", MathType.LIST, "\\boxed{[1, 2]}", 0.0),
        ("[1, 2, 3]", MathType.LIST, "\\boxed{[1, 2, 3, 4]}", 0.0),
        # Member comparison uses expression equality.
        ("[1/2, x+1]", MathType.LIST, "\\boxed{[0.5, 1+x]}", 1.0),
        # A comma inside a member does not split it.
        ("[(1,2), 3]", MathType.LIST, "\\boxed{[(1,2), 3]}", 1.0),
        ("[(1,2), 3]", MathType.LIST, "\\boxed{[(2,1), 3]}", 0.0),
    ],
)
def test_math_typed_answers_use_their_comparison(tmp_path, expected, math_type, text, reward):
    _answer(tmp_path, text)
    spec = MathSpec(expected=expected, math_type=math_type)
    assert grade_math.grade(spec, tmp_path, tmp_path).reward == reward


# A HARDMath polynomial_roots answer in the dataset's own form: two \left[…\right] members
# and no outer brackets (row 122 of the train split, issue #11511).
HARDMATH_ROOTS_ANSWER = (
    r"\left[ - \sqrt{\frac{1.00}{\epsilon}}, \  \sqrt{\frac{1.00}{\epsilon}}\right],"
    r" \left[ \frac{1.00}{\epsilon}, \  -1.00\right]"
)


@pytest.mark.parametrize(
    "expected, candidate, reward",
    [
        # The reply in the dataset's form and the bracketed golden control are the same
        # two members: the outer brackets are dropped only when they enclose the whole text.
        (f"[{HARDMATH_ROOTS_ANSWER}]", HARDMATH_ROOTS_ANSWER, 1.0),
        (f"[{HARDMATH_ROOTS_ANSWER}]", f"[{HARDMATH_ROOTS_ANSWER}]", 1.0),
        (f"[{HARDMATH_ROOTS_ANSWER}]", HARDMATH_ROOTS_ANSWER.replace("-1.00", "-2.00"), 0.0),
        # A wrong reply in the dataset's form stays 0.
        (f"[{HARDMATH_ROOTS_ANSWER}]", HARDMATH_ROOTS_ANSWER.replace("- \\sqrt", "\\sqrt"), 0.0),
        # A sequence whose members are parenthesized pairs.
        ("[(1,2),(3,4)]", "(1,2),(3,4)", 1.0),
        ("[(1,2),(3,4)]", "[(1,2),(3,4)]", 1.0),
        ("[(1,2),(3,4)]", "(1,2),(3,5)", 0.0),
        ("[(1,2),(3,4)]", "(1,2),(3,4),(5,6)", 0.0),
    ],
)
def test_list_members_keep_their_own_brackets(expected, candidate, reward):
    spec = MathSpec(expected=expected, math_type=MathType.LIST)
    assert grade_math.grade_math_candidate(spec, candidate).reward == reward


def test_math_empty_output_scores_zero_with_no_output(tmp_path):
    _answer(tmp_path, "\n  \n")
    result = grade_math.grade(MathSpec(expected="42"), tmp_path, tmp_path)
    assert (result.reward, result.detail["reason"]) == (0.0, "no_output")


def test_math_reward_detail_carries_the_extracted_expression(tmp_path):
    _answer(tmp_path, "after simplifying, \\boxed{\\frac{3}{4}}\n")
    detail = grade_math.grade(MathSpec(expected="0.75"), tmp_path, tmp_path).detail
    assert detail["extracted"] == "\\frac{3}{4}"


@pytest.mark.parametrize("expected", ["", "   "])
def test_math_unparsable_expected_is_an_invalid_task(tmp_path, expected):
    _answer(tmp_path, "\\boxed{42}")
    assert dispatch(MathSpec(expected=expected), tmp_path, tmp_path).status == Status.INVALID_TASK


@pytest.mark.parametrize(
    "expected, math_type",
    [
        (r"67.\overline{3}", MathType.SCALAR),
        (r"\begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}.", MathType.SCALAR),
        (r"2007 + \frac\pi 2", MathType.SCALAR),
        (r"\{x \mid 2 < x < 3\}", MathType.SET),
        (r"32 \text{ white}, 16 \text{ black}", MathType.SCALAR),
        (r"32 \text{white}, 16 \text{black}", MathType.SCALAR),
        (r"30 \text{m}/\text{s}+5", MathType.SCALAR),
        (r"\mathrm{x}+32 \text{ white}, 16 \text{ black}", MathType.SCALAR),
        (r"[1, 67.\overline{3}]", MathType.LIST),
        (r"x = \frac\pi 2 = 2", MathType.EQUATION),
        (r"1$ + $2", MathType.SCALAR),
        (r"1$ + ", MathType.SCALAR),
        (r"[1, x = \frac\pi 2 = 2]", MathType.LIST),
        (r"[1, 1$ + $2]", MathType.LIST),
    ],
)
def test_math_partial_reference_is_invalid_before_scoring(expected, math_type):
    with pytest.raises(InvalidTask):
        grade_math.grade_math_candidate(MathSpec(expected=expected, math_type=math_type), "32")


@pytest.mark.parametrize(
    "prose",
    [
        "\\text{No \u2013 the limit need not exist; the matrices can stay bounded\nbut fail to converge.}",
        r"\text{one two three four five}",
    ],
)
@pytest.mark.parametrize("math_type", [MathType.SCALAR, MathType.LIST])
def test_math_prose_only_text_reference_or_list_member_is_invalid_before_scoring(prose, math_type):
    expected = prose if math_type is MathType.SCALAR else f"[1, {prose}]"
    with pytest.raises(InvalidTask):
        grade_math.grade_math_candidate(MathSpec(expected=expected, math_type=math_type), "No")


def test_math_escaped_dollar_reference_stays_parseable():
    reference = r"\$32,\!348"
    assert grade_math.grade_math_candidate(MathSpec(expected=reference), "32,348").reward == 1.0


@pytest.mark.parametrize("answer", ["B", "Evelyn", "Friday", "New York City Hall"])
def test_math_short_text_reference_stays_parseable(answer):
    assert grade_math.grade_math_candidate(MathSpec(expected=rf"\text{{{answer}}}"), answer).reward == 1.0


@pytest.mark.parametrize("math_type", [MathType.SCALAR, MathType.LIST])
@pytest.mark.parametrize("connector", ["and", "or"])
def test_math_text_connector_does_not_invalidate_comma_separated_reference(connector, math_type):
    spec = MathSpec(expected=rf"2, 3, \text{{ {connector} }} 4", math_type=math_type)
    assert grade_math.grade_math_candidate(spec, "2, 3, 4").reward == 1.0


@pytest.mark.parametrize(
    "reference, candidate, math_type",
    [
        (r"15\mbox{ cm}^2", "15", MathType.SCALAR),
        (r"15\mbox{ cm}^{2}", "15", MathType.SCALAR),
        (r"15\mbox{ cm}^{3}", "15", MathType.SCALAR),
        (r"15\mbox{ cm}^{3}", r"15\mbox{ cm}^{3}", MathType.SCALAR),
        (r"15\text{ cm}^3", "15", MathType.SCALAR),
        (r"15\text{ cm}^{4}", "15", MathType.SCALAR),
        (r"[1, 15\mbox{ cm}^{2}]", "[1, 15]", MathType.LIST),
        (r"[1, 15\mbox{ cm}^{3}]", "[1, 15]", MathType.LIST),
        (r"[1, 15\mbox{ cm}^{3}]", r"[1, 15\mbox{ cm}^{3}]", MathType.LIST),
        (r"[1, 15\text{ cm}^3]", "[1, 15]", MathType.LIST),
    ],
)
def test_math_terminal_unit_power_stays_parseable(reference, candidate, math_type):
    spec = MathSpec(expected=reference, math_type=math_type)
    assert grade_math.grade_math_candidate(spec, candidate).reward == 1.0


@pytest.mark.parametrize(
    "reference, candidate, wrong_candidate",
    [
        (r"\frac{4}{3} \mathrm{~m} / \mathrm{s}", "4/3", "7/3"),
        (r"215 \, \text{kJ}/\text{mol}", "215", "216"),
        (r"30 \mathrm{m}/\mathrm{s}", "30", "31"),
        (r"1.8^{\circ} \mathrm{C} / \mathrm{h}", "1.8", "2.8"),
        (r"1000 \, \text{kg} \cdot \text{m}^2/\text{s}", "1000", "1001"),
        (r"11^{\text{th}}", "11", "12"),
        (r"25^{\text{th}}", "25", "26"),
        (r"4^{\text{th}} \text{ minute}", "4", "5"),
        (r"9.9 \, \text{s}^{-1}", "9.9", "10.9"),
        (r"15\mbox{ cm}^{ 2 }", "15", "16"),
        (r"0.33 \, \text{m/s}^2", "0.33", "1.33"),
        (r"1200 \, \text{kg/m}^3", "1200", "1201"),
        (r"\frac{39}{5}\mathrm{~}^{2}", "39/5", "40/5"),
    ],
)
def test_math_reference_unit_chains_ordinals_and_powers_grade_numeric_value(reference, candidate, wrong_candidate):
    spec = MathSpec(expected=reference)
    assert grade_math.grade_math_candidate(spec, candidate).reward == 1.0
    assert grade_math.grade_math_candidate(spec, wrong_candidate).reward == 0.0


@pytest.mark.parametrize(
    "reference, text, reward",
    [
        (r"\textbf{(D)}\ 7", "D", 1.0),
        (r"\textbf{(D)}\ 7", "(D)", 1.0),
        (r"\textbf{(D)}\ 7", r"\boxed{D}", 1.0),
        (r"\textbf{(D)}\ 7", "7", 1.0),
        (r"\textbf{(D)}\ 7", r"\boxed{7}", 1.0),
        (r"\textbf{(D)}\ 7", "The answer is D.", 1.0),
        (r"\textbf{(D)}\ 7", "The answer is 7.", 1.0),
        (r"\textbf{(D)}\ 7", "(D) 7", 1.0),
        (r"\textbf{(D)}\ 7", r"\boxed{\textbf{(D)}\ 7}", 1.0),
        (r"\textbf{(D)}\ 7", "C", 0.0),
        (r"\textbf{(D)}\ 7", "8", 0.0),
        (r"\textbf{(D)}\ 7", "(D) 8", 0.0),
        (r"\textbf{(D)}\ 7", r"\boxed{7D}", 0.0),
        (r"\textbf{(A)}\ 0", r"\boxed{\textbf{(B)}\ 0}", 0.0),
        (r"\textbf{(A)}\ 0", "(B) 0", 0.0),
        (r"\textbf{(A)}\ 0", "B 0", 0.0),
        (r"\textbf{(A)}\ 0", r"\boxed{0}", 1.0),
        (r"\textbf{(C)}\ \frac{1}{2}", r"\boxed{0.5}", 1.0),
        (r"\textbf{(D)}\ \text{Friday}", "Friday", 1.0),
        (r"\textbf{(D)}\ \text{Friday}", r"\boxed{\text{Monday}}", 0.0),
        (r"\textbf{(C)}\ 11^{\text{th}}", r"\boxed{\textbf{(C)}\ 11^{\text{th}}}", 1.0),
        (r"\textbf{(C)}\ 11^{\text{th}}", "11", 1.0),
        (r"\textbf{(C)}\ 11^{\text{th}}", "(C) 12", 0.0),
        (r"\text{(B) }\, \frac{2}{3}", "B", 1.0),
        ("C) 315", r"\boxed{315}", 1.0),
        ("(D) 7", r"\boxed{7}", 1.0),
        (r"\textbf{(D)}\ -7", r"\boxed{-7}", 1.0),
        ("(A) -3", r"\boxed{A}", 1.0),
        ("(A) -3", r"\boxed{-3}", 1.0),
        ("(A) -3", r"\boxed{A - 3}", 0.0),
        ("(A)-3", r"\boxed{(A) -3}", 1.0),
        ("(A) + 2", "A+2", 1.0),
        ("(A) + 2", "A", 0.0),
        (r"\textrm{(A)}\ 7", r"\boxed{A}", 1.0),
        (r"\textbf {(A) } 7", r"\boxed{7}", 1.0),
        (r"\mathbf{(A)}\ 0", r"\boxed{0}", 1.0),
        (r"\mathbf{(A)}\ 0", r"\boxed{\textrm{(B)} 0}", 0.0),
        ("(C)10", r"\boxed{C}", 1.0),
        (r"(\textbf{C}) 10", r"\boxed{10}", 1.0),
        (r"\textbf{(C)}\ \text{I, II, and IV only}", r"\boxed{C}", 1.0),
        (
            r"\textbf{(C)}\ \text{I, II, and IV only}",
            r"\boxed{(C) \text{I, II, and IV only}}",
            1.0,
        ),
        (
            r"\textbf{(C)}\ \text{I, II, and IV only}",
            r"\boxed{(C) \text{I, II, and III only}}",
            0.0,
        ),
        (r"\textbf{(A)} \text{ and } \textbf{(C)}", r"\boxed{\textbf{(A)} \text{ and } \textbf{(C)}}", 1.0),
        (r"\textbf{(A)} \text{ and } \textbf{(C)}", r"\boxed{A}", 0.0),
        (r"B) \( \frac{24}{\sqrt{5}} \).", r"\boxed{B) \( \frac{24}{\sqrt{5}} \).}", 1.0),
        (r"B) \( \frac{24}{\sqrt{5}} \).", r"\boxed{\frac{24}{\sqrt{5}}}", 1.0),
    ],
)
def test_math_labelled_choice_answers(reference, text, reward):
    assert grade_math.grade_math_text_candidate(MathSpec(expected=reference), text).reward == reward


@pytest.mark.parametrize(
    "reference, candidate, math_type",
    [
        (r"\mathrm{x}+1", r"\mathrm{x}+1", MathType.SCALAR),
        (r"2\mathrm{x}+1", r"2\mathrm{x}+1", MathType.SCALAR),
        (r"\frac{1}{2}\mathrm{x}+1", r"\frac{1}{2}\mathrm{x}+1", MathType.SCALAR),
        (r"\frac{\mathrm{x}}{2}", r"\frac{\mathrm{x}}{2}", MathType.SCALAR),
        (r"[1, \frac{\mathrm{x}}{2}]", r"[1, \frac{\mathrm{x}}{2}]", MathType.LIST),
    ],
)
def test_math_upright_symbols_parse_with_following_math(reference, candidate, math_type):
    assert grade_math.grade_math_candidate(MathSpec(expected=reference, math_type=math_type), candidate).reward == 1.0


@pytest.mark.parametrize(
    "reference, candidate, wrong_candidate, math_type",
    [
        (r"2\mathrm{xy}+1", r"2\mathrm{xy}+1", "2", MathType.SCALAR),
        (r"2\text{x}+1", r"2\text{x}+1", "2", MathType.SCALAR),
        (r"[1, 2\mathrm{xy}+1]", r"[1, 2\mathrm{xy}+1]", "[1, 2]", MathType.LIST),
        (r"[1, 2\text{x}+1]", r"[1, 2\text{x}+1]", "[1, 2]", MathType.LIST),
    ],
)
def test_math_adjacent_text_symbols_keep_following_math(reference, candidate, wrong_candidate, math_type):
    spec = MathSpec(expected=reference, math_type=math_type)
    assert grade_math.grade_math_candidate(spec, candidate).reward == 1.0
    assert grade_math.grade_math_candidate(spec, wrong_candidate).reward == 0.0


@pytest.mark.parametrize(
    "reference, candidate, wrong_candidate, math_type",
    [
        (r"1+\mathrm{x}", r"1+\mathrm{x}", "1", MathType.SCALAR),
        (r"1+\text{x}", r"1+\text{x}", "1", MathType.SCALAR),
        (r"[2, 1+\mathrm{x}]", r"[2, 1+\mathrm{x}]", "[2, 1]", MathType.LIST),
    ],
)
def test_math_terminal_text_symbols_after_operators_stay_symbolic(reference, candidate, wrong_candidate, math_type):
    spec = MathSpec(expected=reference, math_type=math_type)
    assert grade_math.grade_math_candidate(spec, candidate).reward == 1.0
    assert grade_math.grade_math_candidate(spec, wrong_candidate).reward == 0.0


@pytest.mark.parametrize(
    "reference,candidate",
    [("18", r"18 \text{ dollars}"), (r"35 \mathrm{~cm}", "35"), (r"\dfrac{1}{2}", r"0.5\text{ m}")],
)
def test_math_candidate_units_match_reference_policy(reference, candidate):
    assert grade_math.grade_math_candidate(MathSpec(expected=reference), candidate).reward == 1.0


def test_math_symbolic_reference_preserves_unit_like_candidate_factor():
    spec = MathSpec(expected="2 s")
    assert grade_math.grade_math_candidate(spec, "2 s").reward == 1.0
    assert grade_math.grade_math_candidate(spec, "2").reward == 0.0


def test_math_list_uses_each_references_unit_policy_for_candidates():
    spec = MathSpec(expected=r"[2 s, 5 \text{ m}]", math_type=MathType.LIST)
    assert grade_math.grade_math_candidate(spec, "[2 s, 5]").reward == 1.0
    assert grade_math.grade_math_candidate(spec, "[2, 5]").reward == 0.0


def test_math_list_candidate_units_ignore_unrelated_reference_normalization():
    spec = MathSpec(expected=r"[\dfrac{1}{2}, 2 s]", math_type=MathType.LIST)
    assert grade_math.grade_math_candidate(spec, r"[0.5\text{ m}, 2 s]").reward == 1.0
    assert grade_math.grade_math_candidate(spec, r"[0.5\text{ m}, 2]").reward == 0.0


def test_math_grades_from_a_worker_thread(tmp_path):
    """Pipelines grade in worker threads, where math-verify's signal-based timeout cannot be armed."""
    _answer(tmp_path, "\\boxed{\\frac{1}{2}}")
    results: list = []
    worker = threading.Thread(
        target=lambda: results.append(grade_math.grade(MathSpec(expected="0.5"), tmp_path, tmp_path))
    )
    worker.start()
    worker.join()
    assert results[0].status == Status.SCORED and results[0].reward == 1.0


@pytest.mark.parametrize(
    "expected,candidate,reward", [("0.5", r"\frac{1}{2}", 1), ("2", "3", 0), ("red", "red", 1), ("2", "???", 0)]
)
def test_boxed_profile_preserves_source_expression_and_text_parsing(expected, candidate, reward):
    result = grade_math.grade_math_candidate(MathSpec(expected, profile=MathProfile.BOXED), candidate)
    assert result.reward == reward


def test_boxed_profile_distinguishes_missing_parse_from_parsed_mismatch():
    spec = MathSpec("2", profile=MathProfile.BOXED)
    assert grade_math.grade_math_candidate(spec, "3").detail.get("reason") != "missing_parse"
    assert grade_math.grade_math_candidate(spec, "").detail["reason"] == "missing_parse"


def test_unknown_direct_math_profile_cannot_award_correct_answer():
    with pytest.raises(InvalidTask, match="unknown"):
        grade_math.grade_math_candidate(MathSpec("2", profile="unknown"), "2")


@pytest.mark.parametrize("profile", ["anchored", "boxed"])
def test_parser_failure_cannot_trigger_fallback_or_positive_reward(monkeypatch, tmp_path, profile):

    def parser_failure(*args, **kwargs):
        raise TimeoutError("parser budget exhausted")

    monkeypatch.setattr(math_verify, "parse", parser_failure)
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(f'mode="math"\nexpected="2"\nprofile="{profile}"\n')
    _answer(tmp_path, "2")
    result = run(spec_path, tmp_path)
    assert (result.status, result.reward) == (Status.INFRA_ERROR, 0.0)


@pytest.mark.parametrize("operation", ["parse", "verify"])
def test_backend_timeout_removes_prior_positive_reward(monkeypatch, tmp_path, operation):
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text('mode="math"\nexpected="2"\n')
    _answer(tmp_path, "2")
    logs = tmp_path / "logs"
    positive = run(spec_path, tmp_path)
    assert (positive.status, positive.reward) == (Status.SCORED, 1.0)
    write_reward(logs, positive)

    def expired(*args, **kwargs):
        raise TimeoutException("backend deadline exhausted")

    monkeypatch.setattr(math_verify, operation, expired)
    result = run(spec_path, tmp_path)
    assert (result.status, result.reward) == (Status.INFRA_ERROR, 0.0)
    assert result.detail["code"] == grade_math.MATH_VERIFIER_DEADLINE
    write_reward(logs, result)
    assert not (logs / "reward.txt").exists()
    assert not (logs / "reward.json").exists()
    assert json.loads((logs / "verdict.json").read_text())["status"] == Status.INFRA_ERROR


def test_math_deadline_verdicts_carry_their_phase(monkeypatch):
    """Reference-parse deadlines are a task property; candidate-parse and comparison
    deadlines follow the reply. The verdict tags which step ran out."""

    spec = MathSpec(expected="y=x+1", math_type=MathType.EQUATION)
    original_reference_parse = latex2sympy2_extended.latex2sympy
    original_parse = math_verify.parse

    def deadline(*args, **kwargs):
        raise TimeoutException("backend deadline exhausted")

    monkeypatch.setattr(latex2sympy2_extended, "latex2sympy", deadline)
    with pytest.raises(GradingInfraError) as reference:
        grade_math.grade_math_candidate(spec, "y=x+2")
    assert reference.value.detail["phase"] == grade_math.MATH_DEADLINE_REFERENCE_PARSE
    monkeypatch.setattr(latex2sympy2_extended, "latex2sympy", original_reference_parse)

    def candidate_deadline(text, **kwargs):
        if "y=x+2" in text:
            raise TimeoutException("candidate deadline")
        return original_parse(text, **kwargs)

    monkeypatch.setattr(math_verify, "parse", candidate_deadline)
    with pytest.raises(GradingInfraError) as candidate:
        grade_math.grade_math_candidate(spec, "y=x+2")
    assert candidate.value.detail["phase"] == grade_math.MATH_DEADLINE_CANDIDATE_PARSE

    monkeypatch.setattr(math_verify, "parse", original_parse)
    monkeypatch.setattr(math_verify, "verify", deadline)
    with pytest.raises(GradingInfraError) as comparison:
        grade_math.grade_math_candidate(spec, "y=x+2")
    assert comparison.value.detail["phase"] == grade_math.MATH_DEADLINE_COMPARISON
    assert all(
        error.value.detail["code"] == grade_math.MATH_VERIFIER_DEADLINE for error in (reference, candidate, comparison)
    )


@pytest.mark.parametrize(
    "expected,candidate,reward",
    [
        ("x^2", "x^2+3", 1.0),
        ("x", "2x", 0.0),
        (r"\sin(x)^2+\cos(x)^2", "4", 1.0),
        (r"\{1,2\}", r"\{3,4\}", 0.0),
        ("[1,2)", "[3,4)", 0.0),
        ("y=x+1", "y=x+2", 0.0),
        (r"\infty", r"-\infty", 0.0),
        (r"\begin{pmatrix}1&2\\3&4\end{pmatrix}", r"\begin{pmatrix}2&3\\4&5\end{pmatrix}", 0.0),
    ],
)
def test_additive_constant_only_matches_finite_scalar_expressions(tmp_path, expected, candidate, reward):
    _answer(tmp_path, candidate)
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(MathSpec(expected=expected, allow_additive_constant=True)))
    result = run(spec_path, tmp_path)
    assert result.status is Status.SCORED
    assert result.reward == reward


def test_additive_constant_policy_preserves_default_exact_comparison(tmp_path):
    _answer(tmp_path, "x^2+3")
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(MathSpec(expected="x^2")))
    assert run(spec_path, tmp_path).reward == 0.0


@pytest.mark.parametrize("nonfinite", ["nan", "zoo", "oo", "-oo"])
def test_nonfinite_difference_cannot_be_an_additive_constant(monkeypatch, tmp_path, nonfinite):
    _answer(tmp_path, "x+1")
    monkeypatch.setattr(sympy, "simplify", lambda expression: sympy.sympify(nonfinite))
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(MathSpec(expected="x", allow_additive_constant=True)))
    result = run(spec_path, tmp_path)
    assert result.status is Status.SCORED
    assert result.reward == 0.0


def test_additive_backend_deadline_is_infrastructure_failure(tmp_path, monkeypatch):
    def deadline(expression):
        raise TimeoutException("additive simplification deadline")

    monkeypatch.setattr(sympy, "simplify", deadline)
    _answer(tmp_path, "x+1")
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(MathSpec(expected="x", allow_additive_constant=True)))
    result = run(spec_path, tmp_path)
    assert result.status is Status.INFRA_ERROR
    assert result.reward == 0.0


def test_additive_option_preserves_exact_collection_comparison(tmp_path):
    _answer(tmp_path, r"\{2,1\}")
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(
        render_spec(MathSpec(expected=r"\{1,2\}", math_type=MathType.SET, allow_additive_constant=True))
    )
    result = run(spec_path, tmp_path)
    assert result.status is Status.SCORED
    assert result.reward == 1.0


@pytest.mark.parametrize(
    "expected,candidate,reward",
    [
        ("x^2", "x^2", 0.0),
        ("x*y", "x y", 0.0),
        ("x", "x", 0.0),
        ("2", "2", 1.0),
        ("x^2", r"\boxed{x^2}", 1.0),
        ("x^2", r"\boxed{x^2+3}", 0.0),
    ],
)
def test_raw_math_profile_preserves_unwrapped_prediction_extraction(tmp_path, expected, candidate, reward):
    _answer(tmp_path, candidate)
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(MathSpec(expected=expected, profile=MathProfile.RAW)))
    result = run(spec_path, tmp_path)
    assert result.status is Status.SCORED
    assert result.reward == reward


@pytest.mark.parametrize("candidate,reward", [("x^2+3", 0.0), (r"\boxed{x^2+3}", 1.0), (r"\boxed{2x^2}", 0.0)])
def test_raw_additive_fallback_requires_latex_extraction(tmp_path, candidate, reward):
    _answer(tmp_path, candidate)
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(MathSpec(expected="x^2", profile=MathProfile.RAW, allow_additive_constant=True)))
    result = run(spec_path, tmp_path)
    assert result.status is Status.SCORED
    assert result.reward == reward


def test_raw_math_unparsed_reference_is_invalid_task(tmp_path):
    _answer(tmp_path, r"\boxed{not a math reference???}")
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(MathSpec(expected="not a math reference???", profile=MathProfile.RAW)))
    result = run(spec_path, tmp_path)
    assert (result.status, result.reward) == (Status.INVALID_TASK, 0.0)


@pytest.mark.parametrize("candidate", [r"\boxed{2} then \boxed{", r"\boxed{2} then \boxed{}"])
def test_raw_math_does_not_recover_an_earlier_answer_after_malformed_final_box(tmp_path, candidate):
    _answer(tmp_path, candidate)
    spec_path = tmp_path / "verifier.toml"
    spec_path.write_text(render_spec(MathSpec(expected="2", profile=MathProfile.RAW)))
    result = run(spec_path, tmp_path)
    assert (result.status, result.reward) == (Status.SCORED, 0.0)


@pytest.mark.parametrize(
    "gold,candidate,expected",
    [
        ("1,1,2", "2,1,1", 1),
        ("1,1,2", "1,2,2", 0),
        ("1,1,2", "1,2", 0),
        ("1,2", "1,2,3", 0),
        (r"\frac{1}{2}", "0.5", 1),
        ("0", "0.00000000005", 0),
        ("0", "0.0000000001", 0),
        ("1000000000000000000000000000001", "1000000000000000000000000000002", 0),
        ("1000000000000000000000000000000.1", "1000000000000000000000000000000.2", 0),
    ],
)
def test_canonical_members_preserve_exact_multiset_and_numeric_distinctions(gold, candidate, expected):
    reference = call_bounded(grade_math.canonical_math_members, gold, timeout=10)
    proposed = call_bounded(grade_math.canonical_math_members, candidate, timeout=10)
    verdict = grade_exact_candidate(ExactSpec(expected=reference, ordered=False), ",".join(proposed))
    assert verdict.reward == expected


@pytest.mark.parametrize("value", ["", "[]", "1,,2", "x", r"\infty", "0.1+0.2", "1 +", "1 trailing text", r"\frac{1}"])
def test_canonical_members_reject_empty_nonfinite_and_approximate_expressions(value):
    with pytest.raises(ValueError):
        call_bounded(grade_math.canonical_math_members, value, timeout=10)


@pytest.mark.parametrize("expected", ["", "   ", r"\displaystyle"])
@pytest.mark.parametrize("candidate", ["2", "1/0"])
def test_boxed_empty_reference_parse_is_invalid_before_candidate(expected, candidate):
    with pytest.raises(InvalidTask, match="boxed reference"):
        grade_math.grade_math_candidate(MathSpec(expected, profile=MathProfile.BOXED), candidate)


def test_boxed_preserves_physics_reference_raw_fallback():
    expected = r"n=\frac{e}{\hbar} \sqrt{\frac{m_{\mathrm{e}} \lambda}{4 \pi \varepsilon_{0}}}"
    spec = MathSpec(expected, profile=MathProfile.BOXED)
    assert grade_math.grade_math_candidate(spec, expected).reward == 1
    missing = grade_math.grade_math_candidate(spec, "")
    assert missing.status == Status.SCORED
    assert missing.reward == 0


@pytest.mark.parametrize(
    ("expected", "candidate", "reward"),
    [
        # math-verify raises on chains whose deduplicated relations differ in number
        # (huggingface/Math-Verify#93); chains compare as sets of constraints.
        ("m < -n < n < -m", "m < -n < n < -m", 1),
        ("m < -n < n < -m", "m < -n < n", 1),  # n < -m restates m < -n
        ("m < -n < n < -m", "m < -n < n < -mX'", 0),
        ("1<x<3", r"\frac{2}{2} < x < 3", 1),
        ("1<x<3", "3 > x > 1", 1),
        ("1<x<3", r"1 < x \le 3", 0),
        ("1<x<3", "1<x<3<5", 0),
        ("1<x<3", "x<3", 0),
        # Unit stripping deleted a chain's spaced trailing letter (s as seconds).
        ("m < -n < n < -m", "-m > n > -n > m", 1),
        ("p < q < r < s", "p<q<r<s", 1),
        ("p < q < r < s", "p < q < r", 0),
        ("x - s", "x-s", 1),
        ("x - s", "x", 0),
        (r"5 \text{ m}", "5", 1),  # a real unit is still stripped
        # math-verify cannot compare Unequality; a != b compares as a = b.
        (r"x \neq 3", r"x \ne 3", 1),
        (r"x \neq 3", r"x \neq 4", 0),
        (r"x \neq 3", "x = 3", 0),
    ],
)
def test_answers_math_verify_cannot_compare_grade_by_their_meaning(expected, candidate, reward):
    spec = MathSpec(expected=expected, profile=MathProfile.ANCHORED)
    assert grade_math.grade_math_candidate(spec, candidate).reward == reward


def test_an_undefined_reference_is_an_invalid_task():
    with pytest.raises(InvalidTask, match="undefined"):
        grade_math.grade_math_candidate(MathSpec(expected=r"\frac{1}{0}"), "1")


def test_interval_comparison_keeps_a_relation_chain_whole():
    spec = MathSpec(expected="(1,3)", math_type=MathType.INTERVAL, profile=MathProfile.ANCHORED)

    assert grade_math.grade_math_candidate(spec, "1<x<3").reward == 1


@pytest.mark.parametrize("profile", [MathProfile.ANCHORED, MathProfile.BOXED])
def test_a_candidate_sympy_cannot_build_scores_zero(profile):
    huge = "8" * 70_000
    result = grade_math.grade_math_text_candidate(MathSpec("12", profile=profile), rf"\boxed{{{huge}}}")
    assert result.reward == 0
