# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import threading
from pathlib import Path

import pytest

pytest.importorskip("math_verify", reason="math mode needs the `answer` extra")

import math_verify
import sympy
from math_verify.errors import TimeoutException
from tasktrove_verify.execution.worker import call_bounded
from tasktrove_verify.grade import InvalidTask, Status, run, write_reward
from tasktrove_verify.grade import grade as dispatch
from tasktrove_verify.modes import grade_math
from tasktrove_verify.modes.grade_exact import grade_exact_candidate
from tasktrove_verify.spec import ExactSpec, MathProfile, MathSpec, MathType, render_spec


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
    write_reward(logs, result)
    assert not (logs / "reward.txt").exists()
    assert not (logs / "reward.json").exists()
    assert '"status": "infra_error"' in (logs / "verdict.json").read_text()


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
