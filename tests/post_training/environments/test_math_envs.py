# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from marin.post_training.environments.math_utils import grade_answer


@pytest.mark.skip("Dataset is too large.")
@pytest.mark.parametrize(
    "env_module,env_class,init_args",
    [
        (
            "marin.post_training.environments.olym_math_env",
            "OlymMathEnv",
            {"tokenizer": None, "difficulty": "hard", "language": "en"},
        ),
        ("marin.post_training.environments.open_math_reasoning_env", "OpenMathReasoningEnv", {"tokenizer": None}),
        ("marin.post_training.environments.numina_math_env", "NuminaMathEnv", {"tokenizer": None}),
        ("marin.post_training.environments.aqua_rat_env", "AquaRatEnv", {"tokenizer": None}),
        ("marin.post_training.environments.svamp_env", "SVAMPEnv", {"tokenizer": None}),
        (
            "marin.post_training.environments.olympiad_bench_env",
            "OlympiadBenchEnv",
            {"tokenizer": None, "subject": "maths", "language": "en"},
        ),
        ("marin.post_training.environments.orz_env", "ORZEnv", {"tokenizer": None}),
    ],
)
def test_grade_answer_with_math_envs(env_module, env_class, init_args):
    """Test grade_answer functionality across all math environments."""
    import importlib

    module = importlib.import_module(env_module)
    env_cls = getattr(module, env_class)
    env = env_cls(**init_args)

    # Use appropriate examples based on environment
    examples = getattr(env, "eval_examples", None) or env.train_examples
    example = examples[0]

    # Test that grade_answer returns boolean without crashing
    result = grade_answer(given_answer="test", ground_truth=example["answer"])
    assert isinstance(result, bool)

    # Test self-grading (answer should match itself)
    assert grade_answer(given_answer=example["answer"], ground_truth=example["answer"]) is True


def test_math_env_weak_correct_validation():
    """Test the weak correct validation logic in MathEnv reward computation."""
    import re

    test_cases = [
        # (true_answer, decoded_response, expected_weak_correct)
        ("441", "The answer is 441 pounds.", True),
        ("441", "approximately 441.0022", True),
        ("441", "The result is 4411", False),
        ("441", "We get 1441 as intermediate", False),
        ("441", "441", True),
        ("441", "answer: 441.", True),
        ("23", "The answer is 23 km.", True),
        ("23", "123 is wrong", False),
        ("1/3", "The fraction is 1/3.", True),
        ("1/3", "We get 21/3 which", False),
        ("441 pounds", "The answer is 441 pounds.", True),
        ("441 pounds", "approximately 441", False),
    ]

    for true_answer, decoded_response, expected in test_cases:
        if re.search(rf"\b{re.escape(true_answer)}\b", decoded_response):
            weak_correct = 1.0
        else:
            weak_correct = 0.0

        expected_weak_correct = 1.0 if expected else 0.0
        assert weak_correct == expected_weak_correct, (
            f"Failed for true_answer='{true_answer}', "
            f"response='{decoded_response}': expected {expected_weak_correct}, got {weak_correct}"
        )


def test_latex_to_text():
    """Comprehensive test for LaTeX to text conversion."""
    from marin.post_training.environments.math_utils import latex_to_text

    # Fraction tests
    assert latex_to_text(r"\frac{1}{2}") == "1/2"
    assert latex_to_text(r"\frac{a+b}{c-d}") == "(a+b)/(c-d)"
    assert latex_to_text(r"\tfrac{1}{2}") == "1/2"
    assert latex_to_text(r"\dfrac{a}{b}") == "a/b"

    # Sqrt tests
    assert latex_to_text(r"\sqrt{2}") == "sqrt(2)"
    assert latex_to_text(r"\sqrt{x+y}") == "sqrt(x+y)"
    assert latex_to_text(r"\sqrt 2") == "sqrt(2)"

    # Powers and subscripts
    assert latex_to_text(r"x^2") == "x**2"
    assert latex_to_text(r"x^{2}") == "x**(2)"
    assert latex_to_text(r"x^{n+1}") == "x**(n+1)"
    assert latex_to_text(r"x_{1}") == "x_1"
    assert latex_to_text(r"a_{n}") == "a_n"

    # Greek letters and symbols
    assert latex_to_text(r"\pi") == "pi"
    assert latex_to_text(r"\theta") == "theta"
    assert latex_to_text(r"\alpha + \beta") == "alpha + beta"
    assert latex_to_text(r"\cdot") == "*"
    assert latex_to_text(r"\times") == "*"
    assert latex_to_text(r"\div") == "/"
    assert latex_to_text(r"\infty") == "oo"

    # Functions
    assert latex_to_text(r"\sin(x)") == "sin(x)"
    assert latex_to_text(r"\cos(\theta)") == "cos(theta)"
    assert latex_to_text(r"\log(n)") == "log(n)"

    # Text removal and formatting
    assert latex_to_text(r"\text{hello}") == "hello"
    assert latex_to_text(r"\mathbf{M}") == "M"
    assert latex_to_text(r"\mathcal{H}") == "H"
    assert latex_to_text(r"\left(\frac{1}{2}\right)") == "(1/2)"

    # Advanced features
    assert latex_to_text(r"\dotsb") == "..."
    assert latex_to_text(r"\sum_{n=1}^{\infty}") == "sum"
    assert latex_to_text(r"\int_{0}^{1}") == "integral"
    assert latex_to_text(r"\begin{matrix}1 & 2 \\ 3 & 4\end{matrix}") == "[1 & 2 3 & 4]"

    # Complex expressions
    assert latex_to_text(r"\frac{1}{2} + \sqrt{3}") == "1/2 + sqrt(3)"
    assert latex_to_text(r"2\pi r^2") == "2pi r**2"


def test_latex_parser_edge_cases():
    """Test LaTeX parser robustness with problematic inputs."""
    from marin.post_training.environments.math_utils import latex_to_text

    # Incomplete fractions (should not crash)
    assert latex_to_text(r"\frac{1}") == "1/"
    assert latex_to_text(r"\frac{") == r"\frac{"  # Malformed input preserved
    assert latex_to_text(r"\frac") == r"\frac"  # Malformed input preserved
    assert latex_to_text(r"\frac{1}{2}{3}") == "1/2{3}"

    # Empty and malformed inputs
    assert latex_to_text("") == ""
    # This malformed case is preserved as-is
    malformed_result = latex_to_text(r"\frac{1}{\\")
    assert isinstance(malformed_result, str)  # Should not crash

    # Unicode replacements
    assert latex_to_text("π") == "pi"
    assert latex_to_text("∞") == "oo"

    # ASY code removal
    assert latex_to_text(r"Before [asy]draw((0,0)--(1,1));[/asy] After") == "Before After"


def test_answer_normalization():
    """Test answer normalization functions."""
    from marin.post_training.environments.math_utils import (
        _normalize,
        last_boxed_only_string,
        normalize_answer,
        remove_boxed,
        split_tuple,
    )

    # normalize_answer tests
    assert normalize_answer("42") == "42"
    assert normalize_answer("\\text{hello}") == "hello"
    assert normalize_answer(None) is None
    assert normalize_answer("") is None  # Empty string returns None

    # remove_boxed tests
    assert remove_boxed("\\boxed{42}") == "42"
    assert remove_boxed("\\boxed{x + y}") == "x + y"

    # last_boxed_only_string tests
    assert last_boxed_only_string("Some text \\boxed{42} more text") == "\\boxed{42}"
    assert last_boxed_only_string("\\boxed{1} and \\boxed{2}") == "\\boxed{2}"
    assert last_boxed_only_string("No boxed content") == "No boxed content"

    # split_tuple tests
    assert split_tuple("(1, 2, 3)") == ["1", "2", "3"]
    assert split_tuple("[0, 1]") == ["0", "1"]
    assert split_tuple("42") == ["42"]
    assert split_tuple("") == []
    assert split_tuple("(1,000, 2,000)") == ["1000", "2000"]

    # _normalize tests for malformed expressions
    result = _normalize("192sqrt(14)25")
    assert "*" in result and "/" in result
    assert _normalize("2sqrt(3)") == "2*sqrt(3)"
    assert _normalize("sqrt(2)5") == "sqrt(2)*5"


def test_sympy_compatibility():
    """Test that LaTeX parser produces sympy-compatible expressions."""
    import sympy

    from marin.post_training.environments.math_utils import latex_to_text

    test_cases = [
        r"\frac{1}{2}",
        r"x^2 + y^2",
        r"\sqrt{x}",
        r"\pi * r^2",
        r"\sin(x) + \cos(y)",
    ]

    for latex_expr in test_cases:
        parsed = latex_to_text(latex_expr)
        try:
            sympy_expr = sympy.sympify(parsed)
            assert sympy_expr is not None
        except Exception as e:
            pytest.fail(f"Sympy failed to parse '{parsed}' (from '{latex_expr}'): {e}")


def test_are_equal_under_sympy():
    """Test the are_equal_under_sympy function."""
    from marin.post_training.environments.math_utils import are_equal_under_sympy

    # Equal expressions
    assert are_equal_under_sympy("1/2", "0.5") is True
    assert are_equal_under_sympy("x+1", "1+x") is True

    # Non-equal expressions
    assert are_equal_under_sympy("1/2", "1/3") is False

    # Error cases should return False
    assert are_equal_under_sympy("badexpr", "1") is False


def test_should_allow_eval():
    """Test the should_allow_eval safety check."""
    from marin.post_training.environments.math_utils import should_allow_eval

    # Safe expressions
    assert should_allow_eval("1+2") is True
    assert should_allow_eval("x+y") is True

    # Unsafe expressions
    assert should_allow_eval("a+b+c+d") is False  # Too many variables
    assert should_allow_eval("x^{2}") is False  # Bad substring
    assert should_allow_eval("x^(2)") is False  # Bad substring


def test_count_unknown_letters_in_expr():
    """Test the count_unknown_letters_in_expr function."""
    from marin.post_training.environments.math_utils import count_unknown_letters_in_expr

    assert count_unknown_letters_in_expr("x+y") == 2
    assert count_unknown_letters_in_expr("x+x") == 1  # x appears twice but is one letter
    assert count_unknown_letters_in_expr("123") == 0  # no letters
    assert count_unknown_letters_in_expr("sqrt(x)") == 1  # sqrt is removed, only x counts
    assert count_unknown_letters_in_expr("frac{x}{y}") == 2  # frac is removed, x and y count


def test_latex_parser_integration_with_grade_answer():
    """Test that the LaTeX parser integrates correctly with grade_answer."""
    test_cases = [
        (r"2\sqrt{2}-1", r"2\sqrt{2}-1", True),
        (r"\frac{1}", "1/", None),  # Incomplete but processed - don't expect specific result
    ]

    for given, truth, expected in test_cases:
        try:
            result = grade_answer(given, truth)
            assert isinstance(result, bool)
            if expected is not None:
                assert result == expected, f"Expected {expected} for grade_answer('{given}', '{truth}'), got {result}"
        except Exception as e:
            pytest.fail(f"grade_answer crashed on inputs: given='{given}', truth='{truth}': {e}")
