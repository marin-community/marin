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
def test_grade_answer_with_olym_math_env():
    """
    Test whether `grade_answer` works correctly with OlymMathEnv
    by ensuring basic functionality without relying on specific answers.
    """
    from marin.post_training.environments.olym_math_env import OlymMathEnv

    hard_olymp_math_env = OlymMathEnv(tokenizer=None, difficulty="hard", language="en")

    # Test that grade_answer works without crashing and returns boolean
    example = hard_olymp_math_env.eval_examples[0]
    result = grade_answer(given_answer="test", ground_truth=example["answer"])
    assert isinstance(result, bool)

    # Test self-grading (answer should match itself)
    assert grade_answer(given_answer=example["answer"], ground_truth=example["answer"]) is True


@pytest.mark.skip("Dataset is too large.")
def test_grade_answer_with_open_math_reasoning_env():
    """Test basic grade_answer functionality with OpenMathReasoningEnv."""
    from marin.post_training.environments.open_math_reasoning_env import OpenMathReasoningEnv

    env = OpenMathReasoningEnv(tokenizer=None)

    # Test basic functionality without specific answer checking
    example = env.train_examples[0]
    result = grade_answer(given_answer="test", ground_truth=example["answer"])
    assert isinstance(result, bool)

    # Test self-grading
    assert grade_answer(given_answer=example["answer"], ground_truth=example["answer"]) is True


@pytest.mark.skip("Dataset is too large.")
def test_grade_answer_with_numina_math_env():
    """Test basic grade_answer functionality with NuminaMathEnv."""
    from marin.post_training.environments.numina_math_env import NuminaMathEnv

    env = NuminaMathEnv(tokenizer=None)

    example = env.train_examples[0]
    result = grade_answer(given_answer="test", ground_truth=example["answer"])
    assert isinstance(result, bool)
    assert grade_answer(given_answer=example["answer"], ground_truth=example["answer"]) is True


@pytest.mark.skip("Dataset is too large.")
def test_grade_answer_with_aqua_rat_env():
    """Test basic grade_answer functionality with AquaRatEnv."""
    from marin.post_training.environments.aqua_rat_env import AquaRatEnv

    env = AquaRatEnv(tokenizer=None)

    example = env.train_examples[0]
    result = grade_answer(given_answer="test", ground_truth=example["answer"])
    assert isinstance(result, bool)
    assert grade_answer(given_answer=example["answer"], ground_truth=example["answer"]) is True


@pytest.mark.skip("Dataset is too large.")
def test_grade_answer_with_svamp_env():
    """Test basic grade_answer functionality with SVAMPEnv."""
    from marin.post_training.environments.svamp_env import SVAMPEnv

    env = SVAMPEnv(tokenizer=None)

    example = env.train_examples[0]
    result = grade_answer(given_answer="test", ground_truth=example["answer"])
    assert isinstance(result, bool)
    assert grade_answer(given_answer=example["answer"], ground_truth=example["answer"]) is True


@pytest.mark.skip("Dataset is too large.")
def test_grade_answer_with_olympiad_bench_env():
    """Test basic grade_answer functionality with OlympiadBenchEnv."""
    from marin.post_training.environments.olympiad_bench_env import OlympiadBenchEnv

    env = OlympiadBenchEnv(tokenizer=None, subject="maths", language="en")

    example = env.train_examples[0]
    result = grade_answer(given_answer="test", ground_truth=example["answer"])
    assert isinstance(result, bool)
    assert grade_answer(given_answer=example["answer"], ground_truth=example["answer"]) is True


def test_math_env_weak_correct_validation():
    """Test the weak correct validation logic in MathEnv reward computation."""
    import re

    # Test cases for weak correct validation
    test_cases = [
        # (true_answer, decoded_response, expected_weak_correct)
        ("441", "The answer is 441 pounds.", True),  # Should match with word boundary
        ("441", "approximately 441.0022", True),  # Word boundary matches here (441 is at start of 441.0022)
        ("441", "The result is 4411", False),  # Should not match as part of larger number
        ("441", "We get 1441 as intermediate", False),  # Should not match as part of larger number
        ("441", "441", True),  # Exact match should work
        ("441", "answer: 441.", True),  # Should match with punctuation
        ("23", "The answer is 23 km.", True),  # Should match with units
        ("23", "123 is wrong", False),  # Should not match as part of larger number
        ("1/3", "The fraction is 1/3.", True),  # Should work with fractions
        ("1/3", "We get 21/3 which", False),  # Should not match as part of larger fraction
        ("441 pounds", "The answer is 441 pounds.", True),  # Full answer with units should match
        ("441 pounds", "approximately 441", False),  # Should not match partial answer
    ]

    for true_answer, decoded_response, expected in test_cases:
        # Replicate the logic from MathEnv._compute_rewards
        if re.search(rf"\b{re.escape(true_answer)}\b", decoded_response):
            weak_correct = 1.0
        else:
            weak_correct = 0.0

        expected_weak_correct = 1.0 if expected else 0.0
        assert weak_correct == expected_weak_correct, (
            f"Failed for true_answer='{true_answer}', "
            f"response='{decoded_response}': expected {expected_weak_correct}, got {weak_correct}"
        )


def test_grade_answer_with_orz_env():
    """Test basic grade_answer functionality with ORZEnv."""
    from marin.post_training.environments.orz_env import ORZEnv

    env = ORZEnv(tokenizer=None)

    example = env.train_examples[0]
    result = grade_answer(given_answer="test", ground_truth=example["answer"])
    assert isinstance(result, bool)
    assert grade_answer(given_answer=example["answer"], ground_truth=example["answer"]) is True


def test_simple_latex_parser():
    """Test the new comprehensive regex-based LaTeX parser."""
    from marin.post_training.environments.math_utils import latex_to_text

    # Test complete fractions
    assert latex_to_text(r"\frac{1}{2}") == "(1)/(2)"
    assert latex_to_text(r"\frac{a+b}{c-d}") == "(a+b)/(c-d)"

    # Test incomplete fractions (the problematic case)
    assert latex_to_text(r"\frac{1}") == "(1)/"
    assert latex_to_text(r"\frac{a+b}") == "(a+b)/"

    # Test bare frac without braces
    assert latex_to_text(r"\frac 1 2") == "(1)/(2)"
    assert latex_to_text(r"\frac x y") == "(x)/(y)"

    # Test sqrt
    assert latex_to_text(r"\sqrt{2}") == "sqrt(2)"
    assert latex_to_text(r"\sqrt{x+y}") == "sqrt(x+y)"
    assert latex_to_text(r"\sqrt 2") == "sqrt(2)"

    # Test powers (key for sympy compatibility)
    assert latex_to_text(r"x^2") == "x**2"
    assert latex_to_text(r"x^{2}") == "x**(2)"
    assert latex_to_text(r"x^{n+1}") == "x**(n+1)"

    # Test subscripts
    assert latex_to_text(r"x_{1}") == "x_1"
    assert latex_to_text(r"a_{n}") == "a_n"

    # Test text removal
    assert latex_to_text(r"\text{hello}") == "hello"
    assert latex_to_text(r"x = \text{some text}") == "x = some text"

    # Test Greek letters
    assert latex_to_text(r"\pi") == "pi"
    assert latex_to_text(r"\theta") == "theta"
    assert latex_to_text(r"\alpha + \beta") == "alpha + beta"

    # Test math operators
    assert latex_to_text(r"\cdot") == "*"
    assert latex_to_text(r"\times") == "*"
    assert latex_to_text(r"\div") == "/"
    assert latex_to_text(r"\pm") == "+-"

    # Test special functions
    assert latex_to_text(r"\sin(x)") == "sin(x)"
    assert latex_to_text(r"\cos(\theta)") == "cos(theta)"
    assert latex_to_text(r"\log(n)") == "log(n)"

    # Test special values (sympy uses 'oo' for infinity)
    assert latex_to_text(r"\infty") == "oo"
    assert latex_to_text(r"\infinity") == "oo"

    # Test Unicode replacements
    assert latex_to_text("π") == "pi"
    assert latex_to_text("∞") == "oo"

    # Test \tfrac and \dfrac
    assert latex_to_text(r"\tfrac{1}{2}") == "(1)/(2)"
    assert latex_to_text(r"\dfrac{a}{b}") == "(a)/(b)"

    # Test left/right removal
    assert latex_to_text(r"\left(\frac{1}{2}\right)") == "((1)/(2))"

    # Test spacing command removal
    assert latex_to_text(r"a\,b\:c\;d") == "a b c d"
    assert latex_to_text(r"x\!y") == "xy"

    # Test complex expressions
    assert latex_to_text(r"\frac{1}{2} + \sqrt{3}") == "(1)/(2) + sqrt(3)"
    assert latex_to_text(r"2\pi r^2") == "2pi r**2"
    assert latex_to_text(r"\sin(\pi x) + \cos(\theta)") == "sin(pi x) + cos(theta)"

    # Test environment removal
    assert latex_to_text(r"\begin{equation}x=1\end{equation}") == "x=1"

    # Test malformed expressions (should not crash)
    assert latex_to_text(r"\frac{") == ""  # Empty content
    assert latex_to_text(r"\frac") == ""
    assert latex_to_text(r"\frac{1}{") == ""  # Malformed - removed entirely

    # Test empty input
    assert latex_to_text("") == ""


def test_parse_latex_robustness():
    """Test that the _parse_latex function is robust against problematic inputs."""
    from marin.post_training.environments.math_utils import latex_to_text

    # Test inputs that previously caused infinite loops
    problematic_inputs = [
        r"\frac{1}{",  # Incomplete frac
        r"\frac{",  # Very incomplete frac
        r"\frac",  # Bare frac
        r"\frac{1}{2}{3}",  # Too many braces
        r"\frac{1}{2\frac{3}",  # Nested incomplete frac
    ]

    for inp in problematic_inputs:
        # Should not hang or crash
        result = latex_to_text(inp)
        assert isinstance(result, str)

    # Test normal cases still work
    assert latex_to_text(r"\frac{1}{2}") == "(1)/(2)"
    assert latex_to_text(r"\sqrt{2}") == "sqrt(2)"
    assert latex_to_text(r"\pi") == "pi"
    assert latex_to_text(r"\infty") == "oo"  # Updated to reflect sympy convention


def test_latex_parser_integration_with_grade_answer():
    """Test that the new parser works correctly with grade_answer."""

    # Test cases that should work with the new parser
    test_cases = [
        # (given_answer, ground_truth, expected_result)
        (r"2\sqrt{2}-1", r"2\sqrt{2}-1", True),
        (r"\pi", "pi", None),  # May or may not match depending on normalization
        (r"x^2", "x**2", None),  # May or may not match
        (r"\sin(\pi)", "sin(pi)", None),  # May or may not match
        # Malformed inputs should still be processed without crashing
        (r"\frac{1}", "1/", None),  # Incomplete but processed - don't expect specific result
    ]

    for given, truth, expected in test_cases:
        try:
            result = grade_answer(given, truth)
            # The main goal is that it doesn't crash
            assert isinstance(result, bool)
            # For some cases we can verify the expected result
            if expected is not None:
                assert result == expected, f"Expected {expected} for grade_answer('{given}', '{truth}'), got {result}"
        except Exception as e:
            pytest.fail(f"grade_answer crashed on inputs: given='{given}', truth='{truth}': {e}")


def test_new_latex_features():
    """Test the new LaTeX parsing features that were added."""
    from marin.post_training.environments.math_utils import latex_to_text

    # Test dots notation
    assert latex_to_text(r"\dotsb") == "..."
    assert latex_to_text(r"\dots") == "..."
    assert latex_to_text(r"\ldots") == "..."
    assert latex_to_text(r"\cdots") == "..."

    # Test sum/integral notation - simplified to just the operator
    assert latex_to_text(r"\sum_{n=1}^{\infty}") == "sum"
    assert latex_to_text(r"\sum_{n=1}") == "sum"
    assert latex_to_text(r"\sum^{\infty}") == "sum"
    assert latex_to_text(r"\sum") == "sum"

    assert latex_to_text(r"\prod_{i=1}^{n}") == "prod"
    assert latex_to_text(r"\int_{0}^{1}") == "integral"

    # Test bold math commands - should just strip the formatting
    assert latex_to_text(r"\mathbf{M}") == "M"
    assert latex_to_text(r"\mathcal{H}") == "H"
    assert latex_to_text(r"\mathbb{R}") == "R"
    assert latex_to_text(r"\mathrm{text}") == "text"

    # Test ASY code block removal
    assert latex_to_text(r"Before [asy]draw((0,0)--(1,1));[/asy] After") == "Before After"

    # Test matrix notation
    assert latex_to_text(r"\begin{matrix}1 & 2 \\ 3 & 4\end{matrix}") == "[1 & 2 3 & 4]"
    assert latex_to_text(r"\begin{pmatrix}a & b\end{pmatrix}") == "[a & b]"


def test_normalize_malformed_expressions():
    """Test that _normalize handles malformed expressions properly."""
    from marin.post_training.environments.math_utils import _normalize

    # Test the case from the prompts: "192sqrt(14)25" should become "192*sqrt(14)/25"
    result = _normalize("192sqrt(14)25")
    assert "*" in result and "/" in result  # Should have operators inserted

    # Test adjacent numbers and functions get proper operators
    result = _normalize("2sqrt(3)")
    assert result == "2*sqrt(3)"

    result = _normalize("sqrt(2)5")
    assert result == "sqrt(2)*5"


def test_sympy_compatibility():
    """Test that our LaTeX parser produces sympy-compatible expressions."""
    import sympy

    from marin.post_training.environments.math_utils import latex_to_text

    # Test cases that should be parseable by sympy
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
            # Try to parse with sympy to verify compatibility
            sympy_expr = sympy.sympify(parsed)
            assert sympy_expr is not None
        except Exception as e:
            pytest.fail(f"Sympy failed to parse '{parsed}' (from '{latex_expr}'): {e}")


def test_normalize_answer():
    """Test the normalize_answer function."""
    from marin.post_training.environments.math_utils import normalize_answer

    assert normalize_answer("42") == "42"
    assert normalize_answer("\\text{hello}") == "hello"
    assert normalize_answer(None) is None
    assert normalize_answer("") == ""

    # Test LaTeX processing
    result = normalize_answer("\\frac{1}{2}")
    assert result is not None


def test_remove_boxed():
    """Test the remove_boxed function."""
    from marin.post_training.environments.math_utils import remove_boxed

    assert remove_boxed("\\boxed{42}") == "42"
    assert remove_boxed("\\boxed 42$") == "42$"  # This function only handles the specific format

    # Test nested braces
    assert remove_boxed("\\boxed{x + y}") == "x + y"


def test_last_boxed_only_string():
    """Test the last_boxed_only_string function."""
    from marin.post_training.environments.math_utils import last_boxed_only_string

    # Test normal case
    result = last_boxed_only_string("Some text \\boxed{42} more text")
    assert result == "\\boxed{42}"

    # Test multiple boxed expressions - should return the last one
    result = last_boxed_only_string("\\boxed{1} and \\boxed{2}")
    assert result == "\\boxed{2}"

    # Test no boxed expression
    result = last_boxed_only_string("No boxed content")
    assert result is None


def test_split_tuple():
    """Test the split_tuple function."""
    from marin.post_training.environments.math_utils import split_tuple

    # Test simple tuple
    assert split_tuple("(1, 2, 3)") == ["1", "2", "3"]

    # Test interval notation
    assert split_tuple("[0, 1]") == ["0", "1"]

    # Test single element
    assert split_tuple("42") == ["42"]

    # Test empty
    assert split_tuple("") == []

    # Test with commas in numbers
    assert split_tuple("(1,000, 2,000)") == ["1000", "2000"]


def test_are_equal_under_sympy():
    """Test the are_equal_under_sympy function."""
    from marin.post_training.environments.math_utils import are_equal_under_sympy

    # Test equal expressions
    assert are_equal_under_sympy("1/2", "0.5") is True
    assert are_equal_under_sympy("x+1", "1+x") is True

    # Test non-equal expressions
    assert are_equal_under_sympy("1/2", "1/3") is False

    # Test expressions that cause errors - should return False
    assert are_equal_under_sympy("badexpr", "1") is False


def test_should_allow_eval():
    """Test the should_allow_eval function."""
    from marin.post_training.environments.math_utils import should_allow_eval

    # Safe expressions
    assert should_allow_eval("1+2") is True
    assert should_allow_eval("x+y") is True

    # Unsafe expressions with too many variables
    assert should_allow_eval("a+b+c+d") is False

    # Expressions with bad substrings
    assert should_allow_eval("x^{2}") is False
    assert should_allow_eval("x^(2)") is False


def test_count_unknown_letters_in_expr():
    """Test the count_unknown_letters_in_expr function."""
    from marin.post_training.environments.math_utils import count_unknown_letters_in_expr

    assert count_unknown_letters_in_expr("x+y") == 2
    assert count_unknown_letters_in_expr("x+x") == 1  # x appears twice but is one letter
    assert count_unknown_letters_in_expr("123") == 0  # no letters
    assert count_unknown_letters_in_expr("sqrt(x)") == 1  # sqrt is removed, only x counts
    assert count_unknown_letters_in_expr("frac{x}{y}") == 2  # frac is removed, x and y count
