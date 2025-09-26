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

"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence).
Answer checker API that uses sympy to simplify expressions and check for equality.
Call grade_answer(given_answer: str, ground_truth: str).
"""

import re

import sympy
from sympy.parsing import sympy_parser


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\\{(?P<text>.+?)\\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        stripped = _strip_string(answer)
        return _normalize(remove_boxed(stripped))
    except BaseException:
        return answer


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except BaseException:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == f"{a}/{b}"
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except BaseException:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
    # Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in
    # case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


# sympy might hang -- we don't care about trying to be lenient in these cases
BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\\^[0-9]+\\^", "\\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"


def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(
            *sympy_parser.standard_transformations,
            sympy_parser.implicit_multiplication_application,
        ),
    )


def latex_to_text(expr: str) -> str:
    """Simple regex-based LaTeX to text converter for math expressions."""
    if not expr:
        return expr

    # Remove ASY (Asymptote) code blocks first
    expr = re.sub(r"\[asy\].*?\[/asy\]", "", expr, flags=re.DOTALL)

    # Handle dots notation first
    expr = expr.replace("\\dotsb", "...")
    expr = expr.replace("\\dots", "...")
    expr = expr.replace("\\ldots", "...")
    expr = expr.replace("\\cdots", "...")

    # Handle \tfrac and \dfrac first
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")

    # Sum, product, and integral notation - do this before subscript/superscript processing
    expr = re.sub(r"\\sum\s*_{[^}]*}\s*\^{[^}]*}", "sum", expr)
    expr = re.sub(r"\\sum\s*_{[^}]*}", "sum", expr)
    expr = re.sub(r"\\sum\s*\^{[^}]*}", "sum", expr)
    expr = expr.replace("\\sum", "sum")

    expr = re.sub(r"\\prod\s*_{[^}]*}\s*\^{[^}]*}", "prod", expr)
    expr = re.sub(r"\\prod\s*_{[^}]*}", "prod", expr)
    expr = re.sub(r"\\prod\s*\^{[^}]*}", "prod", expr)
    expr = expr.replace("\\prod", "prod")

    expr = re.sub(r"\\int\s*_{[^}]*}\s*\^{[^}]*}", "integral", expr)
    expr = re.sub(r"\\int\s*_{[^}]*}", "integral", expr)
    expr = re.sub(r"\\int\s*\^{[^}]*}", "integral", expr)
    expr = expr.replace("\\int", "integral")

    # Handle complete fractions: \frac{a}{b} -> (a)/(b)
    expr = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", expr)

    # Handle incomplete fractions: \frac{a} -> (a)/
    expr = re.sub(r"\\frac\s*\{([^{}]*)\}(?!\s*\{)", r"(\1)/", expr)

    # Handle bare fractions: \frac a b -> (a)/(b)
    expr = re.sub(r"\\frac\s+(\S+)\s+(\S+)", r"(\1)/(\2)", expr)

    # Clean up malformed \frac cases
    expr = re.sub(r"\\frac\s*\{[^}]*\}\s*\{[^}]*$", "", expr)  # \frac{x}{ incomplete
    expr = re.sub(r"\\frac\s*\{[^}]*$", "", expr)  # \frac{ incomplete
    expr = re.sub(r"\\frac\b", "", expr)  # bare \frac

    # Square roots: \sqrt{x} -> sqrt(x)
    expr = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", expr)
    expr = re.sub(r"\\sqrt\s*(\S+)", r"sqrt(\1)", expr)  # \sqrt x -> sqrt(x)

    # Powers: x^{n} -> x**(n) or x^n -> x**n
    expr = re.sub(r"\^{([^{}]*)}", r"**(\1)", expr)
    expr = re.sub(r"\^(\w+)", r"**\1", expr)

    # Subscripts (often just remove for sympy): x_{1} -> x_1
    expr = re.sub(r"_{([^{}]*)}", r"_\1", expr)

    # Remove \text{} wrappers
    expr = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", expr)

    # Remove math font commands (bold, calligraphic, blackboard bold, etc.)
    expr = re.sub(r"\\mathbf\s*\{([^{}]*)\}", r"\1", expr)
    expr = re.sub(r"\\mathcal\s*\{([^{}]*)\}", r"\1", expr)
    expr = re.sub(r"\\mathbb\s*\{([^{}]*)\}", r"\1", expr)
    expr = re.sub(r"\\mathrm\s*\{([^{}]*)\}", r"\1", expr)
    expr = re.sub(r"\\mathit\s*\{([^{}]*)\}", r"\1", expr)
    expr = re.sub(r"\\mathsf\s*\{([^{}]*)\}", r"\1", expr)
    expr = re.sub(r"\\mathtt\s*\{([^{}]*)\}", r"\1", expr)

    # Remove \left and \right (they're just sizing hints)
    expr = expr.replace("\\left", "")
    expr = expr.replace("\\right", "")

    # Greek letters and symbols
    expr = expr.replace("\\alpha", "alpha")
    expr = expr.replace("\\beta", "beta")
    expr = expr.replace("\\gamma", "gamma")
    expr = expr.replace("\\delta", "delta")
    expr = expr.replace("\\theta", "theta")
    expr = expr.replace("\\lambda", "lambda")
    expr = expr.replace("\\mu", "mu")
    expr = expr.replace("\\pi", "pi")
    expr = expr.replace("\\sigma", "sigma")
    expr = expr.replace("\\tau", "tau")
    expr = expr.replace("\\phi", "phi")
    expr = expr.replace("\\omega", "omega")

    # Math operators
    expr = expr.replace("\\cdot", "*")
    expr = expr.replace("\\times", "*")
    expr = expr.replace("\\div", "/")
    expr = expr.replace("\\pm", "+-")
    expr = expr.replace("\\mp", "-+")

    # Special functions
    expr = expr.replace("\\sin", "sin")
    expr = expr.replace("\\cos", "cos")
    expr = expr.replace("\\tan", "tan")
    expr = expr.replace("\\log", "log")
    expr = expr.replace("\\ln", "ln")
    expr = expr.replace("\\exp", "exp")

    # Special values - sympy uses 'oo' for infinity
    expr = expr.replace("\\infty", "oo")
    expr = expr.replace("\\infinity", "oo")

    # Remove common spacing commands
    expr = expr.replace("\\,", " ")
    expr = expr.replace("\\:", " ")
    expr = expr.replace("\\;", " ")
    expr = expr.replace("\\!", "")
    expr = expr.replace("\\quad", " ")
    expr = expr.replace("\\qquad", " ")

    # Remove display style commands
    expr = expr.replace("\\displaystyle", "")

    # Handle matrix environments - convert to simple bracket notation
    # \begin{matrix}...\end{matrix} -> [...]
    expr = re.sub(r"\\begin\{matrix\}(.*?)\\end\{matrix\}", r"[\1]", expr, flags=re.DOTALL)
    expr = re.sub(r"\\begin\{pmatrix\}(.*?)\\end\{pmatrix\}", r"[\1]", expr, flags=re.DOTALL)
    expr = re.sub(r"\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}", r"[\1]", expr, flags=re.DOTALL)
    expr = re.sub(r"\\begin\{vmatrix\}(.*?)\\end\{vmatrix\}", r"[\1]", expr, flags=re.DOTALL)
    expr = re.sub(r"\\begin\{Vmatrix\}(.*?)\\end\{Vmatrix\}", r"[\1]", expr, flags=re.DOTALL)

    # Handle other \begin{} \end{} environments by removing them
    expr = re.sub(r"\\begin\{[^}]*\}", "", expr)
    expr = re.sub(r"\\end\{[^}]*\}", "", expr)

    # Replace Unicode symbols that might come from other processing
    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "oo")
    expr = expr.replace("∪", "U")  # noqa: RUF001
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")  # noqa: RUF001

    # Clean up: remove backslashes that might be left
    # Be careful here - only remove isolated backslashes
    expr = re.sub(r"\\(?![a-zA-Z])", "", expr)

    # Clean up multiple spaces
    expr = re.sub(r"\s+", " ", expr)

    return expr.strip()


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False


def _is_int(x: float) -> bool:
    try:
        return abs(x - round(x)) <= 1e-7
    except BaseException:
        return False


def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - round(x)) <= 1e-7
    except BaseException:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):
    """
    Automatically make a mixed number evalable
    e.g. 7 3/4 => 7+3/4
    """
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)  # implicit mults
    return step


def _strip_properly_formatted_commas(expr: str):
    # We want to be careful because we don't want to strip tuple commas
    p1 = re.compile("(\\d)(,)(\\d\\d\\d)($|\\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _normalize(expr: str) -> str:
    """Normalize answer expressions."""
    if expr is None:
        return None

    # Remove enclosing `\text{}`.
    m = re.search("^\\\\text\\{(?P<text>.+?)\\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\\^[0-9]+)?", "", expr)
    expr = re.sub("\\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(round(float(expr)))
    if "\\" in expr:
        try:
            expr = latex_to_text(expr)
        except BaseException:
            pass

    # Handle malformed expressions like "192sqrt(14)25" -> "192*sqrt(14)/25"
    # Pattern: number + function + number -> number * function / number
    expr = re.sub(r"(\d+)(sqrt|sin|cos|tan|log|ln|exp)(\([^)]*\))(\d+)", r"\1*\2\3/\4", expr)

    # Handle cases where numbers are adjacent to functions without operators
    expr = re.sub(r"(\d+)(sqrt|sin|cos|tan|log|ln|exp)(\([^)]*\))", r"\1*\2\3", expr)
    expr = re.sub(r"(sqrt|sin|cos|tan|log|ln|exp)(\([^)]*\))(\d+)", r"\1\2*\3", expr)

    # edge case with mixed numbers and negative signs
    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)
    expr = expr.replace(" ", "")

    # if we somehow still have latex braces here, just drop them
    expr = expr.replace("{", "")
    expr = expr.replace("}", "")

    # don't be case sensitive for text answers
    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)


def should_allow_eval(expr: str):
    # we don't want to try parsing unknown text or functions of more than two
    # variables
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    for bad_regex in BAD_REGEXES:
        if re.search(bad_regex, expr) is not None:
            return False

    return True


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except BaseException:
        pass
    return are_equal


def split_tuple(expr: str):
    """
    Split the elements in a tuple/interval, while handling well-formatted commas in large numbers
    """
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if (
        len(expr) > 2
        and expr[0] in TUPLE_CHARS
        and expr[-1] in TUPLE_CHARS
        and all([ch not in expr[1:-1] for ch in TUPLE_CHARS])
    ):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems


def grade_answer(given_answer: str, ground_truth: str) -> bool:
    """
    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    """
    if given_answer is None:
        return False

    ground_truth_normalized = normalize_answer(ground_truth)
    given_normalized = normalize_answer(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (
        ground_truth_normalized[0] != given_normalized[0] or ground_truth_normalized[-1] != given_normalized[-1]
    ):
        is_correct = False
    elif len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems, strict=False):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                # if fractions aren't reduced, then shouldn't be marked as correct
                # so, we don't want to allow sympy.simplify in this case
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                # if the ground truth answer is an integer, we require the
                # given answer to be a strict match (no sympy.simplify)
                is_correct = False
            else:
                is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
            if not is_correct:
                break

    return is_correct


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval
