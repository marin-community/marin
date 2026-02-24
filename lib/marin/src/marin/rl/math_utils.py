# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence).
Answer checker API that uses sympy to simplify expressions and check for equality.
Call grade_answer(given_answer: str, ground_truth: str).
"""

import logging
import re
from collections.abc import Callable
from typing import Any

import sympy
from sympy.parsing import sympy_parser

logger = logging.getLogger(__name__)


# Core LaTeX processing helpers
def find_matching_brace(text: str, start: int, open_b: str = "{", close_b: str = "}") -> int | None:
    """Find position of matching closing brace. Returns None if not found."""
    if start >= len(text) or not text[start:].startswith(open_b):
        return None

    nesting = 0
    i = start
    while i < len(text):
        if text[i : i + len(open_b)] == open_b:
            nesting += 1
            i += len(open_b)
        elif text[i : i + len(close_b)] == close_b:
            nesting -= 1
            if nesting == 0:
                return i + len(close_b) - 1
            i += len(close_b)
        else:
            i += 1
    return None


def extract_braced_content(text: str, start: int = 0) -> str | None:
    """Extract content from {content} starting at position start."""
    if start >= len(text) or text[start] != "{":
        return None
    end = find_matching_brace(text, start)
    return text[start + 1 : end] if end else None


def replace_latex_command(text: str, command: str, replacer: Callable[[list[str]], str], max_args: int = 2) -> str:
    """
    Replace LaTeX commands like \\command{arg1}{arg2} or \\command arg.

    Args:
        text: Input text
        command: LaTeX command name (without backslash)
        replacer: Function that takes list of arguments and returns replacement
        max_args: Maximum number of arguments to capture
    """
    pattern = f"\\{command}"
    result = []
    i = 0

    while i < len(text):
        pos = text.find(pattern, i)
        if pos == -1:
            result.append(text[i:])
            break

        result.append(text[i:pos])
        i = pos + len(pattern)

        # Collect arguments
        args = []
        for _ in range(max_args):
            # Skip whitespace
            while i < len(text) and text[i] in " \t":
                i += 1

            if i >= len(text):
                break

            if text[i] == "{":
                # Braced argument
                content = extract_braced_content(text, i)
                if content is not None:
                    args.append(content)
                    i = find_matching_brace(text, i) + 1
                else:
                    break
            elif args:
                # Only take non-braced args if we haven't found any braced ones
                break
            else:
                # Non-braced single argument (e.g., \\sqrt 2)
                j = i
                while j < len(text) and text[j] not in " \t{}\\\n":
                    j += 1
                if j > i:
                    args.append(text[i:j])
                    i = j
                break  # Only take one non-braced argument

        # Apply replacement
        if args:
            result.append(replacer(args))
        else:
            result.append(pattern)  # No args found, keep original

    return "".join(result)


def replace_latex_environment(text: str, env_name: str, replacer: Callable[[str], str]) -> str:
    """Replace \\begin{env}...\\end{env} with processed content."""
    pattern = f"\\\\begin{{{env_name}}}(.*?)\\\\end{{{env_name}}}"
    return re.sub(pattern, lambda m: replacer(m.group(1)), text, flags=re.DOTALL)


def normalize_answer(answer: str | None) -> str | None:
    """Main normalization function using new abstractions."""
    if not answer:
        return None

    answer = answer.strip()

    try:
        # Extract from formatting wrappers
        if "\\text{" in answer:
            content = extract_braced_content(answer, answer.find("\\text{") + 5)
            if content:
                answer = content

        # Extract from boxed if present
        answer = extract_boxed(answer)

        # Process LaTeX structures
        answer = remove_latex_formatting(answer)
        answer = process_latex_fractions(answer)
        answer = process_latex_sqrt(answer)
        answer = normalize_math_symbols(answer)

        # Normalize numbers
        answer = normalize_number_format(answer)

        # Remove redundant parentheses
        answer = remove_redundant_parens(answer)

        # Final normalization using existing logic
        answer = _normalize(answer)

        return answer
    except Exception as e:
        logger.warning(f"Normalization failed for answer '{answer}': {e}")
        return answer


def _fix_fracs(string):
    """Convert \\frac{a}{b} to (a)/(b), handling all variants. (Legacy - use process_latex_fractions)"""
    # Use new abstraction
    return process_latex_fractions(string)


def process_latex_fractions(text: str) -> str:
    """Convert \\frac{a}{b} to (a)/(b), handling all variants."""
    # Normalize variants
    text = text.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")

    def frac_replacer(args: list[str]) -> str:
        if len(args) >= 2:
            # Only add parentheses if arguments contain operators or spaces
            # Simple numbers like "7", "25" don't need parentheses
            num = args[0]
            den = args[1]

            # Check if numerator needs parentheses (contains operators/spaces/multiple terms)
            if re.search(r"[+\-*\s]|sqrt|sin|cos|tan|log|pi", num) or "," in num:
                num = f"({num})"

            # Check if denominator needs parentheses
            if re.search(r"[+\-*\s]|sqrt|sin|cos|tan|log|pi", den) or "," in den:
                den = f"({den})"

            return f"{num}/{den}"
        elif len(args) == 1:
            num = args[0]
            if re.search(r"[+\-*\s]|sqrt|sin|cos|tan|log|pi", num) or "," in num:
                num = f"({num})"
            return f"{num}/"
        return "\\frac"

    return replace_latex_command(text, "frac", frac_replacer, max_args=2)


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
    """Convert \\sqrt{x} or \\sqrt x to \\sqrt{x}. (Legacy - use process_latex_sqrt)"""
    # Use new abstraction for converting to text format
    return process_latex_sqrt(string)


def process_latex_sqrt(text: str) -> str:
    """Convert \\sqrt{x} or \\sqrt x to sqrt(x)."""
    return replace_latex_command(text, "sqrt", lambda args: f"sqrt({args[0]})" if args else "sqrt", max_args=1)


# Consolidated symbol replacements
MATH_SYMBOL_REPLACEMENTS = {
    # Greek letters
    "\\alpha": "alpha",
    "\\beta": "beta",
    "\\gamma": "gamma",
    "\\delta": "delta",
    "\\theta": "theta",
    "\\lambda": "lambda",
    "\\mu": "mu",
    "\\pi": "pi",
    "\\sigma": "sigma",
    "\\tau": "tau",
    "\\phi": "phi",
    "\\omega": "omega",
    # Operators
    "\\cdot": "*",
    "\\times": "*",
    "\\div": "/",
    "\\pm": "+-",
    "\\mp": "-+",
    # Functions (that don't take arguments in LaTeX)
    "\\sin": "sin",
    "\\cos": "cos",
    "\\tan": "tan",
    "\\log": "log",
    "\\ln": "ln",
    "\\exp": "exp",
    # Special
    "\\infty": "oo",
    "\\infinity": "oo",
    # Spacing
    "\\,": " ",
    "\\:": " ",
    "\\;": " ",
    "\\!": "",
    "\\quad": " ",
    "\\qquad": " ",
    # Display
    "\\displaystyle": "",
}


def normalize_math_symbols(text: str) -> str:
    """Replace LaTeX symbols with text equivalents."""
    for latex, replacement in MATH_SYMBOL_REPLACEMENTS.items():
        text = text.replace(latex, replacement)
    return text


def remove_latex_formatting(text: str) -> str:
    """Remove LaTeX formatting commands while preserving content."""

    def content_only(args: list[str]) -> str:
        return args[0] if args else ""

    for cmd in [
        "text",
        "mathbf",
        "mathcal",
        "mathbb",
        "mathrm",
        "mathit",
        "mathsf",
        "mathtt",
        "emph",
        "textbf",
        "textit",
    ]:
        text = replace_latex_command(text, cmd, content_only, max_args=1)

    # Remove left/right
    text = text.replace("\\left", "").replace("\\right", "")

    return text


def process_latex_matrices(text: str) -> str:
    """Convert matrix environments to simple bracket notation."""

    def matrix_replacer(content: str) -> str:
        return f"[{content}]"

    for env in ["matrix", "pmatrix", "bmatrix", "vmatrix", "Vmatrix"]:
        text = replace_latex_environment(text, env, matrix_replacer)

    return text


def normalize_number_format(text: str) -> str:
    """Normalize number formatting (commas, decimals, mixed numbers)."""
    # Remove commas from numbers like 1,234
    while True:
        new_text = re.sub(r"(\d),(\d{3})(?=\D|$)", r"\1\2", text)
        if new_text == text:
            break
        text = new_text

    # Fix leading decimals: .5 -> 0.5
    text = re.sub(r"(^|\s|\{)\.(\d)", r"\g<1>0.\2", text)

    # Convert mixed numbers: 7 3/4 -> 7+3/4
    text = re.sub(r"(\d+)\s+(\d+/\d+)", r"\1+\2", text)

    return text


def remove_redundant_parens(text: str) -> str:
    """Remove redundant parentheses around simple numbers and expressions."""
    # Handle mixed numbers first: 11(2)/(3) -> 11+2/3
    text = re.sub(r"(\d+)\((\d+)\)/\((\d+)\)", r"\1+\2/\3", text)

    # Remove parens around simple fractions: (5)/(3) -> 5/3
    text = re.sub(r"\((\d+)\)/\((\d+)\)", r"\1/\2", text)

    # Remove parens around standalone numbers
    text = re.sub(r"(?<![a-zA-Z()])\((\d+(?:\.\d+)?)\)(?![a-zA-Z()])", r"\1", text)

    return text


def extract_boxed(text: str) -> str | None:
    """Extract content from last \\boxed{} or \\boxed."""
    # Try \\boxed{content} first
    last_pos = text.rfind("\\boxed")
    if last_pos == -1:
        return text

    # Check if it's followed by a brace
    check_pos = last_pos + 6
    while check_pos < len(text) and text[check_pos] in " \t":
        check_pos += 1

    if check_pos < len(text) and text[check_pos] == "{":
        return extract_braced_content(text, check_pos)

    # Handle \\boxed content (space-separated)
    if text[last_pos:].startswith("\\boxed "):
        content = text[last_pos + 7 :]
        dollar_pos = content.find("$")
        return content[:dollar_pos] if dollar_pos >= 0 else content

    return text


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

    # remove spaces
    string = string.replace(" ", "")

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
    """LaTeX to text converter using new abstractions."""
    if not expr:
        return expr

    # Remove ASY (Asymptote) code blocks first
    expr = re.sub(r"\[asy\].*?\[/asy\]", "", expr, flags=re.DOTALL)

    # Handle dots notation first
    expr = expr.replace("\\dotsb", "...")
    expr = expr.replace("\\dots", "...")
    expr = expr.replace("\\ldots", "...")
    expr = expr.replace("\\cdots", "...")

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

    # Use our new abstractions for consistent processing
    expr = process_latex_fractions(expr)
    expr = process_latex_sqrt(expr)

    # Powers: x^{n} -> x**(n) or x^n -> x**n
    expr = re.sub(r"\^{([^{}]*)}", r"**(\1)", expr)
    expr = re.sub(r"\^(\w+)", r"**\1", expr)

    # Subscripts (often just remove for sympy): x_{1} -> x_1
    expr = re.sub(r"_{([^{}]*)}", r"_\1", expr)

    # Remove LaTeX formatting using our abstraction
    expr = remove_latex_formatting(expr)

    # Handle matrices
    expr = process_latex_matrices(expr)

    # Use consolidated symbol replacements (includes spacing and display commands)
    expr = normalize_math_symbols(expr)

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


def _str_to_int(x: str) -> int:
    x = x.replace(",", "")
    x_float = float(x)
    return int(x_float)


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
    if not given_normalized:
        return False

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

    logger.info(f"Grading given_answer: '{given_answer}' vs ground_truth: '{ground_truth}'")
    ground_truth_normalized = normalize_answer(ground_truth)
    given_normalized = normalize_answer(given_answer)

    if (not ground_truth_normalized) or (not given_normalized):
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


def remove_matching_braces(s, brace_open="{", brace_close="}"):
    """Remove content up to and including the first matching brace pair."""
    end_pos = find_matching_brace(s, 0, brace_open, brace_close)
    return s[: end_pos + 1] if end_pos is not None else s


def remove_boxed(s):
    """Remove \\boxed wrapper and return content."""
    if "\\boxed " in s:
        left = "\\boxed "
        if s.startswith(left):
            return s[len(left) :]

    idx = s.find("\\boxed{")
    if idx >= 0:
        content = extract_braced_content(s, idx + 6)
        return content if content is not None else s

    return s


def last_boxed_only_string(value):
    """Extract the last \\boxed{} content as a full string."""
    idx = value.rfind("\\boxed")
    if idx == -1:
        return value

    # Handle \\boxed content (space-separated)
    if value[idx:].startswith("\\boxed "):
        content = value[idx + 7 :]
        dollar_pos = content.find("$")
        content = content[:dollar_pos] if dollar_pos >= 0 else content
        return "\\boxed " + content

    # Handle \\boxed{content}
    brace_pos = idx + 6  # len("\\boxed")
    while brace_pos < len(value) and value[brace_pos] in " \t":
        brace_pos += 1

    if brace_pos < len(value) and value[brace_pos] == "{":
        end_pos = find_matching_brace(value, brace_pos)
        if end_pos is not None:
            return value[idx : end_pos + 1]

    return None


def validate_format(input_text: str) -> dict[str, Any]:
    """
    Validates if input text follows the required format with <answer> tag.
    Returns a dictionary with validation status and extracted answer.
    """
    # Regex to match the entire expected pattern
    # - Looks for <answer> tag, content, and </answer> tag
    regex = r"<answer>([\s\S]*?)</answer>"

    # Search for the pattern in the input
    match = re.search(regex, input_text)

    if match:
        # If we have a match, extract the answer content
        answer_content = match.group(1).strip()

        return {
            "is_valid": True,
            "answer": answer_content,
        }
    else:
        # If no match, format is invalid
        return {
            "is_valid": False,
            "answer": None,
        }
