# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-contained grader for the lm-eval-harness ``minerva_math`` task.

Ports the ``exact_match`` metric of ``lm_eval/tasks/minerva_math/utils.py`` at
lm-evaluation-harness ``d5e3391`` so grading needs only the problem text, the
model's solution text, and the reference answer.

The reference parses both sides with sympy's ANTLR LaTeX parser and simplifies
the difference; ~98% of its runtime is ``parse_latex`` (2.4 ms per call, twice
per grade) against ~0.05 ms for ``simplify``. This module returns the same
verdicts while skipping the parser wherever the answer shape already settles
the question, and falls through to sympy otherwise. The shape rules are
documented at each pattern below.

sympy and its ANTLR runtime are optional (the ``math`` extra), so they are
imported lazily and only on the fallback path.
"""

import functools
import re
import signal
from fractions import Fraction

INVALID_ANSWER = "[invalidanswer]"
_SYMPY_TIMEOUT = 5
_PARSE_CACHE_SIZE = 8192

_FINAL_ANSWER = re.compile(r"Final Answer: The final answer is(.*?). I hope it is correct.")

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "ft",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]

_DOLLAR_SPAN = re.compile(r"(.*?)(\$)(.*?)(\$)(.*)")
_TEXT = re.compile(r"(\\text\{)(.*?)(\})")
_TEXTBF = re.compile(r"(\\textbf\{)(.*?)(\})")
_OVERLINE = re.compile(r"(\\overline\{)(.*?)(\})")
_BOXED = re.compile(r"(\\boxed\{)(.*)(\})")
_SHORT_FRAC = re.compile(r"(frac)([^{])(.)")
_SHORT_SQRT = re.compile(r"(sqrt)([^{])")

# Bracketed comma lists (tuples, intervals) have no production in sympy's LaTeX
# grammar, so the reference scores them 0 against anything. Three details make
# the pattern sound: a bare "1,2,3" parses as its first element rather than
# failing; a comma followed by three digits is a thousands separator, so
# "(100,101)" lexes as 100101 and parses; and the comma must fall inside the
# brackets, since trailing punctuation like "(E)," parses fine.
_BRACKETED_TUPLE = re.compile(r"^(?:\\left)?[(\[][^)\]]*,(?!\d{3})")

# Integers reach sympy as Python int literals, so a leading zero is a
# SyntaxError: "007" and "\frac{007}{2}" both fail to parse. Decimals skip that
# path ("00.5" is fine) but need a digit before the point (".2" fails). A
# leading "+" is left to the fallback rather than assumed parseable.
_UNSIGNED_INT = r"(?:0|[1-9][0-9]*)"
_INTEGER = re.compile(rf"^-?{_UNSIGNED_INT}$")
_DECIMAL = re.compile(r"^-?[0-9]+\.[0-9]+$")
_FRACTION = re.compile(rf"^(?P<sign>-?)\\frac\{{(?P<num>-?{_UNSIGNED_INT})\}}\{{(?P<den>-?{_UNSIGNED_INT})\}}$")

# simplify() rationalizes a Float against a Rational at sympy's default 15-digit
# precision, so decimals compare exactly as Fractions only up to that width.
_FLOAT_SIGNIFICANT_DIGITS = 15


def extract_answer(solution: str) -> str:
    """Pull the answer out of a minerva-style completion.

    Returns ``INVALID_ANSWER`` when the completion has no
    ``Final Answer: The final answer is ...`` line.
    """
    match = _FINAL_ANSWER.search(solution + "I hope it is correct.")
    return match.group(1).strip() if match else INVALID_ANSWER


def normalize_final_answer(final_answer: str) -> str:
    """Normalize an answer per appendix D of Lewkowycz et al. (2022)."""
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = _DOLLAR_SPAN.sub("$\\3$", final_answer)
    final_answer = _TEXT.sub("\\2", final_answer)
    final_answer = _TEXTBF.sub("\\2", final_answer)
    final_answer = _OVERLINE.sub("\\2", final_answer)
    final_answer = _BOXED.sub("\\2", final_answer)

    final_answer = _SHORT_FRAC.sub("frac{\\2}{\\3}", final_answer)
    final_answer = _SHORT_SQRT.sub("sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer


def _rational_value(string: str) -> Fraction | None:
    """Exact value of a plain integer, decimal, or integer fraction, else ``None``."""
    if _INTEGER.match(string):
        return Fraction(int(string))

    if _DECIMAL.match(string):
        digits = string.lstrip("+-").replace(".", "").lstrip("0")
        if len(digits) > _FLOAT_SIGNIFICANT_DIGITS:
            return None
        return Fraction(string)

    match = _FRACTION.match(string)
    if match is None:
        return None
    denominator = int(match["den"])
    if denominator == 0:
        return None
    value = Fraction(int(match["num"]), denominator)
    return -value if match["sign"] else value


@functools.lru_cache(maxsize=_PARSE_CACHE_SIZE)
def _sympy_parses(string: str):
    """Parse a LaTeX string, or return ``None`` if sympy rejects it.

    Memoized: parsing is pure and answers repeat across a run (MATH's 12.5k
    answers cover 4.3k distinct strings).
    """
    import sympy  # noqa: PLC0415  # optional dep: sympy
    from sympy.parsing.latex import parse_latex  # noqa: PLC0415  # optional dep: sympy
    from sympy.parsing.latex.errors import LaTeXParsingError  # noqa: PLC0415  # optional dep: sympy

    try:
        return parse_latex(string)
    except (LaTeXParsingError, sympy.SympifyError, TypeError):
        return None


def _sympy_is_equiv(candidate: str, reference: str) -> bool:
    """Parse both sides and simplify the difference, as the reference does.

    Identical strings need only one parse, but still take the difference: a
    parsed relation such as ``a\\leq b`` cannot be subtracted from itself and so
    scores 0.
    """
    import sympy  # noqa: PLC0415  # optional dep: sympy

    def on_timeout(signum, frame):
        raise TimeoutError

    # Arming the alarm sits inside the guard: off the main thread signal.signal
    # raises, which the reference also scores as a non-match.
    previous = None
    try:
        previous = signal.signal(signal.SIGALRM, on_timeout)
        signal.alarm(_SYMPY_TIMEOUT)
        parsed_candidate = _sympy_parses(candidate)
        if parsed_candidate is None:
            return False
        if candidate == reference:
            parsed_reference = parsed_candidate
        else:
            parsed_reference = _sympy_parses(reference)
            if parsed_reference is None:
                return False
        return sympy.simplify(parsed_candidate - parsed_reference) == 0
    except ImportError:
        # Without antlr4-python3-runtime 4.11 every comparison would quietly
        # score 0, so surface it rather than report a non-match.
        raise
    except Exception:
        # Parse, subtraction, simplify, and timeout failures all score 0.
        return False
    finally:
        if previous is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)


def is_equiv(candidate: str, reference: str) -> bool:
    """Decide equivalence of two already-normalized LaTeX answers."""
    if _BRACKETED_TUPLE.match(candidate) or _BRACKETED_TUPLE.match(reference):
        return False

    candidate_value = _rational_value(candidate)
    if candidate_value is not None:
        reference_value = _rational_value(reference)
        if reference_value is not None:
            return candidate_value == reference_value

    return _sympy_is_equiv(candidate, reference)


def grade(problem: str, solution: str, reference_answer: str) -> float:
    """Grade one completion against the reference answer.

    Args:
        problem: Problem statement. Unused; part of the shared grader signature.
        solution: The model's completion.
        reference_answer: Gold answer, already unwrapped from ``\\boxed{...}``.

    Returns:
        ``1.0`` when the completion matches the reference, else ``0.0``.
    """
    candidate = normalize_final_answer(extract_answer(solution))
    reference = normalize_final_answer(reference_answer)
    return 1.0 if is_equiv(candidate, reference) else 0.0
