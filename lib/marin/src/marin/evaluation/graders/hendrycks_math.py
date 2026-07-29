# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Self-contained grader for the lm-eval-harness ``hendrycks_math`` task.

Ports ``lm_eval/tasks/hendrycks_math/utils.py`` at lm-evaluation-harness
``d5e3391``: grading needs only the problem text, the model's solution, and the
reference answer.

Equivalence is pure string normalization -- two answers match when
``strip_string`` maps them to the same text -- and already costs ~2 us per
problem, so the chain is reproduced verbatim rather than tuned. That includes
the points where it raises: ``is_equiv`` swallows those and falls back to raw
string equality, making the raising behavior part of the graded contract.
"""


def extract_answer(solution: str) -> str:
    """Return the span between the first and last ``$`` in a completion.

    Mirrors the extraction in ``process_results``: with fewer than two ``$``
    the whole completion is the answer.
    """
    first = solution.find("$")
    last = solution.rfind("$")
    if first == last:
        return solution
    return solution[first + 1 : last]


def fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
                continue
            if len(substr) < 2:
                return string
            a, b = substr[0], substr[1]
            post_substr = substr[2:]
            if b != "{":
                new_str += "{" + a + "}{" + b + "}" + post_substr
            else:
                new_str += "{" + a + "}" + b + post_substr
    return new_str


def fix_a_slash_b(string: str) -> str:
    """Rewrite an integer ``a/b`` as ``\\frac{a}{b}``.

    Only ``AssertionError`` is absorbed, matching the reference: a non-integer
    operand lets ``ValueError`` escape and drops ``is_equiv`` to raw equality.
    """
    parts = string.split("/")
    if len(parts) != 2:
        return string
    a = int(parts[0])
    b = int(parts[1])
    try:
        assert string == f"{a}/{b}"
    except AssertionError:
        return string
    return "\\frac{" + str(a) + "}{" + str(b) + "}"


def remove_right_units(string: str) -> str:
    """Drop a trailing ``\\text{ ...}`` unit.

    Raises ``AssertionError`` on a second ``\\text{ ``, which drops ``is_equiv``
    to raw equality.
    """
    if "\\text{ " not in string:
        return string
    splits = string.split("\\text{ ")
    assert len(splits) == 2
    return splits[0]


def fix_sqrt(string: str) -> str:
    """Brace a bare ``\\sqrt`` argument: ``\\sqrt3`` becomes ``\\sqrt{3}``.

    A trailing ``\\sqrt`` with no argument raises ``IndexError``, which drops
    ``is_equiv`` to raw equality.
    """
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    # The reference strips "\%" twice and never strips a bare "%" (its second
    # literal is an invalid escape Python leaves as backslash-percent). The
    # repeat is not redundant: str.replace is single-pass, so removing one match
    # can leave a newly adjacent one that only a second pass catches.
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # Drop a short variable binding such as "k =" or "q =".
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    return fix_a_slash_b(string)


def is_equiv(candidate: str | None, reference: str | None) -> bool:
    if candidate is None and reference is None:
        return True
    if candidate is None or reference is None:
        return False
    try:
        return strip_string(candidate) == strip_string(reference)
    except Exception:
        return candidate == reference


def grade(problem: str, solution: str, reference_answer: str) -> float:
    """Grade one completion against the reference answer.

    Args:
        problem: Problem statement. Unused; part of the shared grader signature.
        solution: The model's completion.
        reference_answer: Gold answer, already unwrapped from ``\\boxed{...}``.

    Returns:
        ``1.0`` when the completion matches the reference, else ``0.0``.
    """
    return 1.0 if is_equiv(extract_answer(solution), reference_answer) else 0.0
