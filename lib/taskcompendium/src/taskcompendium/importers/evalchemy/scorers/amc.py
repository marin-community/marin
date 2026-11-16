# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""AMC23's Hendrycks comparator, adopted from lm-eval-harness v0.4.12 (MIT).

This accepts normalized string identity and retains raw equality when
normalization raises, rather than mathematical equivalence.
"""

from verifyit.grade import Reward, scored


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


def last_boxed_answer(response: str) -> str:
    """Extract using Hendrycks without reasoning or stop postprocessing."""
    if r"\boxed " in response:
        return response.split(r"\boxed ")[-1].split("$")[0]
    start = response.rfind(r"\boxed")
    if start < 0:
        return ""
    depth = 0
    for index in range(start, len(response)):
        if response[index] == "{":
            depth += 1
        elif response[index] == "}":
            depth -= 1
            if depth == 0:
                boxed = response[start : index + 1]
                return boxed[len(r"\boxed{") : -1] if boxed.startswith(r"\boxed{") else ""
    return ""


def grade_amc(expected: str, response: str) -> Reward:
    """Apply the source's last-box extraction and literal normalization comparator."""
    candidate = last_boxed_answer(response)
    return scored(float(is_equiv(expected, candidate)), extracted=candidate)
