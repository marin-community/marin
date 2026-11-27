# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ChemBench ``[ANSWER]`` extraction and ``all_correct`` grading.

Adapted from lamalab-org/chembench (MIT) at 45f8bad062fe552810c52be3a328d5da8597ed30:
``prepare_mcq_answer``, ``prepare_general_answer`` and ``BasePrompter._calculate_metrics`` in
``src/chembench/prompter.py``, ``find_numbers`` and ``convert_to_number`` in ``src/chembench/utils.py``,
and the regexes in ``src/chembench/constant.py``.

A multiple-choice answer is correct when the letters inside every ``[ANSWER]`` tag equal the set of
correct letters. A numeric answer is the first number inside the first tag, correct when its absolute
error is below 1% of the target's magnitude, or zero. The source uses ``0.01 * target``, which no answer
satisfies for a negative target. Its LLM extraction fallback, number-word parsing and refusal retry are
omitted, so a response without an answer tag scores zero. ``convert_to_number``'s pint parsing is
replaced by arithmetic for the forms ``NUMBER_PATTERN`` matches: ``N^E``, ``N x B^E`` and ``eE``. The
source's pattern matches the literal text ``\\u00d7``; this one matches the U+00D7 multiplication sign,
reads a U+2212 or en-dash minus sign directly before a digit as ``-``, and accepts space-separated letters, which
the prompt allows and which the source's regex leaves to its LLM extractor.
"""

import math
import re

from verifyit.adapters.completion import final_text
from verifyit.grade import Reward, scored

MULTIPLE_CHOICE_ANSWER = re.compile(
    r"(?:\[ANSWER\]|\[ANS\]|<ANS>|<ANSWER>|<ans>)\s*([A-Z](?:(?:\s*,\s*|\s+)[A-Z](?![A-Za-z]))*)\s*(?:\..*?|,.*?)?"
    r"(?:\[/?ANSWER\]|\[/ANS\]|</ANS>|</ANSWER>|</ans>)",
    re.DOTALL,
)
NUMERIC_ANSWER = re.compile(r"\[ANSWER\][\s\n]*(.*?)[\s\n]*\[/?ANSWER\]")
NUMBER_PATTERN = re.compile(
    r"(?<!\w)([+-]?[0-9]*\.?[0-9]+\s?)?([\^]([-+]?[0-9]+))?(?:(?:[eE]|exp|EXP)[-+]?[0-9]+)?"
    r"(?:([*x\u00d7]|\\times)\s?[0-9]+[\^]([-+]?[0-9]+|{([-+]?[0-9]+)}))?(?!\w)"
)
POWER_FORM = re.compile(r"([+-]?)([0-9]*\.?[0-9]+)\s?(?:\^([-+]?[0-9]+))?(?:\*\s?([0-9]+)\^([-+]?[0-9]+))?")
BARE_EXPONENT = re.compile(r"e[0-9]+")
UNICODE_MINUS = re.compile(r"[\u2212\u2013](?=[0-9.])")
RELATIVE_TOLERANCE = 0.01


def multiple_choice_letters(text: str, letters: list[str]) -> list[str] | None:
    """Sorted offered letters across all answer tags, or None when no tag holds letters."""
    matches = set(MULTIPLE_CHOICE_ANSWER.findall(text))
    if not matches:
        return None
    return sorted({letter for match in matches for letter in re.findall("[A-Z]", match) if letter in letters})


def number_value(token: str) -> float | None:
    """Evaluate a ``NUMBER_PATTERN`` match as the source's float-then-pint conversion does."""
    text = token.replace("x", "*").replace("\u00d7", "*").replace("\\times", "*")
    try:
        return float(text)
    except ValueError:
        pass
    if BARE_EXPONENT.fullmatch(text):
        return float("1" + text)
    power = POWER_FORM.fullmatch(text)
    if power is None:
        return None
    sign, mantissa, exponent, base, base_exponent = power.groups()
    try:
        value = float(mantissa) ** int(exponent) if exponent else float(mantissa)
        if base:
            value *= float(base) ** int(base_exponent)
    except (OverflowError, ZeroDivisionError):
        return None
    return -value if sign == "-" else value


def numeric_answer(text: str) -> tuple[str | None, float | None]:
    """The first answer tag and the value of the first number inside it."""
    tag = NUMERIC_ANSWER.search(text)
    if tag is None:
        return None, None
    answer = UNICODE_MINUS.sub("-", tag.group(0))
    number = next((match.group(0) for match in NUMBER_PATTERN.finditer(answer) if match.group(0)), None)
    value = None if number is None else number_value(number)
    return tag.group(0), value if value is not None and math.isfinite(value) else None


def grade_chembench(config: dict, response: str) -> Reward:
    text = final_text(response)
    if config["kind"] == "multiple_choice":
        chosen = multiple_choice_letters(text, config["letters"])
        correct = chosen is not None and set(chosen) == set(config["correct"])
        return scored(float(correct), extracted=chosen, expected=config["correct"])
    target = config["target"]
    tag, value = numeric_answer(text)
    correct = value is not None and (value == target or abs(value - target) < RELATIVE_TOLERANCE * abs(target))
    return scored(float(correct), extracted=tag, value=value, expected=target)
