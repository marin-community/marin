# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Whether a synthesized Reasoning Gym instance is right: its question agrees with its entry, and its answer is the
answer recomputed from the problem.

Most Reasoning Gym scorers compare a reply with ``entry["answer"]``, so an instance whose own answer is wrong still
scores 1.0 against itself; this check recomputes the answer instead. ``label_correct`` returns ``None`` for a type it
cannot check.
"""

import re
from collections.abc import Callable, Mapping
from decimal import ROUND_HALF_UP, Decimal, getcontext

import sympy

DECIMAL_SIGNIFICANT_DIGITS = 12
DECIMAL_TOLERANCE = Decimal("1e-9")


def _words(question: str) -> list[str]:
    return [word.strip() for word in question.rsplit(":", 1)[1].strip().split(",") if word.strip()]


def _word_sorting(question: str, entry: Mapping) -> bool:
    metadata = entry["metadata"]
    words = metadata["original_words"]
    if sorted(_words(question)) != sorted(words):
        return False
    descending = str(metadata["direction"]).startswith("desc")
    return entry["answer"].strip() == ", ".join(sorted(words, reverse=descending))


def _spell_backward(question: str, entry: Mapping) -> bool:
    word = entry["metadata"]["word"]
    return question.rstrip().endswith(word) and entry["answer"].strip() == word[::-1]


def _rotate_matrix(question: str, entry: Mapping) -> bool:
    metadata = entry["metadata"]
    matrix = [list(row) for row in metadata["matrix"]]
    numbers = re.findall(r"-?\d+", question.rsplit(":", 1)[1])
    if numbers != [str(value) for row in matrix for value in row]:
        return False
    degrees = int(re.search(r"by (\d+) degrees", question).group(1))
    for _ in range(degrees // 90 % 4):
        matrix = [list(row) for row in zip(*matrix[::-1], strict=True)]
    expected = "\n".join(" ".join(str(value) for value in row) for row in matrix)
    return entry["answer"].strip() == expected


def _simple_equations(question: str, entry: Mapping) -> bool:
    metadata = entry["metadata"]
    if metadata["equation"] not in question:
        return False
    left, right = metadata["equation"].split("=")
    variable = sympy.Symbol(metadata["variable"])
    solutions = sympy.solve(sympy.sympify(left) - sympy.sympify(right), variable)
    return len(solutions) == 1 and str(solutions[0]) == entry["answer"].strip()


def _decimal_arithmetic(question: str, entry: Mapping) -> bool:
    expression = question.rsplit("value.", 1)[1].split("=")[0].strip()
    if not re.fullmatch(r"[0-9.+\-*/() ]+", expression):
        return False
    getcontext().prec = 50
    value = eval(re.sub(r"(\d+\.?\d*)", r"Decimal('\1')", expression), {"Decimal": Decimal})
    exponent = value.adjusted() - DECIMAL_SIGNIFICANT_DIGITS + 1
    rounded = value.quantize(Decimal(1).scaleb(exponent), rounding=ROUND_HALF_UP)
    return abs(Decimal(entry["answer"].strip()) - rounded) <= DECIMAL_TOLERANCE * max(abs(rounded), 1)


ACRE_EXPERIMENT = re.compile(r"^(.+?) → (on|off)$")


def _acre_answer(question: str) -> str:
    """The detector light of the new case under the disjunctive rule: an object alone turning it on is a cause, every
    object of a set that leaves it off is not, and a set whose objects are neither is undetermined."""
    experiments, new_case = question.split("New test case:", 1)
    causes, inert = set(), set()
    trials = []
    for line in experiments.splitlines():
        match = ACRE_EXPERIMENT.match(line.strip())
        if match:
            trials.append((frozenset(part.strip() for part in match.group(1).split(",")), match.group(2) == "on"))
    for objects, on in trials:
        if not on:
            inert |= objects
    changed = True
    while changed:
        changed = False
        for objects, on in trials:
            unknown = objects - inert
            if on and len(unknown) == 1 and not unknown <= causes:
                causes |= unknown
                changed = True
    case = {part.strip() for part in new_case.split("What is", 1)[0].strip().split(",") if part.strip()}
    if case & causes:
        return "on"
    if case <= inert:
        return "off"
    return "undetermined"


def _acre(question: str, entry: Mapping) -> bool:
    return entry["answer"].strip() == _acre_answer(question)


CHECKS: Mapping[str, Callable[[str, Mapping], bool]] = {
    "word_sorting": _word_sorting,
    "spell_backward": _spell_backward,
    "rotate_matrix": _rotate_matrix,
    "simple_equations": _simple_equations,
    "decimal_arithmetic": _decimal_arithmetic,
    "acre": _acre,
}


def label_correct(name: str, question: str, entry: Mapping) -> bool | None:
    """Whether the instance's answer is the recomputed one; ``None`` when ``name`` has no check."""
    check = CHECKS.get(name)
    if check is None:
        return None
    try:
        return check(question, entry)
    except (
        KeyError,
        ValueError,
        TypeError,
        IndexError,
        AttributeError,
        ArithmeticError,
        SyntaxError,
        sympy.SympifyError,
    ):
        # An instance whose question or metadata does not parse is not a right instance.
        return False
