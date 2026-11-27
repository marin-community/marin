# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade final assistant text without Polaris's lossy expression normalization."""

import json
import re
from pathlib import Path

from polaris_answer_extract import BOXED, extract_boxed, last_line, strip_math_delimiters
from polaris_final_text import final_answer_text
from verifyit.modes.grade_math import grade_math_candidate
from verifyit.spec import MathSpec

FINAL_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
FINAL_CLAIM = re.compile(r"\b(?:is|equals)\s+|^answer:\s*", re.IGNORECASE)
OPENING_ANSWER = re.compile(r"^(?:the\s+.+?\s+(?:is|equals)\s+|answer:\s*)", re.IGNORECASE)
FINAL_EXPRESSION = re.compile(r"^(?:[+\-\d.(√]|\\(?:frac|dfrac|sqrt|boxed)\b|\\\(|\\\[|\$|[A-Za-z]\s*[=^])")
NEGATION = re.compile(r"\b(?:not|never|incorrect|wrong|false)\b", re.IGNORECASE)
UNITS = frozenset({"mm", "cm", "m", "km", "mg", "g", "kg", "s", "sec", "min", "h", "hr", "mile", "miles"})
QUANTITY = re.compile(r"([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s+([a-z]+)")
UNICODE_ROOT = re.compile(r"√([0-9]+(?:\.[0-9]+)?)(?![\w.])")


def final_candidate(answer: str, problem: str) -> str | None:
    """Read one final claim without consulting the reference or earlier reasoning."""
    final = final_answer_text(answer)
    line = last_line(final)
    if not line:
        return None
    sentence = FINAL_SENTENCE.split(line)[-1]
    # A direct opening answer may be followed by a proof containing other numbers.
    # Select its claim before mathematical comparison; the reference never chooses it.
    opening_sentences = FINAL_SENTENCE.split(final.splitlines()[0])
    opening_answers = [value for value in opening_sentences if OPENING_ANSWER.match(value)]
    if opening_answers and not FINAL_EXPRESSION.match(line):
        sentence = opening_answers[-1]
    if NEGATION.search(sentence):
        return None
    if BOXED in final:
        candidate = extract_boxed(final)
    else:
        claims = list(FINAL_CLAIM.finditer(sentence))
        candidate = sentence[claims[-1].end() :] if claims else sentence
    if not candidate:
        return None
    candidate = strip_math_delimiters(candidate.strip().rstrip(".!"))
    quantity = QUANTITY.fullmatch(candidate)
    if quantity:
        number, unit = quantity.groups()
        problem_units = UNITS.intersection(re.findall(r"\b[a-z]+\b", problem.lower()))
        if problem_units != {unit}:
            return None
        candidate = number
    return UNICODE_ROOT.sub(r"sqrt(\1)", candidate)


def score(answer: str, contract: dict[str, str]) -> float:
    candidate = final_candidate(answer, contract["problem"])
    if not candidate:
        return 0.0
    reference = contract["answer"]
    if BOXED in reference:
        reference = extract_boxed(reference)
        if reference is None:
            raise ValueError("Polaris reference contains an incomplete box")
    # A word denotes a categorical answer, not a commutative product of letters.
    if reference.isalpha() and len(reference) > 1:
        return float(candidate.strip().casefold() == reference.casefold())
    return grade_math_candidate(MathSpec(expected=reference), candidate).reward


def main() -> None:
    print(score(Path("/app/answer.txt").read_text(), json.loads(Path("/tests/config.json").read_text())))


if __name__ == "__main__":
    main()
