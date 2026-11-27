# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade final assistant text without Polaris's lossy expression normalization."""

import json
import re
from pathlib import Path

from polaris_answer_extract import BOXED, extract_boxed, last_line, strip_math_delimiters
from polaris_final_text import final_answer_text
from verifyit.grade import InvalidTask
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
PROOF_END = re.compile(r"(?:This completes the proof\.|Q\.E\.D\.)\s*$", re.IGNORECASE)
DISPLAY = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
EXPLANATION = re.compile(r"\*{0,2}(?:justification|explanation|proof)\*{0,2}:\s*", re.IGNORECASE)
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?")
ANSWER_UPDATE = re.compile(
    r"\b(?:answer|result|maximum|minimum)(?:\s+(?:count|value|number))?\s+(?:is|equals)\s+",
    re.IGNORECASE,
)
SIGNED_CHANGE = re.compile(
    r"\b(?P<direction>decrease|increase)(?:s|d)?\s+by\s+"
    r"(?P<number>[+-]?(?:\d+(?:\.\d+)?|\.\d+))(?:\s+points?)?[.!]?\s*$",
    re.IGNORECASE,
)
FIND_VARIABLE = re.compile(r"\bfind\s+(?:\\\(\s*|\$\s*)([A-Za-z])\s*(?:\\\)|\$)", re.IGNORECASE)
ASSIGNMENTS = re.compile(r"\s*[A-Za-z]\s*=\s*[^,;\n]+(?:[,;]\s*[A-Za-z]\s*=\s*[^,;\n]+)*\s*")
CHOICE_LABEL = re.compile(r"\\textbf\{\s*\(([A-Z])\)\s*\}\s*")
CHOICE_PADDING = re.compile(r"(?:\\qquad|\\quad|\$|\s)+$")


def choice_values(problem: str) -> dict[str, str]:
    """Read the explicitly labelled choices without consulting the answer key."""
    labels = list(CHOICE_LABEL.finditer(problem))
    result = {}
    for index, label in enumerate(labels):
        end = labels[index + 1].start() if index + 1 < len(labels) else len(problem)
        value = CHOICE_PADDING.sub("", problem[label.end() : end]).strip()
        if label[1] in result:
            raise InvalidTask("Polaris question repeats a choice label")
        result[label[1]] = value.replace("{-}", "-")
    return result


def answer_reward(candidate: str, reference: str) -> float:
    """Compare categorical words literally and mathematical answers symbolically."""
    if reference.isalpha() and len(reference) > 1:
        return float(candidate.strip().casefold() == reference.casefold())
    return grade_math_candidate(MathSpec(expected=reference), candidate).reward


def math_reference(reference: str, choices: dict[str, str]) -> str:
    """Remove a presentation label only when it agrees with its declared choice."""
    label = CHOICE_LABEL.match(reference)
    if label is None:
        return reference
    value = reference[label.end() :].strip().replace("{-}", "-")
    if label[1] not in choices:
        raise InvalidTask("Polaris reference names an undeclared choice")
    if answer_reward(value, choices[label[1]]) != 1.0:
        raise InvalidTask("Polaris reference label and choice content disagree")
    return value


def final_candidate(answer: str, problem: str) -> str | None:
    """Read one final claim without consulting the reference or earlier reasoning."""
    final = final_answer_text(answer)
    targets = list(FIND_VARIABLE.finditer(problem))
    if targets and ASSIGNMENTS.fullmatch(final):
        assignments = dict(re.findall(r"([A-Za-z])\s*=\s*([^,;\n]+)", final))
        final = assignments.get(targets[-1][1], "")
    final = PROOF_END.sub("", final).strip()
    line = last_line(final)
    if not line:
        return None
    sentence = FINAL_SENTENCE.split(line)[-1]
    displays = list(DISPLAY.finditer(final))
    display = displays[-1] if displays and not final[displays[-1].end() :].strip() else None
    if display:
        sentence = display[1].strip()
    leading, separator, explanation = final.partition("\n\n")
    if separator and NUMBER.fullmatch(strip_math_delimiters(leading)) and EXPLANATION.match(explanation):
        updates = [value for value in FINAL_SENTENCE.split(line) if ANSWER_UPDATE.search(value)]
        sentence = updates[-1] if updates else leading
    # A direct opening answer may be followed by a proof containing other numbers.
    # Select its claim before mathematical comparison; the reference never chooses it.
    opening_sentences = FINAL_SENTENCE.split(final.splitlines()[0])
    opening_answers = [value for value in opening_sentences if OPENING_ANSWER.match(value)]
    if opening_answers and not FINAL_EXPRESSION.match(line) and not display:
        sentence = opening_answers[-1]
    if NEGATION.search(sentence):
        return None
    change = SIGNED_CHANGE.search(sentence)
    if change and re.search(r"\bchange\b", problem, re.IGNORECASE):
        number = change["number"]
        return f"-({number})" if change["direction"].lower().startswith("decrease") else number
    if BOXED in final:
        candidate = extract_boxed(final)
    else:
        claims = list(FINAL_CLAIM.finditer(sentence))
        candidate = sentence[claims[-1].end() :] if claims else sentence
    if not candidate:
        return None
    candidate = strip_math_delimiters(candidate.strip().rstrip(".!"))
    candidate = re.sub(r"^(?:exactly|precisely)\s+", "", candidate, flags=re.IGNORECASE)
    quantity = QUANTITY.fullmatch(candidate)
    if quantity:
        number, unit = quantity.groups()
        problem_units = UNITS.intersection(re.findall(r"\b[a-z]+\b", problem.lower()))
        if problem_units != {unit}:
            return None
        candidate = number
    return UNICODE_ROOT.sub(r"sqrt(\1)", candidate)


def score(answer: str, contract: dict[str, str]) -> float:
    choices = choice_values(contract["problem"])
    reference = contract["answer"]
    if BOXED in reference:
        reference = extract_boxed(reference)
        if reference is None:
            raise ValueError("Polaris reference contains an incomplete box")
    reference = math_reference(reference, choices)
    candidate = final_candidate(answer, contract["problem"]) or ""
    if candidate in choices:
        candidate = choices[candidate]
    return answer_reward(candidate, reference)


def main() -> None:
    print(score(Path("/app/answer.txt").read_text(), json.loads(Path("/tests/config.json").read_text())))


if __name__ == "__main__":
    main()
