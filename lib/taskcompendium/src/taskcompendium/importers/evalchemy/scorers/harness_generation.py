# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Native short-answer protocols from Marin's Evalchemy overrides and lm-eval v0.4.12 (MIT)."""

import re
import string
from collections.abc import Mapping
from typing import Any

from verifyit.adapters.completion import END_OF_TURN
from verifyit.grade import InvalidTask, Reward, scored
from verifyit.modes.grade_exact import grade_collection_f1, grade_normalized_aliases

INVALID_ANSWER = "[invalid]"
GSM8K_STOPS = tuple("Question:" if stop == "\nQuestion:" else stop for stop in END_OF_TURN)
COMPLETION_REQUEST_STOPS = ("\nYou are an AI assistant", "\nQuestion:", "\nQ:", "\n[Question]")
GSM8K_REQUEST_STOPS = ("\nYou are an AI assistant", "Question:", "\nQ:", "\n[Question]")
_NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?"
_GSM_FLEXIBLE_PATTERNS = (
    re.compile(rf"\\boxed\{{\s*({_NUMBER})\s*\}}"),
    re.compile(rf"(?i)\b(?:final\s+answer|answer)\s*(?:is|:|=)?\s*({_NUMBER})"),
    re.compile(_NUMBER),
)
_ANSWER_MARKER = re.compile(r"(?im)^[ \t]*(?:(?:final)[ \t]+)?(?:answer|a)[ \t]*:[ \t]*(?P<answer>\S[^\r\n]*)")
_EXPLICIT_ANSWER = re.compile(r"(?im)\b(?:the[ \t]+)?(?:final[ \t]+)?answer[ \t]+is[ \t]+(?P<answer>\S[^\r\n]*)")
_LEADING_ANSWER_MARKER = re.compile(r"(?i)^(?:(?:final)[ \t]+)?(?:answer|a)[ \t]*:[ \t]*")
_DROP_NUMBER = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DROP_DATE = re.compile(rf"\b(?:(\d{{1,2}})\s+)?({_MONTHS})\.?(?:\s+(\d{{3,4}}))?\b", re.IGNORECASE)
_DROP_MARKER = re.compile(r"(?i)(?:final\s+answer|answer)\s*(?:is|:|=|-|—)?\s*")
_DROP_ENTITY = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def truncate_response(response: str, stops: tuple[str, ...] = END_OF_TURN) -> str:
    indices = [index for stop in stops if (index := response.find(stop)) >= 0]
    return response[: min(indices)] if indices else response


def _clean_short_answer(answer: str) -> str:
    answer = answer.strip().rstrip(".").strip()
    while marker := _LEADING_ANSWER_MARKER.match(answer):
        answer = answer[marker.end() :].strip()
    if len(answer) >= 2 and (answer[0], answer[-1]) in {("(", ")"), ("[", "]")}:
        answer = answer[1:-1].strip()
    return answer.rstrip(".").strip()


def extract_short_answer(response: str, *, marked: bool) -> str:
    response = truncate_response(response)
    signals = [(match.start(), match["answer"]) for match in _ANSWER_MARKER.finditer(response)]
    if not marked:
        signals.extend((match.start(), match["answer"]) for match in _EXPLICIT_ANSWER.finditer(response))
    if signals:
        answer = _clean_short_answer(max(signals, key=lambda signal: signal[0])[1])
        # The strict source filter preserves an explicitly marked empty extraction.
        return answer if marked else answer or INVALID_ANSWER
    if marked:
        return INVALID_ANSWER
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    answer = lines[-1] if lines else ""
    if len(answer) > 256 or answer.startswith("<") or _LEADING_ANSWER_MARKER.fullmatch(answer):
        return INVALID_ANSWER
    return _clean_short_answer(answer) or INVALID_ANSWER


def extract_drop_answer(response: str) -> str:
    response = response.strip()
    if not response:
        return ""
    markers = list(_DROP_MARKER.finditer(response))
    if markers:
        tail = response[markers[-1].end() :]
        line = tail.splitlines()[0].strip() if tail.strip() else ""
        if line:
            date = _DROP_DATE.search(line)
            numbers = _DROP_NUMBER.findall(line)
            if date:
                candidate = " ".join(part for part in date.groups() if part)
            elif numbers:
                candidate = numbers[-1].replace("$", "").replace(",", "").rstrip("%").strip()
            else:
                candidate = line
            return candidate.rstrip(".").strip() or candidate
    entities = _DROP_ENTITY.findall(response)
    return entities[-1] if entities else response


def drop_references(doc: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    """Preserve source answer priority, validated aliases, span grouping, and deduplication."""
    validated = doc["validated_answers"]
    alternatives = [
        doc["answer"],
        *(
            {key: validated[key][index] for key in ("number", "date", "spans")}
            for index in range(len(validated["number"]))
        ),
    ]
    references = []
    for answer in alternatives:
        spans = (
            (str(answer["number"]),)
            if answer["number"] != ""
            else (
                tuple(answer["spans"])
                if answer["spans"]
                else (" ".join(answer["date"][key] for key in ("day", "month", "year")).strip(),)
            )
        )
        if spans not in references:
            references.append(spans)
    return tuple(references)


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _normalize_drop(answer: str) -> str:
    tokens = []
    for value in re.split(" |-", answer):
        value = value.lower()
        value = (
            value
            if _is_number(value)
            else "".join(character for character in value if character not in string.punctuation)
        )
        value = str(float(value)) if _is_number(value) else value
        value = " ".join(_ARTICLES.sub(" ", value).split())
        if value.strip():
            tokens.append(value)
    return " ".join(tokens).strip()


def _drop_metrics(candidate: str, references: tuple[tuple[str, ...], ...]) -> dict[str, float]:
    normalized = _normalize_drop(candidate)
    predicted = set(normalized.split())
    predicted_numbers = {token for token in predicted if _is_number(token)}
    best_em = best_f1 = 0.0
    for reference in references:
        if not reference[0].strip():
            continue
        normalized_spans = [_normalize_drop(span) for span in reference]
        best_em = max(best_em, float(set(normalized_spans) == {normalized} and len(normalized_spans) == 1))
        scores = []
        for span in normalized_spans:
            gold = set(span.split())
            gold_numbers = {token for token in gold if _is_number(token)}
            f1 = (
                1.0
                if not predicted and not gold
                else grade_collection_f1(
                    gold, predicted, multiplicity="set", empty_reference="zero", round_digits=None
                ).reward
            )
            scores.append(f1 if not gold_numbers or gold_numbers & predicted_numbers else 0.0)
        # The source predicts one span: optimal one-to-one alignment leaves the
        # other gold spans unmatched, and rounds their mean to two decimal places.
        best_f1 = max(best_f1, round(max(scores) / len(reference) * 100) / 100)
    return {"em,extract_answer": best_em, "f1,extract_answer": best_f1}


def validate_harness_generation(profile: str, references: tuple[tuple[str, ...], ...]) -> None:
    if (
        profile not in {"gsm8k", "triviaqa", "nq_open", "drop"}
        or not references
        or any(not spans or any(not isinstance(span, str) for span in spans) for spans in references)
    ):
        raise InvalidTask("Harness generation requires a known profile and answer alternatives")
    if profile != "drop" and any(len(spans) != 1 for spans in references):
        raise InvalidTask("Only DROP accepts multi-span references")


def grade_harness_generation_candidate(profile: str, references: tuple[tuple[str, ...], ...], response: str) -> Reward:
    """Grade all source extraction variants and retain each named metric and extracted answer."""
    validate_harness_generation(profile, references)
    if profile == "drop":
        extracted = {"extract_answer": extract_drop_answer(response)}
        metrics = _drop_metrics(extracted["extract_answer"], references)
        primary = "f1,extract_answer"
    elif profile == "gsm8k":
        from .minerva import is_equiv, normalize_final_answer  # noqa: PLC0415

        strict = re.findall(r"#### (\-?[0-9\.\,]+)", response)
        flexible = truncate_response(response, GSM8K_STOPS)
        answer = INVALID_ANSWER
        for pattern in _GSM_FLEXIBLE_PATTERNS:
            if matches := pattern.findall(flexible):
                answer = matches[-1].strip()
                break
        extracted = {"strict-match": strict[0] if strict else INVALID_ANSWER, "flexible-extract": answer}
        reference = normalize_final_answer(references[0][0])
        metrics = {
            f"exact_match,{name}": float(
                candidate != INVALID_ANSWER and is_equiv(normalize_final_answer(candidate), reference)
            )
            for name, candidate in extracted.items()
        }
        primary = "exact_match,strict-match"
    else:
        extracted = {
            "strict_answer": extract_short_answer(response, marked=True),
            "extract_answer": extract_short_answer(response, marked=False),
        }
        answers = [spans[0] for spans in references]
        regexes = (r"\b(?:The |the |An |A |The |a |an )",) if profile == "nq_open" else ()
        metrics = {
            f"exact_match,{name}": (
                grade_normalized_aliases(
                    candidate, answers, ignore_case=True, ignore_punctuation=True, regexes_to_ignore=regexes
                ).reward
            )
            for name, candidate in extracted.items()
        }
        primary = "exact_match,strict_answer"
    return scored(metrics[primary], metrics=metrics, extracted=extracted)
