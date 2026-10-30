"""GSM8K answer extraction and Minerva-equivalent scoring."""

import re

from eval.generation_stops import GSM8K_STOP_SEQUENCES, truncate_at_stop
from eval.graders.answer_extraction import extract_final_boxed_answer
from eval.graders.minerva_math import is_equiv, normalize_final_answer

_BARE_NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
_NUMBER = rf"\$?{_BARE_NUMBER}"
_FINAL_ANSWER = re.compile(rf"(?i)\b(?:final\s+answer|answer)\s*(?:is|:|=)?\s*({_NUMBER})")
_NUMERIC_CANDIDATE = re.compile(_NUMBER)
_ANNOTATED_SCALAR = re.compile(
    rf"^\s*(?:\\?\$|€|£)?\s*(?P<number>{_BARE_NUMBER})"
    r"(?P<unit>\s+[A-Za-z]+(?:\s+[A-Za-z]+)*|\s*%|/(?:day|week|month|year|hour|minute|second)s?)?"
    r"(?:\s*\(\s*=\s*(?:\d+(?:\.\d+)?\s+[A-Za-z]+\s*)+\))?\s*\.?\s*$"
)
_PROSE_SCALAR = re.compile(
    rf"^[A-Za-z][A-Za-z\s]*\s+(?P<number>{_BARE_NUMBER})\s+[A-Za-z][A-Za-z\s]*\.?$"
)
_CLOCK_ANSWER = re.compile(
    r"^\s*(?P<hour>\d{1,2}):(?P<minute>\d{2})(?:\s*(?:a\.?m\.?|p\.?m\.?))?\s*$",
    re.IGNORECASE,
)
_TEXT_ANNOTATION = re.compile(r"\\(?:text|textbf|mathrm|operatorname)\s*\{([^{}]*)\}")
_TEXT_BLOCK = re.compile(r"\\(?:text|textbf|mathrm|operatorname)\s*\{[^{}]*\}")
_FALLBACK = "[invalid]"


def extract_gsm8k_flexible_answer(response: str) -> str:
    """Prefer final-answer syntax before falling back to the last number."""
    response = truncate_at_stop(response, GSM8K_STOP_SEQUENCES)
    boxed = extract_final_boxed_answer(response)
    if boxed:
        return boxed.strip()
    for pattern in (_FINAL_ANSWER, _NUMERIC_CANDIDATE):
        matches = pattern.findall(response)
        if matches:
            return matches[-1].strip()
    return _FALLBACK


def _gsm8k_equivalence_candidates(prediction: str, question: str) -> list[str]:
    candidates = [prediction]
    scalar_text = prediction.replace("{,}", ",").replace(",\\!", ",").replace("\\,", "")
    scalar_text = _TEXT_ANNOTATION.sub(r" \1 ", scalar_text)
    scalar_text = scalar_text.replace(r"^\circ", " degrees ").replace(r"\%", "%")
    scalar_text = re.sub(r"\\[ ,;!]", " ", scalar_text).strip()
    scalar = _ANNOTATED_SCALAR.fullmatch(scalar_text)
    if scalar is not None:
        candidates.append(scalar.group("number"))
    prose = _PROSE_SCALAR.fullmatch(scalar_text)
    if prose is not None:
        candidates.append(prose.group("number"))

    if "=" in scalar_text:
        right_hand = _ANNOTATED_SCALAR.fullmatch(scalar_text.rsplit("=", 1)[-1].strip())
        if right_hand is not None:
            candidates.append(right_hand.group("number"))

    arithmetic = _TEXT_BLOCK.sub("", prediction).replace("\\over", "/")
    if not re.search(r"[A-Za-z]", arithmetic) and re.search(r"[+*/]", arithmetic):
        candidates.append(arithmetic)

    if "time" in question.lower():
        clock_text = _TEXT_ANNOTATION.sub(r"\1", prediction).replace("\\,", " ").replace("\\ ", " ")
        clock = _CLOCK_ANSWER.fullmatch(clock_text)
        if clock is not None and clock.group("minute") == "00":
            candidates.append(str(int(clock.group("hour"))))
    return list(dict.fromkeys(candidates))


def gsm8k_flexible_extraction_filter(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    """Extract one auditable flexible answer per GSM8K completion."""
    del docs
    return [
        [extract_gsm8k_flexible_answer(response) if isinstance(response, str) else _FALLBACK for response in responses]
        for responses in resps
    ]


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    """Score an extracted GSM8K answer with Minerva math equivalence."""
    prediction = results[0]
    if prediction == _FALLBACK:
        return {"exact_match": 0.0}

    reference = doc["answer"].rsplit("#### ", 1)[-1]
    reference = normalize_final_answer(reference)
    candidates = [
        normalize_final_answer(candidate)
        for candidate in _gsm8k_equivalence_candidates(prediction, doc.get("question", ""))
    ]
    equivalent = any(is_equiv(candidate, reference) for candidate in candidates)
    question = doc.get("question", "").lower()
    if not equivalent and ("percent" in question or "pass rate" in question):
        percent_reference = rf"\frac{{{reference}}}{{100}}"
        equivalent = any(is_equiv(candidate, percent_reference) for candidate in candidates)
    return {"exact_match": float(equivalent)}
