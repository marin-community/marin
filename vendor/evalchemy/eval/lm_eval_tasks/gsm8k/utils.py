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
    rf"^\s*(?:\\?\$)?(?P<number>{_BARE_NUMBER})"
    r"(?:\s|[A-Za-z/.:_-]|%|\\[ ,;!%]|(?:\\text|\\mathrm|\\operatorname)\s*\{[^{}]*\})*$"
)
_LEADING_SCALAR = re.compile(rf"^\s*(?:\\?\$|€|£)?\s*(?P<number>{_BARE_NUMBER})")
_NUMERIC_TOKEN = re.compile(_BARE_NUMBER)
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
    annotated = _ANNOTATED_SCALAR.fullmatch(prediction)
    if annotated is not None:
        candidates.append(annotated.group("number"))

    scalar_text = prediction.replace("{,}", ",").replace(",\\!", ",").replace("\\,", "")
    numeric_tokens = _NUMERIC_TOKEN.findall(scalar_text)
    if len(numeric_tokens) == 1:
        candidates.append(numeric_tokens[0])
    leading = _LEADING_SCALAR.match(scalar_text)
    if leading is not None:
        candidates.append(leading.group("number"))

    if "=" in scalar_text:
        right_hand_tokens = _NUMERIC_TOKEN.findall(scalar_text.rsplit("=", 1)[-1])
        if len(right_hand_tokens) == 1:
            candidates.append(right_hand_tokens[0])

    arithmetic = _TEXT_BLOCK.sub("", scalar_text).replace("\\over", "/")
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
