"""AIME answer extraction and grading.

Verbatim port of `src/aime_grader.py` from
https://github.com/Suzehva/emergent-doordash (branch `fresh/main`). Logic is
unchanged: extract a boxed/tagged/final answer, interpret it as an integer via a
small LaTeX->sympy pass, and require both `interpreted == gold` and that the gold
integer literally appears in the extracted text.

Used by `aime_prompt_playground.ipynb`.
"""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
import re
from dataclasses import dataclass
from tokenize import TokenError

import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


AIME_GRADER_TYPE = "aime_explicit_answer_sympy_v1"
MAX_INTERPRETED_ABS_VALUE = 999_999
SYMPY_TIMEOUT_SECONDS = 1


@dataclass(frozen=True)
class AIMEGrade:
    extracted_answer: str | None
    is_correct: bool
    method: str
    interpreted_answer: str | None
    source: str | None

    @property
    def metadata(self) -> dict[str, str | None]:
        return {
            "grader_type": AIME_GRADER_TYPE,
            "method": self.method,
            "interpreted_answer": self.interpreted_answer,
            "source": self.source,
            "requires_explicit_gold_integer": "true",
        }


def extract_aime_answer(response_text: str) -> str | None:
    candidate, _method = _extract_candidate(response_text)
    return candidate


def is_aime_correct(extracted_answer: str | None, gold_answer: str) -> bool:
    gold = _gold_integer(gold_answer)
    if extracted_answer is None or gold is None:
        return False
    interpreted = _candidate_final_integer(extracted_answer)
    return interpreted == gold and _contains_integer_value(extracted_answer, gold)


def grade_aime_response(response_text: str, gold_answer: str) -> AIMEGrade:
    extracted, method = _extract_candidate(response_text)
    gold = _gold_integer(gold_answer)
    interpreted = None if extracted is None else _candidate_final_integer(extracted)
    is_correct = (
        extracted is not None
        and gold is not None
        and interpreted == gold
        and _contains_integer_value(extracted, gold)
    )
    return AIMEGrade(
        extracted_answer=extracted,
        is_correct=is_correct,
        method=method,
        interpreted_answer=None if interpreted is None else str(interpreted),
        source=extracted,
    )


def _extract_candidate(response_text: str) -> tuple[str | None, str]:
    boxed_fallback: str | None = None
    for candidate in reversed(_boxed_candidates(response_text)):
        cleaned = _clean_candidate(candidate)
        if not cleaned:
            continue
        if _candidate_has_answer_shape(cleaned):
            return cleaned, "boxed"
        if boxed_fallback is None:
            boxed_fallback = cleaned

    for candidate in reversed(_tagged_candidates(response_text)):
        cleaned = _clean_candidate(candidate)
        if cleaned:
            return cleaned, "answer_tag"

    for candidate in reversed(_boxed_context_candidates(response_text)):
        cleaned = _clean_candidate(candidate)
        if _candidate_has_answer_shape(cleaned):
            return cleaned, "boxed_context"
        if cleaned and boxed_fallback is None:
            boxed_fallback = cleaned

    for candidate in _final_math_candidates(response_text):
        cleaned = _clean_candidate(candidate)
        if _candidate_has_answer_shape(cleaned):
            return cleaned, "final_math"

    if boxed_fallback is not None:
        return boxed_fallback, "boxed"

    return None, "no_answer"


def _tagged_candidates(text: str) -> list[str]:
    return [
        match.group(1)
        for match in re.finditer(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    ]


def _boxed_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        parsed = _balanced_brace_content(text, match.end() - 1)
        if parsed is not None:
            candidates.append(parsed[0])
    return candidates


def _boxed_context_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for span in _math_spans(text):
        if "\\boxed" not in span and "\\fbox" not in span:
            continue
        if not _is_compact_math_context(span):
            continue
        boxed = _boxed_candidates(span)
        if len(boxed) != 1:
            continue
        stripped = span.strip()
        if re.fullmatch(r"\\(?:boxed|fbox)\s*\{.*\}", stripped, flags=re.DOTALL):
            continue
        candidates.append(stripped)
    return candidates


def _is_compact_math_context(span: str) -> bool:
    stripped = span.strip()
    if "\n" in stripped or len(stripped) > 240:
        return False
    if re.search(r"</?think>|```", stripped, flags=re.IGNORECASE):
        return False
    return True


def _balanced_brace_content(text: str, open_brace: int) -> tuple[str, int] | None:
    depth = 0
    for idx in range(open_brace, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : idx], idx + 1
    return None


def _final_math_candidates(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tail = "\n".join(lines[-6:])
    candidates = list(reversed(_math_spans(tail)))
    for line in reversed(lines[-4:]):
        if "=" in line and not re.search(r"^[A-Za-z ]+:$", line):
            candidates.append(line)
    return candidates


def _math_spans(text: str) -> list[str]:
    spans: list[str] = []
    for match in re.finditer(r"\$\$(.*?)\$\$|\$(.*?)\$", text, flags=re.DOTALL):
        spans.append((match.group(1) if match.group(1) is not None else match.group(2)).strip())
    return spans


def _candidate_has_answer_shape(candidate: str) -> bool:
    if _looks_like_code_candidate(candidate):
        return False
    return _candidate_final_integer(candidate) is not None


def _candidate_final_integer(candidate: str) -> int | None:
    if _looks_like_code_candidate(candidate):
        return None
    normalized = candidate.replace("≡", r"\equiv")
    pieces = re.split(r"\\equiv|=", normalized)
    if len(pieces) > 1:
        for piece in reversed(pieces):
            value = _parse_integer_expression(piece)
            if value is not None:
                return value
            tokens = _integer_tokens(piece)
            if tokens:
                return tokens[-1]
        return None

    tokens = _integer_tokens(normalized)
    if _looks_like_plain_integer(normalized) and tokens:
        return tokens[-1]
    if len(tokens) == 1 and not _contains_arithmetic_operator(normalized):
        return tokens[-1]
    return None


def _contains_integer_value(text: str, value: int) -> bool:
    return any(token == value for token in _integer_tokens(text))


def _integer_tokens(text: str) -> list[int]:
    out: list[int] = []
    for match in re.finditer(r"(?<![A-Za-z0-9_.])-?\d+(?![A-Za-z0-9_.])", text):
        try:
            out.append(int(match.group(0)))
        except ValueError:
            continue
    return out


def _gold_integer(gold_answer: str) -> int | None:
    cleaned = _clean_candidate(gold_answer)
    value = _parse_integer_expression(cleaned)
    if value is not None:
        return value
    tokens = _integer_tokens(cleaned)
    if len(tokens) == 1:
        return tokens[0]
    return None


def _parse_integer_expression(text: str) -> int | None:
    expression = _latex_to_sympy_text(text)
    if expression is None:
        return None
    try:
        with _sympy_timeout(SYMPY_TIMEOUT_SECONDS):
            parsed = parse_expr(expression, local_dict={"sqrt": sp.sqrt}, evaluate=False)
    except (
        AttributeError,
        IndexError,
        RecursionError,
        _SympyTimeout,
        TokenError,
        sp.SympifyError,
        TypeError,
        ValueError,
        SyntaxError,
    ):
        return None
    if not isinstance(parsed, sp.Expr) or _has_large_power(parsed):
        return None
    try:
        with _sympy_timeout(SYMPY_TIMEOUT_SECONDS):
            simplified = sp.simplify(parsed)
    except (AttributeError, RecursionError, _SympyTimeout, TypeError, ValueError, ZeroDivisionError):
        return None
    if simplified.is_integer is not True or simplified.is_number is not True:
        return None
    try:
        value = int(simplified)
    except (TypeError, ValueError, OverflowError):
        return None
    if sp.Integer(value) != simplified:
        return None
    if abs(value) > MAX_INTERPRETED_ABS_VALUE:
        return None
    return value


class _SympyTimeout(Exception):
    pass


def _sympy_timeout_handler(signum: int, frame: object) -> None:
    raise _SympyTimeout()


@contextmanager
def _sympy_timeout(seconds: float):
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, _sympy_timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _has_large_power(expr: sp.Expr) -> bool:
    for power in expr.atoms(sp.Pow):
        exponent = power.exp
        if exponent.is_number is not True:
            continue
        try:
            numeric_exponent = int(exponent)
        except (TypeError, ValueError, OverflowError):
            return True
        if numeric_exponent > 12:
            return True
    return False


def _latex_to_sympy_text(text: str) -> str | None:
    value = _clean_candidate(text)
    if not value:
        return None
    if len(value) > 300 or _looks_like_code_candidate(value):
        return None
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = value.replace("×", "*").replace("÷", "/")
    value = re.sub(r"(?<=\d),(?=\d)", "", value)
    value = re.sub(r"\\text\s*\{[^{}]*\}", "", value)
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    value = value.replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    value = value.replace("^{\\circ}", "").replace("^\\circ", "").replace("°", "")
    value = _replace_latex_frac(value)
    value = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", value)
    value = re.sub(r"\\sqrt\s+([A-Za-z0-9]+)", r"sqrt(\1)", value)
    value = value.replace("{", "(").replace("}", ")")
    value = value.replace("^", "**")
    value = value.replace("$", "")
    value = value.strip()
    if "\\" in value:
        return None
    return value


def _replace_latex_frac(text: str) -> str:
    pattern = re.compile(r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    previous = None
    value = text
    while previous != value:
        previous = value
        value = pattern.sub(r"((\1)/(\2))", value)
    return value


def _clean_candidate(text: str) -> str:
    value = text.strip()
    value = value.strip(" \t\n\r`*_")
    while True:
        updated = value.strip()
        if updated.startswith("$$") and updated.endswith("$$") and len(updated) >= 4:
            updated = updated[2:-2].strip()
        elif updated.startswith("$") and updated.endswith("$") and len(updated) >= 2:
            updated = updated[1:-1].strip()
        boxed = re.fullmatch(r"\\(?:boxed|fbox)\s*\{(.*)\}", updated, flags=re.DOTALL)
        if boxed is not None:
            updated = boxed.group(1).strip()
        updated = updated.strip(" .,:;")
        if updated == value:
            return updated
        value = updated


def _looks_like_plain_integer(text: str) -> bool:
    return re.fullmatch(r"\s*-?\d+\s*", _clean_candidate(text)) is not None


def _contains_arithmetic_operator(text: str) -> bool:
    return bool(re.search(r"[+*/^]|\s-\s|\\(?:frac|cdot|times|div)", text))


def _looks_like_code_candidate(text: str) -> bool:
    return bool(
        re.search(
            r"```|\bdef\s+\w+\s*\(|\binput\s*\(|\bprint\s*\(|__name__|"
            r"\breturn\b|\bfor\s+\w+\s+in\b|\bif\s+.+:",
            text,
        )
    )
