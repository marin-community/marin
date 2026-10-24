# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiple-choice reward ported from NVIDIA NeMo Gym."""

from __future__ import annotations

import re
from typing import Any

from skyrl_gym.envs.nemotron_ultra.answer_extraction import last_boxed_answer

LATEX_TEXT_WRAP_PATTERN = re.compile(r"\\text\{\s*(.*?)\s*\}", re.S)
ANSWER_COLON_PATTERN = re.compile(r"(?i)answer\s*:\s*(.+)")
ANSWER_COLON_MD_PATTERN = re.compile(r"(?i)[*_]{0,2}Answer[*_]{0,2}\s*:[*_\s]{0,2}\s*([A-Z])(?![a-zA-Z0-9])")


def _letters(options: list[dict[str, str]] | None) -> set[str]:
    return {
        key.upper()
        for option in options or []
        for key, value in option.items()
        if isinstance(key, str) and len(key) == 1 and key.isalpha() and value is not None
    }


def _strip_latex(value: str) -> str:
    while match := LATEX_TEXT_WRAP_PATTERN.fullmatch(value):
        value = match.group(1)
    return value


def _normalize(value: str) -> str:
    return " ".join(value.lower().split())


def _normalize_extracted(value: str) -> str:
    return (
        value.replace("أ", " A")
        .replace("ب", " B")
        .replace("ج", " C")
        .replace("د", " D")
        .replace("অ", " A")
        .replace("ব", " B")
        .replace("ড", " C")
        .replace("ঢ", " D")
        .replace("Ａ", " A")
        .replace("Ｂ", " B")
        .replace("Ｃ", " C")
        .replace("Ｄ", " D")
        .strip()
    )


def _strict_boxed(text: str, allowed: set[str]) -> str | None:
    boxed = last_boxed_answer(text)
    match = None if boxed is None else re.fullmatch(r"\s*[^A-Za-z]*([A-Z])[^A-Za-z]*\s*", boxed)
    if not match:
        return None
    letter = match.group(1).upper()
    return letter if letter in allowed else None


def _option_text(text: str, options: list[dict[str, str]] | None, allowed: set[str]) -> str | None:
    boxed = last_boxed_answer(text)
    if boxed is None:
        return None
    candidates = {_normalize(boxed), _normalize(_strip_latex(boxed))}
    matches = {
        key.upper()
        for option in options or []
        for key, value in option.items()
        if value is not None and key.upper() in allowed and _normalize(value) in candidates
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _custom_regex(
    text: str, patterns: str | list[str], allowed: set[str], options: list[dict[str, str]] | None
) -> str | None:
    for pattern in [patterns] if isinstance(patterns, str) else patterns:
        try:
            matches = re.findall(pattern, text, re.IGNORECASE)
        except re.error:
            continue
        if not matches:
            continue
        value = matches[-1]
        if isinstance(value, tuple):
            captures = [capture for capture in value if capture]
            if len(captures) != 1:
                raise ValueError("MCQA output_regex must have one unambiguous answer capture")
            value = captures[0]
        captured = _normalize_extracted(value.strip()).upper()
        if len(captured) == 1 and captured.isalpha():
            return captured
        for option in options or []:
            for key, value in option.items():
                if value is not None and key.upper() in allowed and _normalize(value) == _normalize(captured):
                    return key.upper()
    return None


def grade_mcqa(text: str, record: dict[str, Any], *, verifyit_enabled: bool = False) -> tuple[float, dict[str, Any]]:
    text = text.strip()
    options = record.get("options")
    gold = str(record.get("expected_answer") or "").strip().upper()
    allowed = _letters(options)
    if not text and not verifyit_enabled:
        return 0.0, {"expected_answer": gold, "extracted_answer": None}
    prediction = None
    template = record.get("template_metadata")
    if template and "output_regex" in template:
        prediction = _custom_regex(text, template["output_regex"], allowed, options)
    mode = record.get("grading_mode", "strict_single_letter_boxed")
    if prediction is None and mode == "strict_single_letter_boxed":
        prediction = _strict_boxed(text, allowed)
    elif prediction is None and mode == "lenient_boxed":
        prediction = _strict_boxed(text, allowed) or _option_text(text, options, allowed)
    elif prediction is None and mode == "lenient_answer_colon":
        if match := ANSWER_COLON_PATTERN.search(text):
            candidate = _strip_latex(match.group(1)).strip()
            if len(candidate) == 1 and candidate.upper() in allowed:
                prediction = candidate.upper()
            else:
                normalized = _normalize(candidate)
                for option in options or []:
                    for key, value in option.items():
                        if value is not None and key.upper() in allowed and _normalize(value) == normalized:
                            prediction = key.upper()
                            break
    elif prediction is None and mode == "lenient_answer_colon_md":
        if match := ANSWER_COLON_MD_PATTERN.search(text):
            candidate = match.group(1).upper()
            prediction = candidate if candidate in allowed else None
    if verifyit_enabled:
        from skyrl_gym.envs.verifyit_clients import grade_mcqa_option

        reward = grade_mcqa_option(gold, prediction, allowed)
    else:
        reward = float(bool(prediction and gold and prediction == gold))
    return reward, {"expected_answer": gold, "extracted_answer": prediction}
