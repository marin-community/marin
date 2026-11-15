# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace as MCQAVerifyResponse
from types import SimpleNamespace as MCQADualVerifyResponse
import re
import unicodedata
from typing import Any, Optional


def _extract_options_and_expected(
    body: MCQARunRequest,
) -> tuple[Optional[list[dict[str, str]]], Optional[str]]:
    # Only use explicit top-level fields for grading; ignore metadata entirely.
    return body.options, body.expected_answer  # type: ignore[return-value]

CHOICE_LETTER_PATTERN = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")

BOXED_LETTER_ONLY_PATTERN = re.compile(r"^[^A-Za-z]*([A-Z])[^A-Za-z]*$")

BOXED_LETTER_LABEL_PATTERN = re.compile(r"^\s*([A-Z])\s*[:).\-]")

ANSWER_COLON_PATTERN = re.compile(r"(?i)answer\s*:\s*(.+)")

ANSWER_COLON_PAYLOAD_PATTERN = re.compile(r"(?im)[*_]{0,2}Answer[*_]{0,2}\s*:[*_\s]{0,2}(.+)")

MILU_V2_ANSWER_PATTERN = re.compile(
    r"(?i)(?:\\boxed\s*\{\s*\*?\s*([ABCD])\s*\*?\s*\}|answer is\s*\*?\(?\*?([ABCD])\*?\)?\*?)"
)

LOCALIZED_ANSWER_LABELS = frozenset(
    {"উত্তৰ", "উত্তর", "જવાબ", "उत्तर", "ಉತ್ತರ", "ഉത്തരം", "ଉତ୍ତର", "ਉੱਤਰ", "उत्तरम्", "उत्तरं", "பதில்", "விடை", "సమాధానం", "جواب"}
)

TERMINAL_LABELED_CHOICE_PATTERN = re.compile(
    r"^[\s*_`#~-]*(?P<label>[^:：\n]{2,80})[:：]"
    r"[\s*_`#~\[({-]*(?P<choice>[A-Za-z])"
    r"[\s*_`#~\])}.।。!?,;:：-]*$"
)

def _parse_answer_letter_strict_boxed(text: str, allowed_letters: set[str]) -> tuple[Optional[str], str, bool]:
    """Pull the answer letter out of the last \\boxed{...}.

    Handles a bare letter, a \\text{...} wrapper (\\boxed{\\text{E}}), and a
    leading letter with option text after it (\\boxed{E: ...}). Plain option
    text with no leading letter is left for lenient_boxed to handle.
    """
    parsed_text = text
    inner = _extract_boxed_inner(text)
    if inner is None:
        return None, parsed_text, True
    inner = _strip_latex_wrappers(inner.strip())

    letter: Optional[str] = None
    m = BOXED_LETTER_ONLY_PATTERN.match(inner)
    if m:
        letter = m.group(1).upper()
    else:
        m = BOXED_LETTER_LABEL_PATTERN.match(inner)
        if m:
            letter = m.group(1).upper()

    if letter is None or letter not in allowed_letters:
        return None, parsed_text, True
    return letter, parsed_text, False

LATEX_TEXT_WRAP_PATTERN = re.compile(r"\\text\{\s*(.*?)\s*\}", re.S)

LEADING_CHOICE_LETTER_PATTERN = re.compile(r"^([a-zA-Z])(?![a-zA-Z0-9])")

LAST_STANDALONE_CHOICE_LETTER_PATTERN = re.compile(r"\b[A-Z]\b(?!.*\b[A-Z]\b)", re.S)

CHOICE_LIST_SEPARATOR_PATTERN = re.compile(r"(?:\s*[/|&,;]\s*|\s+(?:or|and)\s+|\s+)", re.I)

def _extract_boxed_inner(text: str) -> Optional[str]:
    """Return everything inside the last \\boxed{...}, or None.

    Uses the LAST \\boxed{ so CoT intermediate boxes are skipped and the
    final answer wins (matching the convention in other resources servers,
    e.g. math_with_code, abstention). Counts braces so a nested wrapper like
    \\boxed{\\text{E}} returns the full "\\text{E}" instead of stopping early at
    the first "}".
    """
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    depth = 1
    buf: list[str] = []
    for ch in text[start + len(marker) :]:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(buf)
        buf.append(ch)
    return None  # no matching closing brace

def _strip_latex_wrappers(s: str) -> str:
    """Remove successive \\text{...} wrappers from a LaTeX string."""
    while True:
        m = LATEX_TEXT_WRAP_PATTERN.fullmatch(s)
        if not m:
            break
        s = m.group(1)
    return s

def _normalize_for_match(s: str) -> str:
    """Lowercase and collapse whitespace for robust substring/equality checks."""
    return " ".join(s.lower().split())

def _contains_ambiguous_choice_list(text: str, allowed_letters: set[str]) -> bool:
    """Return whether distinct allowed choices are presented as a list."""
    matches = [
        (match.group(1).upper(), match.span(1))
        for match in CHOICE_LETTER_PATTERN.finditer(text)
        if match.group(1).upper() in allowed_letters
    ]
    return any(
        left_letter != right_letter
        and CHOICE_LIST_SEPARATOR_PATTERN.fullmatch(text[left_span[1] : right_span[0]]) is not None
        for (left_letter, left_span), (right_letter, right_span) in zip(matches, matches[1:])
    )

def _extract_choice_letter(text: str, allowed_letters: set[str]) -> Optional[str]:
    """Extract a single valid choice letter from an answer payload."""
    candidate = _normalize_extracted_answer(text).strip().strip("*_` \t\n\r.,;:")
    if _contains_ambiguous_choice_list(candidate, allowed_letters):
        return None

    if len(candidate) == 1:
        letter = candidate
    else:
        # Prefer a leading letter for "Answer: B because..." style payloads
        m = LEADING_CHOICE_LETTER_PATTERN.match(candidate)
        if m:
            letter = m.group(1)
        else:
            m = LAST_STANDALONE_CHOICE_LETTER_PATTERN.search(candidate)
            letter = m.group(0) if m else None
    if letter:
        up = letter.upper()
        if up in allowed_letters:
            return up
    return None

def _match_option_text(text: str, options: list[dict[str, str]], allowed_letters: set[str]) -> Optional[str]:
    """Match boxed content against option texts and return the option letter.

    - Looks ONLY inside the last \boxed{...} region; returns None if absent.
    - Normalizes (lowercase, collapse whitespace) both boxed content and option texts.
    - Treats a match as substring containment of an option's text in the boxed content.
    - Returns the option letter only if EXACTLY ONE option matches; otherwise returns None.
    """
    # Only match within boxed content; if no boxed content, return None
    inner = _extract_boxed_inner(text)
    if inner is None:
        return None
    inner = inner.strip()
    candidate_texts = [inner, _strip_latex_wrappers(inner)]
    normalized_candidates = [_normalize_for_match(t) for t in candidate_texts]

    # Build list of (letter, normalized_option_text)
    normalized_options: list[tuple[str, str]] = []
    for entry in options or []:
        for k, v in entry.items():
            # Skip null values and only include valid letter keys with string values
            if v is not None and isinstance(k, str) and len(k) == 1 and k.upper() in allowed_letters:
                normalized_options.append((k.upper(), _normalize_for_match(v)))

    matched_letters: set[str] = set()
    for cand in normalized_candidates:
        for letter, opt_norm in normalized_options:
            if opt_norm and opt_norm in cand:
                matched_letters.add(letter)
    if len(matched_letters) == 1:
        return next(iter(matched_letters))
    return None

def _answer_payloads(text: str, answer_prefix: Optional[str]) -> list[str]:
    """Collect final-answer payloads after Answer: or a row-provided prefix."""
    text = unicodedata.normalize("NFKC", text)
    found: list[tuple[int, str]] = []
    for match in ANSWER_COLON_PAYLOAD_PATTERN.finditer(text):
        found.append((match.start(), match.group(1)))
    if answer_prefix:
        normalized_prefix = unicodedata.normalize("NFKC", answer_prefix)
        pattern = re.compile(re.escape(normalized_prefix) + r"\s*", re.IGNORECASE)
        for match in pattern.finditer(text):
            line_end = text.find("\n", match.end())
            if line_end < 0:
                line_end = len(text)
            found.append((match.start(), text[match.end() : line_end]))
    found.sort(key=lambda item: item[0])
    return [payload for _start, payload in found]

def _extract_from_answer_payloads(
    text: str, allowed_letters: set[str], answer_prefix: Optional[str] = None
) -> Optional[str]:
    pred = None
    for candidate in reversed(_answer_payloads(text, answer_prefix)):
        if pred is None:
            pred = _extract_choice_letter(candidate, allowed_letters)
    return pred

def _extract_terminal_labeled_choice(text: str, allowed_letters: set[str]) -> Optional[str]:
    """Accept a recognized Indic label and one choice on the final nonempty line."""
    line = next((line.strip() for line in reversed(text.splitlines()) if line.strip()), "")
    match = TERMINAL_LABELED_CHOICE_PATTERN.fullmatch(unicodedata.normalize("NFKC", line))
    if not match:
        return None
    label = match.group("label").strip("*_`#~ ")
    if label not in LOCALIZED_ANSWER_LABELS:
        return None
    choice = match.group("choice").upper()
    return choice if choice in allowed_letters else None

def _normalize_extracted_answer(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return (
        text.replace("أ", " A")
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

def _parse_answer_with_custom_regex(
    text: str, regex_pattern: str, allowed_letters: set[str], options: Optional[list[dict[str, str]]]
) -> Optional[str]:
    """Parse answer using custom regex from template_metadata.

    Uses rightmost (last) match to handle reasoning before final answer.
    Case-insensitive matching to handle capitalization variations.

    When using template_metadata with custom regex, we trust the regex pattern
    and allow extracted letters even if options metadata is incomplete.
    """
    try:
        # Use IGNORECASE flag and findall to get all matches
        matches = re.findall(regex_pattern, text, re.IGNORECASE)
        if not matches:
            return None

        # Take the LAST match (rightmost)
        captured = _normalize_extracted_answer(matches[-1].strip()).upper()

        # Try direct letter match first
        if len(captured) == 1 and captured.isalpha():
            # If we have options metadata, validate against it
            if allowed_letters and captured in allowed_letters:
                return captured
            # If options metadata is missing/incomplete, trust the regex
            # This handles cases where template_metadata regex is used but options are incomplete
            elif not allowed_letters:
                return captured
            # If captured letter is not in allowed_letters but allowed_letters exists,
            # it might be a data quality issue - still return it when using template_metadata
            else:
                # Trust the regex when using template_metadata (this function is only called for template_metadata)
                return captured

        # Try matching against option text (normalized)
        normalized_captured = _normalize_for_match(captured)
        for entry in options or []:
            for k, v in entry.items():
                if v is not None and k.upper() in allowed_letters and _normalize_for_match(v) == normalized_captured:
                    return k.upper()

        return None
    except re.error:
        # Invalid regex pattern, return None
        return None

def _parse_answer_with_custom_regexes(
    text: str, regex_patterns: str | list[str], allowed_letters: set[str], options: Optional[list[dict[str, str]]]
) -> Optional[str]:
    if isinstance(regex_patterns, str):
        return _parse_answer_with_custom_regex(text, regex_patterns, allowed_letters, options)

    for regex_pattern in regex_patterns:
        pred = _parse_answer_with_custom_regex(text, regex_pattern, allowed_letters, options)
        if pred is not None:
            return pred
    return None

class MCQAResourcesServer:
    def __init__(self, config):
        self.config = config

    def _programmatic_score(self, body: MCQAVerifyRequest) -> MCQADualVerifyResponse | MCQAVerifyResponse:
        # Pull options/expected_answer from dataset-style metadata if available
        options, expected_answer = _extract_options_and_expected(body)
        gold = (expected_answer or "").strip().upper()
        # Derive allowed letters from option keys
        allowed_letters = _get_allowed_letters_from_options(options)

        grading_mode = self.config.grading_mode or body.grading_mode

        pred: Optional[str] = None

        text = body.response.output_text.strip()
        if grading_mode == "boxed_and_normalized":
            strict, normalized = extract_boxed_and_normalized_answers(text, options=options)
            strict_reward = float(strict is not None and strict == gold)
            normalized_reward = float(normalized is not None and normalized == gold)
            return MCQADualVerifyResponse(
                **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
                reward=strict_reward,
                reward_components={"strict_boxed": strict_reward, "normalized": normalized_reward},
                expected_answer=gold,
                extracted_answer=strict,
                strict_boxed_answer=strict,
                normalized_answer=normalized,
                strict_boxed_reward=strict_reward,
                normalized_reward=normalized_reward,
            )
        if grading_mode == "milu_v2":
            match = MILU_V2_ANSWER_PATTERN.search(text)
            pred = next((group.upper() for group in match.groups() if group), None) if match else None
            return MCQAVerifyResponse(
                **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
                reward=float(pred is not None and pred == gold),
                expected_answer=gold,
                extracted_answer=pred,
            )

        if not text:
            return MCQAVerifyResponse(
                **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
                reward=0.0,
                expected_answer=gold,
                extracted_answer=None,
            )

        # Check for template_metadata first (highest priority)
        if body.template_metadata and "output_regex" in body.template_metadata:
            regex_patterns = body.template_metadata["output_regex"]
            pred = _parse_answer_with_custom_regexes(text, regex_patterns, allowed_letters, options)

        # Fallback to existing grading_mode logic if template_metadata didn't work
        if pred is None:
            if grading_mode == "strict_single_letter_boxed":
                pred, _, _ = _parse_answer_letter_strict_boxed(text, allowed_letters)
            elif grading_mode == "lenient_boxed":
                # Try strict boxed first
                pred, _, _ = _parse_answer_letter_strict_boxed(text, allowed_letters)
                if pred is None:
                    # Then try to match option text inside boxed content only
                    letter_from_text = _match_option_text(text, options, allowed_letters)
                    if letter_from_text is not None:
                        pred = letter_from_text
            elif grading_mode == "lenient_answer_colon":
                # Look for Answer: <...>
                m = ANSWER_COLON_PATTERN.search(text)
                if m:
                    candidate = _strip_latex_wrappers(m.group(1)).strip()
                    # Letter case
                    if len(candidate) == 1 and candidate.isalpha():
                        letter_up = candidate.upper()
                        if letter_up in allowed_letters:
                            pred = letter_up
                    # Option text equality (normalized)
                    if pred is None:
                        cand_norm = _normalize_for_match(candidate)
                        for entry in options or []:
                            for k, v in entry.items():
                                k_up = k.upper()
                                if k_up in allowed_letters and _normalize_for_match(v) == cand_norm:
                                    pred = k_up
                                    break
                            if pred is not None:
                                break
            elif grading_mode == "lenient_answer_colon_md":
                answer_prefix = (body.template_metadata or {}).get("answer_prefix")
                pred = _extract_from_answer_payloads(
                    text,
                    allowed_letters,
                    answer_prefix=answer_prefix,
                )
                if pred is None and answer_prefix:
                    pred, _, _ = _parse_answer_letter_strict_boxed(text, allowed_letters)
                # Preserve all existing consumers and fallback precedence. Only
                # explicitly opted-in rows gain localized-label normalization.
                if pred is None and (body.template_metadata or {}).get("allow_localized_answer_labels") is True:
                    pred = _extract_terminal_labeled_choice(text, allowed_letters)

        is_correct = (pred == gold) if (pred is not None and gold) else False
        reward = 1.0 if is_correct else 0.0

        return MCQAVerifyResponse(
            **body.model_dump(exclude={"expected_answer", "extracted_answer"}),
            reward=reward,
            expected_answer=gold,
            extracted_answer=pred,
        )

def _get_allowed_letters_from_options(
    options: Optional[list[dict[str, str]]],
) -> set[str]:
    """Collect uppercase option letters from list of single-key dicts."""
    letters: set[str] = set()
    if options:
        for entry in options:
            # Exclude null values
            for k, v in entry.items():
                if isinstance(k, str) and len(k) == 1 and k.isalpha() and v is not None:
                    letters.add(k.upper())
    return letters

def extract_boxed_and_normalized_answers(
    text: str, *, options: Optional[list[dict[str, str]]]
) -> tuple[Optional[str], Optional[str]]:
    """Return strict boxed and normalized letters without consulting the gold label.

    Strict uses Gym's existing last-box parser, including its supported LaTeX
    wrappers. Normalized preserves any valid strict answer; otherwise it accepts
    a normalized last-box payload, the last Answer:/answer is payload, or a bare
    letter/unique exact option text. An invalid explicit final payload does not
    fall back to an earlier answer or arbitrary letters in the explanation.
    """
    allowed = _get_allowed_letters_from_options(options)
    strict, _, _ = _parse_answer_letter_strict_boxed(text, allowed)
    if strict is not None:
        return strict, strict

    canonical = unicodedata.normalize("NFKC", text).strip()
    canonical = re.sub(r"\\boxed\s+\{", r"\\boxed{", canonical)
    inner = _extract_boxed_inner(canonical)
    if "\\boxed{" in canonical:
        if inner is None:
            return None, None
        payload = _strip_latex_wrappers(inner.strip())
        explicit = True
    else:
        payloads = _answer_payloads(canonical, answer_prefix="answer is")
        payload = payloads[-1] if payloads else canonical
        explicit = bool(payloads)

    # Exact equality avoids awarding credit for a short option inside unrelated prose.
    # Match before label normalization, which can turn letters inside Indic words
    # into apparent choice labels.
    normalized_payload = _normalize_for_match(payload.strip().strip("*_`"))
    matches = {
        label.upper()
        for entry in options or []
        for label, option in entry.items()
        if option is not None and label.upper() in allowed and _normalize_for_match(option) == normalized_payload
    }
    if matches:
        return None, next(iter(matches)) if len(matches) == 1 else None

    candidate = _normalize_extracted_answer(payload)
    if _contains_ambiguous_choice_list(candidate, allowed):
        return None, None
    bare = candidate.strip("*_`[]() \t\n\r.,;:!").upper()
    if len(bare) == 1 and bare in allowed:
        return None, bare

    # Only a leading choice label may precede an explanation. Do not search prose
    # for incidental letters (including letters inside Indic words after normalization).
    leading_label = re.match(r"^[A-Z](?=$|[\s).:\-])", bare)
    if explicit and leading_label and leading_label.group(0) in allowed:
        return None, leading_label.group(0)
    return None, None
