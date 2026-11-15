# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import re
import unicodedata
from io import StringIO
from typing import Any, Optional
from math_verify import grader
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


def _extract_last_boxed_answer(response: str) -> Optional[str]:
    r"""Extract the exact contents of the last complete ``\boxed{...}``."""
    boxed_start = response.rfind("\\boxed{")
    if boxed_start < 0:
        return None

    content_start = boxed_start + len("\\boxed{")
    brace_depth = 1
    for index in range(content_start, len(response)):
        if response[index] == "{":
            brace_depth += 1
        elif response[index] == "}":
            brace_depth -= 1
            if brace_depth == 0:
                return response[content_start:index]

    return None

def _clean_answer_candidate(candidate: str) -> Optional[str]:
    """Remove Markdown wrappers while retaining math delimiters for the verifier."""
    candidate = candidate.strip()
    # Bound cleanup before running regexes, including for explicitly labelled answers.
    if len(candidate) > 512:
        return None
    candidate = re.sub(r"^(?:\*\*|__|`)+", "", candidate)
    candidate = re.sub(r"(?:\*\*|__|`)+$", "", candidate).strip()
    # A list marker is unambiguous before an answer label, but here a leading
    # hyphen may be a unary minus (including the spaced form "- 18").
    candidate = re.sub(r"^[ \t]*[:：=][ \t]*", "", candidate)
    candidate = re.sub(r"(?:\*\*|__)+$", "", candidate.rstrip()).strip()

    candidate = candidate.rstrip(".|।۔").strip()
    return candidate or None

def _extract_leading_answer_block(response: str, start: int = 0) -> Optional[str]:
    """Select one line or one complete math block immediately following a label."""
    while start < len(response) and response[start].isspace():
        start += 1
    if start == len(response):
        return None
    # Inspect only this candidate, not the entire remaining response for every
    # rejected statement. Repeated negated labels must remain linear to scan.
    line_end = response.find("\n", start)
    first_line = response[start : line_end if line_end >= 0 else len(response)]
    for opening, closing in _OUTER_MATH_DELIMITERS:
        if response.startswith(opening, start) and closing not in first_line[len(opening) :]:
            end = response.find(closing, start + len(opening), start + 512)
            if end < 0:
                return None
            return _clean_answer_candidate(response[start : end + len(closing)])
    return _clean_answer_candidate(first_line)

def _extract_unboxed_answer_candidate(response: str) -> Optional[str]:
    """Extract an explicitly marked or concise final answer without using the reference answer."""
    matches = list(_ANSWER_LABEL_RE.finditer(response))
    if matches:
        for match in reversed(matches):
            candidate = _extract_leading_answer_block(response, match.end())
            if candidate and _NONANSWER_PREFIX_RE.match(candidate):
                continue
            return candidate
        return None

    # A final display-math block can span lines even without an answer heading.
    tail = response.rstrip().rstrip(".|।۔").rstrip()
    for opening, closing in _OUTER_MATH_DELIMITERS:
        if tail.endswith(closing):
            start = tail.rfind(opening, max(0, len(tail) - 512), len(tail) - len(closing))
            if start < 0:
                return None
            if "\n" in tail[start:]:
                return _clean_answer_candidate(tail[start:])
            break

    nonempty_lines = [line for line in response.splitlines() if line.strip()]
    if not nonempty_lines:
        return None

    candidate = _clean_answer_candidate(nonempty_lines[-1])
    if candidate is None or len(candidate) > 512 or candidate.startswith(("#", "* ")):
        return None

    # A standalone final line may be a number, LaTeX expression, equation,
    # coordinate/interval, or short textual answer such as "east".
    has_math_signal = re.search(r"\d|\\|[{}\[\]()+\-*/^=<>]", candidate)
    short_text_answer = re.fullmatch(r"[\w\s'\"]+", candidate) and len(candidate.split()) <= 8
    if (has_math_signal and len(candidate.split()) <= 50) or short_text_answer:
        return candidate
    return None

def _leading_math_end(candidate: str) -> int:
    """Find a math/text boundary without splitting a command from its braced arguments."""
    index = end = 0
    stack = []
    while token := _MATH_TOKEN_RE.match(candidate, index):
        value = token[0]
        index = token.end()
        if value.startswith("\\"):
            # Consume each command and all adjacent braced arguments once. Text
            # inside an argument (e.g. \text{ cm}) is part of its LaTeX syntax.
            while True:
                opening = index
                while opening < len(candidate) and candidate[opening].isspace():
                    opening += 1
                if opening == len(candidate) or candidate[opening] != "{":
                    break
                depth = 1
                index = opening + 1
                while index < len(candidate) and depth:
                    character = candidate[index]
                    if character == "\\":
                        index += 2
                        continue
                    depth += (character == "{") - (character == "}")
                    index += 1
                if depth:
                    return end
        elif value in "({[":
            stack.append(value)
        elif value in ")}]":
            if not stack:
                return end
            opening = stack.pop()
            # Parentheses/brackets can mix for half-open intervals.
            if (opening == "{") != (value == "}"):
                return end
        if not stack:
            end = index
    return end

def _prepare_unboxed_verification_response(candidate: str) -> str:
    """Preserve mathematical syntax without treating explanatory prose as LaTeX."""
    expression = candidate[: _leading_math_end(candidate)].rstrip()
    if expression and expression[-1] not in "+-*/^=<>_|:;−±":
        candidate = f"${expression}$" + candidate[len(expression) :]
    # The anchor gives a leading answer precedence over numbers in an explanation.
    # Only the selected span is passed here; earlier answers are never reintroduced.
    return f"Final answer is {candidate}"

def _normalize_unicode_digits(response: str) -> str:
    """Convert Unicode decimal digits to ASCII while preserving all other text."""
    normalized = []
    for character in response:
        try:
            normalized.append(str(unicodedata.decimal(character)))
        except (TypeError, ValueError):
            normalized.append(character)
    return "".join(normalized)

def _repair_final_unclosed_box(response: str) -> str:
    """Close one missing outer brace in the final boxed answer."""
    boxed_start = response.rfind("\\boxed{")
    if boxed_start < 0 or _extract_last_boxed_answer(response) is not None:
        return response

    stripped = response.rstrip()
    suffix = ""
    for delimiter in ("$$", r"\]", "$", r"\)"):
        if stripped.endswith(delimiter):
            stripped = stripped[: -len(delimiter)].rstrip()
            suffix = delimiter
            break

    brace_depth = 0
    for character in stripped[boxed_start:]:
        if character == "{":
            brace_depth += 1
        elif character == "}":
            brace_depth -= 1
            if brace_depth < 0:
                return response

    if brace_depth != 1:
        return response
    return f"{stripped}}}{suffix}"

def _prepare_format_tolerant_response(response: str) -> str:
    """Apply reference-independent repairs used by Indic MATH-500."""
    return _repair_final_unclosed_box(_normalize_unicode_digits(response))

def _run_math_verify(
    library_verifier: Any, expected_answer: str, generated_answer: str
) -> tuple[float, Optional[str]]:
    # This functionality is migrated from Nemo RL.
    # https://github.com/NVIDIA-NeMo/RL/blob/e1f56c42ae175d3863ccaf4e21b7de7e9c46c2e1/nemo_rl/environments/math_environment.py
    try:
        stripped = LibraryJudgeMathResourcesServer._strip_math_delimiters(expected_answer)
        ground_truth_parsable = "\\boxed{" + stripped + "}"
        with LibraryJudgeMathResourcesServer._mute_output():
            ret_score, extracted_answer = library_verifier([ground_truth_parsable], [generated_answer])

        reward = float(ret_score)

        if extracted_answer is not None:
            # Make sure the extracted answer has two elements.
            assert len(extracted_answer) == 2

            extracted_gold, extracted_prediction = extracted_answer

            # Get the extracted answer.
            for pred in extracted_prediction:
                if any(grader.verify(gold, pred) for gold in extracted_gold):
                    extracted_answer = pred
                    break
            else:
                # If no match is found, that means all the answers are
                # incorrect.  The first prediction is used as the extracted
                # answer.
                extracted_answer = extracted_prediction[0] if extracted_prediction else None

        return reward, extracted_answer

    # It's possible to emit a TimeoutException and that wouldn't be caught since
    # it actually subclasses from BaseException and math-verify itself does not
    # catch it.
    except (Exception, TimeoutException):
        return 0.0, None

def _run_math_verify_in_subprocess(expected_answer: str, generated_answer: str, result_connection: Any) -> None:
    # Keep math_verify construction inside the child process. A wedged SymPy call
    # can then be killed by terminating this process without poisoning the server.
    library_verifier = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(
            ExprExtractionConfig(),
            LatexExtractionConfig(),
        ),
    )
    try:
        result_connection.send(_run_math_verify(library_verifier, expected_answer, generated_answer))
    finally:
        result_connection.close()

class LibraryJudgeMathResourcesServer:
    @classmethod
    @contextlib.contextmanager
    def _mute_output(cls):
        devnull_out, devnull_err = StringIO(), StringIO()
        with (
            contextlib.redirect_stdout(devnull_out),
            contextlib.redirect_stderr(devnull_err),
        ):
            yield
    @staticmethod
    def _strip_math_delimiters(s: str) -> str:
        """Strip outer math delimiters from expected answers.

        Many expected_answer values are wrapped in \\(...\\) or $...$,
        which causes the math_verify parser to fail when we wrap them
        in \\boxed{}.  Removing these outer delimiters fixes parsing.
        """
        s = s.strip()
        if s.startswith("\\(") and s.endswith("\\)"):
            s = s[2:-2].strip()
        if s.startswith("$") and s.endswith("$") and len(s) > 1:
            s = s[1:-1].strip()
        return s
