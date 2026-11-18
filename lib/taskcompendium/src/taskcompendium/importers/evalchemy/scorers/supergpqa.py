# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SuperGPQA zero-shot answer extraction.

Ported from SuperGPQA/SuperGPQA (ODC-By) at e31e8bf3e7089aef808be5ddecafd4538dee6e41:
``extract_option_labels``, ``extract_option_content`` and the zero-shot branch of
``calculate_accuracy`` in ``eval/eval.py``. Each letter pattern is tried case-insensitively on the
last line, then on the whole response, and the first match wins; when no letter matches, the
option-text patterns map a quoted option back to its letter. A lowercase match such as ``a`` is
returned as is and scored wrong, as in the reference. Each search stops after ``SEARCH_TIMEOUT`` seconds
and counts as no match, as the reference's ``safe_regex_search`` does; some patterns backtrack
exponentially on runs of characters such as ``*``, ``(`` or ``\\``.
"""

import re
import signal
from collections.abc import Sequence

from verifyit.adapters.completion import final_text
from verifyit.grade import Reward, scored

LETTERS = "ABCDEFGHIJ"
LABEL_PATTERNS = (
    r"[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is?:?\s*(?:[\*\$\{(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*([<OPT>])(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
    r"(?i:Answer)[\*\s]*:\s*(?:[\*\$\{(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*([<OPT>])(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
    r"^[^\w\r\n]*(?:[\*\$\{(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*([<OPT>])(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
)
CONTENT_PATTERNS = (
    r"[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is:?\s*(?:[\*\$\{\(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*(<CONTENT>)(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
    r"(?i:Answer)\s*(?:[\*\$\{\(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*(<CONTENT>)(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
    r"^[^\w\r\n]*(?:[\*\$\{\(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*(<CONTENT>)(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
)
SEARCH_TIMEOUT = 5.0
MAX_SEARCH_TIME = 2 * (len(LABEL_PATTERNS) + len(CONTENT_PATTERNS)) * SEARCH_TIMEOUT


class _SearchTimeout(Exception):
    pass


def _expire(_signum, _frame):
    raise _SearchTimeout


def _search(pattern: str, text: str, flags: int) -> re.Match[str] | None:
    previous = signal.signal(signal.SIGALRM, _expire)
    signal.setitimer(signal.ITIMER_REAL, SEARCH_TIMEOUT)
    try:
        return re.search(pattern, text, flags)
    except _SearchTimeout:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _first_match(patterns: Sequence[str], text: str, flags: int = 0) -> str | None:
    text = text.rstrip()
    for scope in (text.split("\n")[-1], text):
        for pattern in patterns:
            match = _search(pattern, scope, flags)
            if match:
                return match.group(1)
    return None


def extract_option_label(text: str) -> str | None:
    """Return the first answer letter the reference's patterns find, in its original case."""
    return _first_match([pattern.replace("<OPT>", LETTERS) for pattern in LABEL_PATTERNS], text, re.IGNORECASE)


def extract_option_content(text: str, options: Sequence[str]) -> str | None:
    """Return the option whose exact text the reference's patterns find."""
    alternatives = "|".join(re.escape(option) for option in options)
    return _first_match([pattern.replace("<CONTENT>", alternatives) for pattern in CONTENT_PATTERNS], text)


def supergpqa_answer(response: str, options: Sequence[str]) -> str | None:
    """Return the extracted letter, or None when the reference counts the response as a miss."""
    label = extract_option_label(response)
    if label is not None:
        return label
    content = extract_option_content(response, options)
    return LETTERS[list(options).index(content)] if content else None


def grade_supergpqa(expected: str, options: Sequence[str], response: str) -> Reward:
    answer = supergpqa_answer(final_text(response), options)
    return scored(float(answer == expected), extracted=answer, expected=expected)
