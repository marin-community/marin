# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SuperGPQA reference answer extraction for ``McqExtraction.SUPERGPQA_REFERENCE``.

Ports the zero-shot extraction of the SuperGPQA runner (SuperGPQA/SuperGPQA, ODC-By, at
e31e8bf3e7089aef808be5ddecafd4538dee6e41):
``extract_option_labels``, ``extract_option_content`` and the zero-shot branch of
``calculate_accuracy`` in ``eval/eval.py``. Each letter pattern is tried case-insensitively on the
last line, then on the whole response, and the first match wins; when no letter matches, the
option-text patterns map matched option text back to its letter. A lowercase match such as ``a``
is returned as is and scored wrong, as in the reference. Each search stops after
``SUPERGPQA_SEARCH_TIMEOUT`` seconds and counts as no match, as the reference's
``safe_regex_search`` does; some patterns backtrack exponentially on runs of characters such as
``*``, ``(`` or ``\\``. The timeout uses ``SIGALRM``, so this extraction runs only on the main thread.
"""

import re
import signal
import threading
import time
from collections.abc import Sequence

from verifyit.grade import GradingInfraError

SUPERGPQA_LETTERS = "ABCDEFGHIJ"
SUPERGPQA_LABEL_PATTERNS = (
    r"[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is?:?\s*(?:[\*\$\{(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*([<OPT>])(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
    r"(?i:Answer)[\*\s]*:\s*(?:[\*\$\{(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*([<OPT>])(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
    r"^[^\w\r\n]*(?:[\*\$\{(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*([<OPT>])(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
)
SUPERGPQA_CONTENT_PATTERNS = (
    r"[Tt]he\s+(?:\w+\s+)?(?:answer|option)(?:\w+\s+)?\s+is:?\s*(?:[\*\$\{\(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*(<CONTENT>)(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
    r"(?i:Answer)\s*(?:[\*\$\{\(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*(<CONTENT>)(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
    r"^[^\w\r\n]*(?:[\*\$\{\(\[\\(]*?(?:(?:\\boxed|\\mathbf|\\mathrm|\\text){)?)*\s*(<CONTENT>)(?:\\?\}?\$?\)?\]?\}?)*(?:[\s:\.\*)]|$)",
)
SUPERGPQA_SEARCH_TIMEOUT = 5.0
# A caller timer that expired during a search fires as soon as it is restored.
MIN_TIMER = 1e-6


class _SearchTimeout(Exception):
    pass


def _expire(_signum, _frame):
    raise _SearchTimeout


def _timed_search(pattern: str, text: str, flags: int) -> re.Match[str] | None:
    """Search under a ``SUPERGPQA_SEARCH_TIMEOUT`` alarm, then restore the caller's handler and remaining timer."""
    if threading.current_thread() is not threading.main_thread():
        raise GradingInfraError("SuperGPQA reference extraction needs the main thread for its regex timeout")
    previous_handler = signal.signal(signal.SIGALRM, _expire)
    previous_delay, previous_interval = signal.setitimer(signal.ITIMER_REAL, SUPERGPQA_SEARCH_TIMEOUT)
    start = time.monotonic()
    try:
        return re.search(pattern, text, flags)
    except _SearchTimeout:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_delay:
            remaining = previous_delay - (time.monotonic() - start)
            signal.setitimer(signal.ITIMER_REAL, max(remaining, MIN_TIMER), previous_interval)


def _first_match(patterns: Sequence[str], text: str, flags: int = 0) -> str | None:
    text = text.rstrip()
    for scope in (text.split("\n")[-1], text):
        for pattern in patterns:
            match = _timed_search(pattern, scope, flags)
            if match:
                return match.group(1)
    return None


def supergpqa_reference_answer(text: str, option_texts: Sequence[str]) -> str | None:
    """Return the SuperGPQA runner's extracted letter, or None when it counts the response as a miss."""
    label = _first_match(
        [pattern.replace("<OPT>", SUPERGPQA_LETTERS) for pattern in SUPERGPQA_LABEL_PATTERNS], text, re.IGNORECASE
    )
    if label is not None:
        return label
    alternatives = "|".join(re.escape(option) for option in option_texts)
    content = _first_match([pattern.replace("<CONTENT>", alternatives) for pattern in SUPERGPQA_CONTENT_PATTERNS], text)
    return SUPERGPQA_LETTERS[list(option_texts).index(content)] if content else None
