# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grade a standalone option letter or the last ``Answer: X`` in the output.

Answer lines may wrap the letter in boxes, markdown or brackets. A bare letter must
occupy the entire stripped output; prose ending in a letter is not an answer.

``McqExtraction.SUPERGPQA_REFERENCE`` instead ports the zero-shot extraction of the SuperGPQA
runner (SuperGPQA/SuperGPQA, ODC-By, at e31e8bf3e7089aef808be5ddecafd4538dee6e41):
``extract_option_labels``, ``extract_option_content`` and the zero-shot branch of
``calculate_accuracy`` in ``eval/eval.py``. Each letter pattern is tried case-insensitively on the
last line, then on the whole response, and the first match wins; when no letter matches, the
option-text patterns map matched option text back to its letter. A lowercase match such as ``a``
is returned as is and scored wrong, as in the reference. Each search stops after
``SUPERGPQA_SEARCH_TIMEOUT`` seconds and counts as no match, as the reference's
``safe_regex_search`` does; some patterns backtrack exponentially on runs of characters such as
``*``, ``(`` or ``\\``. The timeout uses ``SIGALRM``, so this extraction runs only on the main thread.
"""

import math
import re
import signal
import string
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from verifyit.adapters.completion import final_text
from verifyit.grade import GradingInfraError, InvalidTask, Reward, empty_output_policy, read_output, scored
from verifyit.spec import McqExtraction, McqSpec

ANSWER = re.compile(r"Answer\s*:\s*(?!Answer)\s*([A-Za-z0-9])(?![A-Za-z0-9])\s*")
BOXED_LETTER = re.compile(r"\\boxed\{\s*([A-Za-z0-9])\s*\}")
WRAPPERS = re.compile(r"[*`_()\[\]]")
MAX_OPTIONS = len(string.ascii_uppercase)
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


class LikelihoodScoring(StrEnum):
    MOST_LIKELY = "most_likely"
    PROBABILITY_MASS = "probability_mass"


@dataclass(frozen=True)
class LikelihoodStatistics:
    """Corpus diagnostics, not a bounded candidate Reward."""

    mean_log_likelihood: float
    perplexity: float
    bits_per_unit: float


def _normalization_size(lengths: Sequence[int]) -> int:
    if (
        not isinstance(lengths, Sequence)
        or not lengths
        or any(type(length) is not int or length <= 0 for length in lengths)
    ):
        raise InvalidTask("likelihood normalization requires positive integer lengths")
    return len(lengths)


def _validate_likelihoods(likelihoods: Sequence[float], expected: int) -> None:
    if not isinstance(likelihoods, Sequence) or len(likelihoods) != expected:
        raise InvalidTask("likelihood count must match normalization lengths")
    try:
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in likelihoods
        ):
            raise InvalidTask("model likelihoods must be finite numbers")
    except OverflowError as error:
        raise InvalidTask("model likelihoods must be finite numbers") from error


def summarize_log_likelihoods(
    likelihoods: Sequence[float], *, normalization_lengths: Sequence[int]
) -> LikelihoodStatistics:
    """Compute weighted corpus diagnostics under the supplied word/byte/token counts.

    Trusted lengths are validated before model observations. Log probabilities
    must be finite and nonpositive; source-order summation preserves corpus
    weighting. Invalid or unrepresentable statistics raise InvalidTask rather
    than returning a favorable zero. No result is converted into a Reward.
    """
    size = _normalization_size(normalization_lengths)
    _validate_likelihoods(likelihoods, size)
    if any(value > 0 for value in likelihoods):
        raise InvalidTask("model log-likelihoods must be nonpositive")
    try:
        total = sum(likelihoods)
        if not math.isfinite(total):
            raise InvalidTask("corpus log-likelihood sum must be finite")
        mean = total / sum(normalization_lengths)
        perplexity = math.exp(-mean)
        bits = -mean / math.log(2)
        if not all(math.isfinite(value) for value in (mean, perplexity, bits)):
            raise InvalidTask("corpus likelihood statistics must be finite")
    except OverflowError as error:
        raise InvalidTask("corpus likelihood statistics must be finite") from error
    return LikelihoodStatistics(mean, perplexity, bits)


def grade_mcq_likelihoods(
    likelihoods: Sequence[float],
    correct_indices: Sequence[int],
    *,
    normalization_lengths: Sequence[int],
    policy: LikelihoodScoring,
) -> Reward:
    """Grade choice likelihoods by first argmax or stable correct-answer mass.

    Normalization lengths encode the task's raw, character, byte, or token policy.
    Raw likelihoods use all ones. Correct indices may name several alternatives;
    repeated indices never increase probability mass. Malformed vectors raise
    InvalidTask, so the dispatch boundary emits zero without a reward file.
    """
    if not isinstance(policy, LikelihoodScoring):
        raise InvalidTask("unknown MCQ likelihood scoring policy")
    options = _normalization_size(normalization_lengths)
    if (
        not isinstance(correct_indices, Sequence)
        or not correct_indices
        or any(type(index) is not int or not 0 <= index < options for index in correct_indices)
    ):
        raise InvalidTask("MCQ requires correct indices within the declared choices")
    _validate_likelihoods(likelihoods, options)
    try:
        scores = [value / length for value, length in zip(likelihoods, normalization_lengths, strict=True)]
    except OverflowError as error:
        raise InvalidTask("MCQ likelihoods must be finite numbers") from error
    selected = max(range(options), key=scores.__getitem__)
    correct = sorted(set(correct_indices))
    if policy is LikelihoodScoring.MOST_LIKELY:
        return scored(float(selected in correct), selected_index=selected, correct_indices=correct)
    maximum = scores[selected]
    weights = [math.exp(value - maximum) for value in scores]
    denominator = math.fsum(weights)
    return scored(
        math.fsum(weights[index] for index in correct) / denominator,
        selected_index=selected,
        correct_indices=correct,
        probabilities=[weight / denominator for weight in weights],
    )


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


def answer_letters(text: str) -> list[str]:
    """Every letter stated on an ``Answer:`` line, wrappers around the letter removed."""
    return ANSWER.findall(WRAPPERS.sub("", BOXED_LETTER.sub(r"\1", text)))


def validate_mcq(spec: McqSpec) -> None:
    empty_output_policy(spec)
    if type(spec.options) is not int or not 1 <= spec.options <= MAX_OPTIONS:
        raise InvalidTask(f"mcq options must be 1..{MAX_OPTIONS}, got {spec.options}")
    letters = tuple(string.ascii_uppercase[: spec.options])
    if not isinstance(spec.expected, str):
        raise InvalidTask("mcq expected must be a string")
    expected = spec.expected.strip().upper()
    if expected not in letters:
        raise InvalidTask(f"mcq expected {spec.expected!r} is not one of {letters!r}")
    if spec.extraction is McqExtraction.SUPERGPQA_REFERENCE:
        if spec.options > len(SUPERGPQA_LETTERS):
            raise InvalidTask(f"SuperGPQA reference extraction supports at most {len(SUPERGPQA_LETTERS)} options")
        if len(spec.option_texts) != spec.options or not all(isinstance(text, str) for text in spec.option_texts):
            raise InvalidTask("SuperGPQA reference extraction requires one option text per option")


def grade_mcq_candidate(spec: McqSpec, candidate: str) -> Reward:
    """Score an extracted MCQ option letter, or the full response under the SuperGPQA reference extraction."""
    validate_mcq(spec)
    letters = tuple(string.ascii_uppercase[: spec.options])
    expected = spec.expected.strip().upper()
    if spec.extraction is McqExtraction.SUPERGPQA_REFERENCE:
        answer = supergpqa_reference_answer(final_text(candidate), spec.option_texts)
        return scored(float(answer == expected), extracted=answer, expected=expected)
    extracted = candidate.strip().upper()
    if not extracted:
        return scored(0.0, reason="no_answer_line", expected=expected)
    if extracted not in letters:
        return scored(0.0, reason="out_of_range", extracted=extracted, expected=expected)
    return scored(float(extracted == expected), extracted=extracted, expected=expected)


def grade(spec: McqSpec, tests_dir: Path, workspace: Path) -> Reward:
    no_answer_line = grade_mcq_candidate(spec, "")
    text = read_output(spec, workspace)
    if text is None:
        return scored(0.0, reason="no_output")
    if spec.extraction is McqExtraction.SUPERGPQA_REFERENCE:
        return grade_mcq_candidate(spec, text)
    matches = answer_letters(text)
    if matches:
        return grade_mcq_candidate(spec, matches[-1])
    candidate = text.strip()
    if len(candidate) == 1 and candidate in string.ascii_letters:
        return grade_mcq_candidate(spec, candidate)
    return no_answer_line
