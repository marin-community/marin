"""Self-contained grader for the lm-eval-harness ``gsm8k`` task.

Ports the filter chains and ``exact_match`` metric of
``lm_eval/tasks/gsm8k/gsm8k.yaml`` at the pinned lm-evaluation-harness
``v0.4.12`` so grading needs only the problem text, the model's solution, and
the reference answer.

The task defines two filters over the same metric, and both are reproduced:
``strict-match`` reads the ``#### <number>`` line the few-shot targets end
with, and ``flexible-extract`` falls back to the last number before a shared
next-turn boundary.

Extraction is a single regex in both the reference and here, so the cost that
can be removed is in the metric. ``exact_match_hf_evaluate`` runs its four
``regexes_to_ignore`` as uncompiled ``re.sub`` calls over both sides, wraps each
result in a fresh ``numpy`` array, lowercases through ``np.char.lower``, and
reduces a one-element comparison with ``np.mean`` -- per grade. The
substitutions are all expressible as string operations, which is what this
module does; ``\\.$`` stays a compiled regex because ``$`` also matches before a
trailing newline.
"""

import re

from eval.generation_stops import GSM8K_STOP_SEQUENCES, truncate_at_stop

FALLBACK = "[invalid]"
_FINAL_ANSWER_MARKER = "#### "

_STRICT_MATCH = re.compile(r"#### (\-?[0-9\.\,]+)")
_FLEXIBLE_EXTRACT = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")
_TRAILING_PERIOD = re.compile(r"\.$")


def extract_strict_match(solution: str) -> str:
    """Read the ``#### <number>`` answer line, or ``FALLBACK`` if absent."""
    matches = _STRICT_MATCH.findall(solution)
    if not matches:
        return FALLBACK
    return matches[0].strip()


def extract_flexible(solution: str) -> str:
    """Read the last number before a next-turn boundary, or ``FALLBACK`` if absent.

    The reference pattern has two alternated groups, so ``findall`` yields
    tuples and the filter keeps the first non-empty group of the last match.
    """
    matches = _FLEXIBLE_EXTRACT.findall(truncate_at_stop(solution, GSM8K_STOP_SEQUENCES))
    if not matches:
        return FALLBACK
    groups = [group for group in matches[-1] if group]
    if not groups:
        return FALLBACK
    return groups[0].strip()


def normalize(text: str) -> str:
    """Apply the metric's ``regexes_to_ignore`` and ``ignore_case``, in order.

    The reference list is ``[",", "\\$", "(?s).*#### ", "\\.$"]``. The third is
    greedy under ``DOTALL``, so it drops everything through the *last* answer
    marker -- what ``str.rfind`` does.
    """
    text = text.replace(",", "").replace("$", "")
    marker = text.rfind(_FINAL_ANSWER_MARKER)
    if marker >= 0:
        text = text[marker + len(_FINAL_ANSWER_MARKER) :]
    return _TRAILING_PERIOD.sub("", text).lower()


def exact_match(prediction: str, reference: str) -> bool:
    return normalize(prediction) == normalize(reference)


def grade(problem: str, solution: str, reference_answer: str) -> float:
    """Grade one completion under the ``strict-match`` filter.

    Args:
        problem: Problem statement. Unused; part of the shared grader signature.
        solution: The model's completion.
        reference_answer: Gold answer. The full GSM8K ``answer`` field and a
            bare number both work, since normalization drops the reasoning
            ahead of the ``####`` marker either way.

    Returns:
        ``1.0`` when the completion matches the reference, else ``0.0``.
    """
    return 1.0 if exact_match(extract_strict_match(solution), reference_answer) else 0.0


def grade_flexible_extract(problem: str, solution: str, reference_answer: str) -> float:
    """Grade one completion under the ``flexible-extract`` filter."""
    return 1.0 if exact_match(extract_flexible(solution), reference_answer) else 0.0
