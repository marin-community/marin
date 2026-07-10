# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distills the likely root-cause lines from a batch of noisy task logs.

Task logs bury the real failure signal — Python tracebacks, fatal-error
banners, JAX/NCCL/CUDA/Kueue diagnostics — under high-volume noise: tqdm
progress bars, per-request HTTP access-log lines, and CPython's post-crash
``Extension modules:`` dump. The extractor is a pure text filter over the log
lines: it drops the noise and keeps the lines that name the failure, so an
operator (or the dashboard) sees the crash first. It reads only the text, so
it works on Kubernetes pod logs and GCP/TPU worker-daemon logs alike.
"""

import re
from collections.abc import Sequence

_DEFAULT_MAX_LINES = 20

# The two halves of a tqdm frame, either of which identifies one:
#
#   with a total:    "Loading:  45%|####5     | 450/1000 [00:12<00:15,  3.21it/s]"
#   without a total: "450it [00:12,  3.21it/s]"
#
# The bracketed stats are the reliable half — an unknown total suppresses the
# percentage and the bar, but never the elapsed clock and rate. The rate always
# carries a slash, and tqdm inverts it below one iteration per second, so a slow
# training step reads "1.20s/it" rather than "0.83it/s".
#
# Neither pattern is anchored to the start of the line: a description prefixes
# the bar, and because tqdm redraws with a bare carriage return, one captured
# line often holds several frames at once.
_PROGRESS_BAR_PATTERNS = (
    re.compile(r"\d+%\|[^|]*\|"),
    re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?[^\]]*,\s*[\d.?]+[^\s\]]*/[^\s\],]+"),
)

# Lines that carry no diagnostic value and commonly flood task logs.
_NOISE_PATTERNS = (
    re.compile(r'"(?:GET|POST|PUT|HEAD|DELETE) [^"]* HTTP/1\.\d"\s*\d{3}'),  # HTTP access log line
    re.compile(r"^Extension modules:"),  # CPython post-crash loaded-module dump
)


# Lines likely to name the actual failure. Matched against common
# Python/JAX/NCCL/CUDA/Kueue/k8s fatal-error vocabulary.
_SIGNAL_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r'^\s*File "[^"]+", line \d+'),
    re.compile(r"Fatal Python error"),
    re.compile(r"\b\w*Error\b"),
    re.compile(r"\b\w*Exception\b"),
    re.compile(r"\bDEADLINE_EXCEEDED\b"),
    re.compile(r"\bRESOURCE_EXHAUSTED\b"),
    re.compile(r"\bOOMKilled\b"),
    re.compile(r"\bout of memory\b", re.IGNORECASE),
    re.compile(r"\bSegmentation fault\b"),
    re.compile(r"\bAborted\b"),
    re.compile(r"\bcore dumped\b"),
    re.compile(r"\bCUDA error\b"),
    re.compile(r"\bNCCL\b.*\berror\b", re.IGNORECASE),
    re.compile(r"\bCoscheduled sibling\b"),
    re.compile(r"detected fatal errors"),
)


def is_progress_bar_line(line: str) -> bool:
    """Whether ``line`` holds a tqdm-style progress bar frame anywhere in it."""
    return any(pattern.search(line) for pattern in _PROGRESS_BAR_PATTERNS)


def _is_noise(line: str) -> bool:
    return is_progress_bar_line(line) or any(pattern.search(line) for pattern in _NOISE_PATTERNS)


def extract_failure_highlights(lines: Sequence[str], max_lines: int = _DEFAULT_MAX_LINES) -> list[str]:
    """Return the most diagnostically useful lines from a batch of task logs.

    Drops known-noisy lines (tqdm bars, HTTP access logs, CPython's
    ``Extension modules:`` crash-dump tail) and consecutive duplicates —
    a barrier-timeout error commonly repeats once per straggler — then keeps
    lines matching common failure vocabulary (tracebacks, fatal errors,
    OOM/eviction/timeout signals). Falls back to the de-noised tail when no
    line matches, so the result is never empty for a non-empty input.

    Returns at most ``max_lines`` lines, keeping the most recent ones.
    """
    deduped: list[str] = []
    previous: str | None = None
    for line in lines:
        if _is_noise(line):
            continue
        if line == previous:
            continue
        deduped.append(line)
        previous = line

    signal_lines = [line for line in deduped if any(pattern.search(line) for pattern in _SIGNAL_PATTERNS)]
    result = signal_lines or deduped
    return result[-max_lines:]
