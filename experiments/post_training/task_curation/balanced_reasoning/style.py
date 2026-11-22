# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trace-style filters derived from the Nemotron-overthinking measurements.

The store must not carry the looping style even inside correct traces. The battery
reproduces the per-1k recheck-phrase rates that study measured on the Nemotron SFT stores
and on truncated policy responses -- wait / actually / let-me are the dominant recheck
markers -- and adds a verbatim-line detector for the degenerate repetition that study
measured on truncated reasoning-gym responses.
"""

import re
from dataclasses import dataclass
from itertools import pairwise

RECHECK_PHRASES: tuple[str, ...] = ("wait", "actually", "let me")
"""Case-insensitive recheck markers; rates are per 1k trace tokens."""

DEFAULT_RECHECK_RATE_LIMIT = 6.0
"""Combined recheck rate (per 1k tokens) above which a trace is rejected. The measured
Nemotron stores run 1.4-3.1/1k and truncated policy responses 1.7-4.1/1k, so 6.0 keeps
correct-but-rechecking traces while cutting the pathological tail."""

DEFAULT_MIN_LOOP_REPEATS = 6
"""Consecutive repetitions of one line that mark verbatim-loop degeneration."""

DEFAULT_MIN_LOOP_SHARE = 0.5
"""Share of nonempty lines carried by one repeated line (in an 8+-line trace) that marks
degeneration."""

RECHECK_PATTERN = re.compile("|".join(rf"\b{re.escape(phrase)}\b" for phrase in RECHECK_PHRASES), re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class StyleReport:
    recheck_rate: float
    """Combined recheck-phrase rate per 1k trace tokens."""
    phrase_rates: dict[str, float]
    loop_repeats: int
    """Longest run of consecutive identical nonempty lines."""
    loop_share: float
    """Share of nonempty lines held by the most repeated line."""
    degenerate: bool
    """Whether the trace carries a rejecting style signature."""

    def reason(self) -> str | None:
        """The rejecting signature, or ``None`` when the trace is shippable."""
        if not self.degenerate:
            return None
        if self.loop_repeats >= DEFAULT_MIN_LOOP_REPEATS:
            return f"verbatim loop: {self.loop_repeats} consecutive repeats of one line"
        return f"recheck rate {self.recheck_rate:.2f}/1k exceeds the limit {DEFAULT_RECHECK_RATE_LIMIT:.2f}/1k"


def _longest_consecutive_repeat(lines: list[str]) -> int:
    longest = 0
    run = 0
    for previous, current in pairwise(lines):
        run = run + 1 if current == previous else 1
        longest = max(longest, run)
    return longest


def style_report(trace_text: str, *, recheck_rate_limit: float = DEFAULT_RECHECK_RATE_LIMIT) -> StyleReport:
    """Measure one trace's style signatures and decide whether it is shippable."""
    tokens = trace_text.split()
    per_1k = 1_000.0 / len(tokens) if tokens else 0.0
    phrase_rates = {
        phrase: len(re.findall(rf"\b{re.escape(phrase)}\b", trace_text, flags=re.IGNORECASE)) * per_1k
        for phrase in RECHECK_PHRASES
    }
    lines = [line.strip() for line in trace_text.splitlines() if line.strip()]
    loop_repeats = _longest_consecutive_repeat(lines)
    line_counts: dict[str, int] = {}
    for line in lines:
        line_counts[line] = line_counts.get(line, 0) + 1
    loop_share = max(line_counts.values(), default=0) / len(lines) if lines else 0.0
    recheck_rate = sum(phrase_rates.values())
    degenerate = (
        loop_repeats >= DEFAULT_MIN_LOOP_REPEATS
        or (len(lines) >= 8 and loop_share >= DEFAULT_MIN_LOOP_SHARE)
        or recheck_rate > recheck_rate_limit
    )
    return StyleReport(
        recheck_rate=recheck_rate,
        phrase_rates=phrase_rates,
        loop_repeats=loop_repeats,
        loop_share=loop_share,
        degenerate=degenerate,
    )
