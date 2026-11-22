# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Detect periodic response suffixes without depending on training reward policy."""

from collections.abc import Sequence
from dataclasses import dataclass


def _active_segments(token_ids: Sequence[int], loss_mask: Sequence[int]) -> list[tuple[list[int], list[int]]]:
    if len(token_ids) != len(loss_mask):
        raise ValueError("response IDs and loss masks must have the same length")

    segments: list[tuple[list[int], list[int]]] = []
    segment_tokens: list[int] = []
    segment_positions: list[int] = []
    for position, (token_id, active) in enumerate(zip(token_ids, loss_mask, strict=True)):
        if active:
            segment_tokens.append(int(token_id))
            segment_positions.append(position)
        elif segment_tokens:
            segments.append((segment_tokens, segment_positions))
            segment_tokens = []
            segment_positions = []
    if segment_tokens:
        segments.append((segment_tokens, segment_positions))
    return segments


@dataclass(frozen=True)
class TailLoop:
    """Token coordinates for a periodic suffix within the last trainable segment."""

    start: int
    period: int
    end: int

    @property
    def repeat_start(self) -> int:
        return self.start + self.period


def detect_tail_loop(
    response_ids: Sequence[int],
    loss_mask: Sequence[int],
    *,
    max_period_tokens: int,
    tail_tokens: int,
    minimum_occurrences: int,
) -> TailLoop | None:
    """Find a repeating trainable suffix and return its original token coordinates."""
    segments = _active_segments(response_ids, loss_mask)
    if not segments:
        return None

    # Only a loop that survives to the final trainable suffix is charged. An
    # earlier loop followed by more assistant work is a recovery, not a failure.
    segment_tokens, segment_positions = segments[-1]
    tail = segment_tokens[-tail_tokens:]
    max_period = min(max_period_tokens, len(tail) // minimum_occurrences)
    period = next(
        (
            candidate
            for candidate in range(1, max_period + 1)
            if all(
                tail[-offset] == tail[-offset - candidate]
                for offset in range(1, candidate * (minimum_occurrences - 1) + 1)
            )
        ),
        None,
    )
    if period is None:
        return None

    periodic_start = len(segment_tokens) - period * minimum_occurrences
    while periodic_start > 0 and segment_tokens[periodic_start - 1] == segment_tokens[periodic_start - 1 + period]:
        periodic_start -= 1
    return TailLoop(segment_positions[periodic_start], period, segment_positions[-1] + 1)
