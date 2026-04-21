# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the straggler detection callback aggregation logic."""

from levanter.callbacks.straggler import (
    CHRONIC_STRAGGLER_THRESHOLD,
    compute_rank_stats,
    update_ewma,
)


def test_update_ewma_initial():
    """First value should be returned as-is when previous EWMA is 0."""
    assert update_ewma(0.0, 1.5) == 1.5


def test_update_ewma_blends():
    """Subsequent values should blend with the previous EWMA."""
    alpha = 0.3
    prev = 1.0
    new = 2.0
    expected = alpha * new + (1 - alpha) * prev
    assert update_ewma(prev, new, alpha=alpha) == expected


def test_compute_rank_stats_basic():
    """Min/max/median/mean and top-k slowest are computed correctly."""
    durations = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    ewmas = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    stats = compute_rank_stats(durations, ewmas, top_k=2)

    assert stats.min_duration == 1.0
    assert stats.max_duration == 4.0
    assert stats.median_duration == 2.5
    assert stats.mean_duration == 2.5
    assert len(stats.slowest_ranks) == 2
    assert stats.slowest_ranks[0] == (3, 4.0)
    assert stats.slowest_ranks[1] == (2, 3.0)


def test_compute_rank_stats_chronic_detection():
    """Ranks with EWMA significantly above median are flagged as chronic."""
    # Median EWMA = 1.0; rank 3 at 2.0 exceeds 1.0 * 1.15 = 1.15
    durations = {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0}
    ewmas = {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0}
    stats = compute_rank_stats(durations, ewmas, top_k=2, chronic_threshold=CHRONIC_STRAGGLER_THRESHOLD)

    assert len(stats.chronic_stragglers) == 1
    assert stats.chronic_stragglers[0][0] == 3


def test_compute_rank_stats_no_chronic_when_uniform():
    """No chronic stragglers when all ranks have similar EWMA."""
    durations = {0: 1.0, 1: 1.01, 2: 0.99, 3: 1.0}
    ewmas = {0: 1.0, 1: 1.01, 2: 0.99, 3: 1.0}
    stats = compute_rank_stats(durations, ewmas, top_k=2)

    assert len(stats.chronic_stragglers) == 0


def test_compute_rank_stats_single_rank():
    """Single rank should produce valid stats with no chronic stragglers."""
    durations = {0: 1.5}
    ewmas = {0: 1.5}
    stats = compute_rank_stats(durations, ewmas, top_k=3)

    assert stats.min_duration == 1.5
    assert stats.max_duration == 1.5
    assert stats.median_duration == 1.5
    assert len(stats.slowest_ranks) == 1
    assert len(stats.chronic_stragglers) == 0


def test_compute_rank_stats_top_k_larger_than_ranks():
    """top_k larger than rank count should return all ranks without error."""
    durations = {0: 1.0, 1: 2.0}
    ewmas = {0: 1.0, 1: 2.0}
    stats = compute_rank_stats(durations, ewmas, top_k=10)

    assert len(stats.slowest_ranks) == 2


def test_ewma_converges():
    """EWMA should converge toward a constant input value."""
    ewma = 0.0
    for _ in range(50):
        ewma = update_ewma(ewma, 2.0, alpha=0.3)
    assert abs(ewma - 2.0) < 0.01
