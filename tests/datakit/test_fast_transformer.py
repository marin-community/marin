# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fast-transformer quality scorer's two algorithmic contracts:

- ``scorer.score_bme`` — whole-doc (begin/middle/end) token-window coverage +
  mean-pooling, the fix for scoring long docs on a truncated lead / prefix-degenerate
  sources.
- ``calibrate.fit_cutpoints`` / ``calibration_knots`` — the monotonic cutpoint remap
  that makes the fixed 0.2-bucket quantization recover the oracle quality level.

Both use a deterministic fake scorer / synthetic labels, so no model or I/O is needed.
"""

from itertools import pairwise
from typing import cast

import numpy as np
import pytest

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.cluster.quality.fast_transformer.calibrate import calibration_knots, fit_cutpoints
from experiments.datakit.cluster.quality.fast_transformer.score import _systematic_take
from experiments.datakit.cluster.quality.fast_transformer.scorer import PooledScorer, score_bme

MAX_TOKENS = 8


class _FakeScorer:
    """Deterministic stand-in for ``PooledScorer``: ``score_windows(windows)`` returns a
    value per window keyed on its first token id (default otherwise), and records the
    exact window lists it was called with so tests can assert which windows were scored."""

    def __init__(self, by_first_id: dict[int, float] | None = None, default: float = 0.0) -> None:
        self._map = by_first_id or {}
        self._default = default
        self.max_tokens = MAX_TOKENS
        self.calls: list[list[np.ndarray]] = []

    def score_windows(self, windows: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
        self.calls.append(list(windows))
        return np.array([self._map.get(int(w[0]), self._default) for w in windows], dtype=float)


def _as_scorer(fake: _FakeScorer) -> PooledScorer:
    return cast(PooledScorer, fake)


# ---------- score_bme: whole-doc window coverage + pooling ----------


def test_bme_short_doc_scores_as_single_window():
    fake = _FakeScorer({10: 0.3})
    doc = np.full(5, 10)  # <= MAX_TOKENS
    out = score_bme(_as_scorer(fake), [doc])
    assert len(fake.calls) == 1 and len(fake.calls[0]) == 1
    assert fake.calls[0][0].tolist() == doc.tolist()  # exactly one window = the whole doc
    assert out.tolist() == pytest.approx([0.3])


def test_bme_long_doc_covers_begin_middle_end_and_mean_pools():
    fake = _FakeScorer({10: 0.0, 20: 0.6, 30: 0.9})
    # begin -> 10 block, middle -> 20 block, end -> 30 block (each exactly one window)
    doc = np.array([10] * MAX_TOKENS + [20] * MAX_TOKENS + [30] * MAX_TOKENS)
    out = score_bme(_as_scorer(fake), [doc])

    windows = fake.calls[0]
    assert len(windows) == 3
    assert all(len(w) == MAX_TOKENS for w in windows)
    # the three windows are begin / middle / end of the whole doc -- not just the lead
    assert (windows[0][0], windows[1][0], windows[2][0]) == (10, 20, 30)
    assert out.tolist() == pytest.approx([(0.0 + 0.6 + 0.9) / 3])  # mean-pooled


def test_bme_batch_pools_each_doc_independently():
    fake = _FakeScorer({10: 0.3, 11: 0.0, 20: 0.6, 30: 0.9})
    short = np.full(5, 10)
    long = np.array([11] * MAX_TOKENS + [20] * MAX_TOKENS + [30] * MAX_TOKENS)
    out = score_bme(_as_scorer(fake), [short, long])
    # all 1 + 3 windows scored in a single batched call; spans map back per doc
    assert len(fake.calls) == 1 and len(fake.calls[0]) == 4
    assert out.tolist() == pytest.approx([0.3, (0.0 + 0.6 + 0.9) / 3])


def test_bme_window_count_switches_at_max_tokens():
    fake = _FakeScorer(default=0.5)
    score_bme(_as_scorer(fake), [np.arange(MAX_TOKENS)])  # == threshold
    score_bme(_as_scorer(fake), [np.arange(MAX_TOKENS + 1)])  # one token over
    assert len(fake.calls[0]) == 1  # <= MAX_TOKENS -> single window
    assert len(fake.calls[1]) == 3  # > MAX_TOKENS  -> begin/middle/end
    assert all(len(w) == MAX_TOKENS for w in fake.calls[1])  # each window exactly MAX_TOKENS


# ---------- calibrate: monotonic cutpoint remap ----------


def test_fit_cutpoints_are_midpoints_of_adjacent_level_medians():
    # level L docs all have raw = L/10 -> medians {1:.1, ..., 5:.5}
    levels = np.repeat([1, 2, 3, 4, 5], 4).astype(float)
    raw = levels / 10.0
    med, cuts = fit_cutpoints(raw, levels)
    assert med == pytest.approx({1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5})
    assert cuts == pytest.approx([0.15, 0.25, 0.35, 0.45])


def test_fit_cutpoints_enforced_non_decreasing():
    # medians whose raw midpoints would dip (0.55 -> 0.35); accumulate must fix it
    raw = np.array([0.2, 0.8, 0.3, 0.4, 0.5])
    levels = np.array([1, 2, 3, 4, 5], dtype=float)
    _, cuts = fit_cutpoints(raw, levels)
    assert cuts == pytest.approx([0.5, 0.55, 0.55, 0.55])
    assert all(b >= a for a, b in pairwise(cuts))


def test_calibration_knots_are_strictly_increasing_and_recover_levels():
    levels = np.repeat([1, 2, 3, 4, 5], 4).astype(float)
    raw = levels / 10.0
    knots = calibration_knots(raw, levels)
    xk, yk = knots["xk"], knots["yk"]

    assert yk == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert len(xk) == 6
    # np.interp requires strictly increasing knots
    assert all(b > a for a, b in pairwise(xk))
    # each oracle level's median maps into (within one of) the matching bucket
    for level in (1, 2, 3, 4, 5):
        bucket = int(np.digitize(np.interp(level / 10.0, xk, yk), BUCKET_EDGES))
        assert abs(bucket - (level - 1)) <= 1


# ---------- score: deterministic non-hashing sample ----------


def test_systematic_sample_is_deterministic_and_hits_target_fraction():
    for pct in (0.1, 0.25, 0.5):
        kept = [i for i in range(1000) if _systematic_take(i, pct)]
        # deterministic: no RNG / no hashing -> identical across calls
        assert kept == [i for i in range(1000) if _systematic_take(i, pct)]
        # ~pct of records, evenly spaced
        assert abs(len(kept) / 1000 - pct) < 0.01
