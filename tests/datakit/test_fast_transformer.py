# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fast-transformer quality scorer's two algorithmic contracts:

- ``scorer.score_bme`` — whole-doc (begin/middle/end) window coverage + mean-pooling,
  the fix for scoring long docs on a truncated lead / prefix-degenerate sources.
- ``calibrate.fit_cutpoints`` / ``calibration_knots`` — the monotonic cutpoint remap
  that makes the fixed 0.2-bucket quantization recover the oracle quality level.

Both use a deterministic fake scorer / synthetic labels, so no model or I/O is needed.
"""

from itertools import pairwise
from typing import cast

import numpy as np
import pytest

from experiments.datakit.cluster.quality.fast_transformer.calibrate import (
    BUCKET_EDGES,
    calibration_knots,
    fit_cutpoints,
)
from experiments.datakit.cluster.quality.fast_transformer.score import _output_paths, _source_of, _systematic_take
from experiments.datakit.cluster.quality.fast_transformer.scorer import CHUNK_CHARS, PooledScorer, score_bme


class _FakeScorer:
    """Deterministic stand-in for ``PooledScorer``: ``score(texts)`` returns a value
    per text keyed on its first character (default otherwise), and records the exact
    chunk lists it was called with so tests can assert which windows were scored."""

    def __init__(self, by_first_char: dict[str, float] | None = None, default: float = 0.0) -> None:
        self._map = by_first_char or {}
        self._default = default
        self.calls: list[list[str]] = []

    def score(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        self.calls.append(list(texts))
        return np.array([self._map.get(t[:1], self._default) for t in texts], dtype=float)


def _as_scorer(fake: _FakeScorer) -> PooledScorer:
    return cast(PooledScorer, fake)


# ---------- _score_bme: whole-doc window coverage + pooling ----------


def test_bme_short_doc_scores_as_single_window():
    fake = _FakeScorer({"x": 0.3})
    doc = "x" * 100  # <= CHUNK_CHARS
    out = score_bme(_as_scorer(fake), [doc])
    assert fake.calls == [[doc]]  # exactly one chunk = the whole doc
    assert out.tolist() == pytest.approx([0.3])


def test_bme_long_doc_covers_begin_middle_end_and_mean_pools():
    fake = _FakeScorer({"A": 0.0, "B": 0.6, "C": 0.9})
    # begin -> A block, middle -> B block, end -> C block (each exactly one chunk)
    doc = "A" * CHUNK_CHARS + "B" * CHUNK_CHARS + "C" * CHUNK_CHARS
    out = score_bme(_as_scorer(fake), [doc])

    chunks = fake.calls[0]
    assert len(chunks) == 3
    assert all(len(c) == CHUNK_CHARS for c in chunks)
    # the three windows are begin / middle / end of the whole doc -- not just the lead
    assert (chunks[0][0], chunks[1][0], chunks[2][0]) == ("A", "B", "C")
    assert out.tolist() == pytest.approx([(0.0 + 0.6 + 0.9) / 3])  # mean-pooled


def test_bme_batch_pools_each_doc_independently():
    fake = _FakeScorer({"x": 0.3, "A": 0.0, "B": 0.6, "C": 0.9})
    short = "x" * 100
    long = "A" * CHUNK_CHARS + "B" * CHUNK_CHARS + "C" * CHUNK_CHARS
    out = score_bme(_as_scorer(fake), [short, long])
    # all 1 + 3 chunks scored in a single batched call; spans map back per doc
    assert len(fake.calls) == 1 and len(fake.calls[0]) == 4
    assert out.tolist() == pytest.approx([0.3, (0.0 + 0.6 + 0.9) / 3])


def test_bme_window_count_switches_at_chunk_boundary():
    fake = _FakeScorer(default=0.5)
    score_bme(_as_scorer(fake), ["y" * CHUNK_CHARS])  # == threshold
    score_bme(_as_scorer(fake), ["y" * (CHUNK_CHARS + 1)])  # one char over
    assert len(fake.calls[0]) == 1  # <= CHUNK_CHARS -> single window
    assert len(fake.calls[1]) == 3  # > CHUNK_CHARS  -> begin/middle/end


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


# ---------- score: source recovered from the input file path ----------


def test_output_paths_nest_under_source_only_when_sharing_a_prefix():
    # several sources share one prefix -> nest under <source>/ to keep them apart
    main, samp = _output_paths("s3://b/scored", "cp/foodista", "part-0.parquet", nest_by_source=True)
    assert main == "s3://b/scored/cp/foodista/outputs/main/part-0.parquet"
    assert samp == "s3://b/scored/cp/foodista/outputs/samples/part-0.parquet"
    # single source: the prefix is already that source's dir (e.g. a per-source step),
    # so write straight under it -- no redundant <source>/ nesting
    main1, samp1 = _output_paths("s3://b/scored/cp/foodista", "cp/foodista", "part-0.parquet", nest_by_source=False)
    assert main1 == "s3://b/scored/cp/foodista/outputs/main/part-0.parquet"
    assert samp1 == "s3://b/scored/cp/foodista/outputs/samples/part-0.parquet"


def test_source_of_recovers_multi_segment_source_from_path():
    sp = "s3://bucket/marin/datakit/sample_100b_abc"
    # source may itself contain slashes (e.g. finepdfs/spa_Latn) -> split only on /outputs/main/
    assert _source_of(f"{sp}/cp/foodista/outputs/main/part-0.parquet", sp) == "cp/foodista"
    assert _source_of(f"{sp}/finepdfs/spa_Latn/outputs/main/data-00003.parquet", sp) == "finepdfs/spa_Latn"
    # trailing slash on the prefix must not change the recovered source
    assert _source_of(f"{sp}/hplt_v3/outputs/main/x.parquet", sp + "/") == "hplt_v3"
