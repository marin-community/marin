# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fast-transformer quality scorer's two algorithmic contracts:

- ``scorer.score_bme`` — whole-doc (begin/middle/end) token-window coverage +
  mean-pooling, the fix for scoring long docs on a truncated lead / prefix-degenerate
  sources.
- ``calibrate.fit_cutpoints`` / ``fit_calibration`` — the monotonic cutpoint remap
  that makes the fixed 0.2-bucket quantization recover the oracle quality level,
  and ``Calibration``'s global vs per-type routing.
- the knot guards: a curve ``np.interp`` cannot be trusted with (non-increasing
  ``xk``) is refused at construction and at load, the fitter's anchors cannot land
  inside the cutpoints, and the one released file known to carry an inverted knot
  loads only under its own digest, with a warning.

Both use a deterministic fake scorer / synthetic labels, so no model or I/O is needed.
"""

import logging
from itertools import pairwise
from typing import cast

import numpy as np
import pytest

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.cluster.quality.fast_transformer.calibrate import (
    KNOWN_INVERTED_KNOT_DIGESTS,
    Calibration,
    Curve,
    anchored_knots,
    fit_calibration,
    fit_cutpoints,
)
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
    curve = fit_calibration(raw, levels).default
    xk, yk = curve.xk, curve.yk

    assert yk == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert len(xk) == 6
    # np.interp requires strictly increasing knots
    assert all(b > a for a, b in pairwise(xk))
    # each oracle level's median maps into (within one of) the matching bucket
    for level in (1, 2, 3, 4, 5):
        bucket = int(np.digitize(np.interp(level / 10.0, xk, yk), BUCKET_EDGES))
        assert abs(bucket - (level - 1)) <= 1


# ---------- Calibration: global vs per-type routing ----------

FLAT_JSON = {"xk": [0.0, 0.5, 1.0], "yk": [0.0, 0.2, 1.0]}
PER_TYPE_JSON = {"default": FLAT_JSON, "types": {"code": {"xk": [0.0, 0.5, 1.0], "yk": [0.0, 0.8, 1.0]}}}
FLAT = Calibration.from_json(FLAT_JSON)
PER_TYPE = Calibration.from_json(PER_TYPE_JSON)


def test_global_calibration_applies_one_remap_to_every_document():
    raw = np.array([0.0, 0.5, 1.0])
    assert FLAT.apply(raw, None).tolist() == pytest.approx([0.0, 0.2, 1.0])
    # Types are ignored under a global calibration.
    assert FLAT.apply(raw, np.array(["code", "prose", "code"], dtype=object)).tolist() == pytest.approx([0.0, 0.2, 1.0])


def test_per_type_calibration_routes_by_type_and_falls_back_to_default():
    raw = np.array([0.5, 0.5, 0.5])
    types = np.array(["code", "prose", "code"], dtype=object)
    assert PER_TYPE.apply(raw, types).tolist() == pytest.approx([0.8, 0.2, 0.8])


def test_per_type_calibration_requires_types():
    with pytest.raises(ValueError, match="content type"):
        PER_TYPE.apply(np.array([0.5]), None)


@pytest.mark.parametrize("data", [FLAT_JSON, PER_TYPE_JSON], ids=["global", "per-type"])
def test_calibration_json_round_trips_both_layouts(data):
    assert Calibration.from_json(data).to_json() == data


# ---------- knot guards: np.interp needs strictly increasing xk ----------

# The ``agentic`` curve of the released ``nemotron_88k/calib_bme.json``: the floor
# anchor (raw.min() - 1e-6 under the old fitter) landed above the first cutpoint.
INVERTED_AGENTIC = {
    "xk": [
        0.31555674807929995,
        0.28216151148080826,
        0.477378249168396,
        0.5607410967350006,
        0.7496630549430847,
        0.886433349681854,
    ],
    "yk": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
}


def test_curve_refuses_non_increasing_knots_at_construction():
    with pytest.raises(ValueError, match=r"strictly increasing.*xk\[0\]=0.315557 >= xk\[1\]=0.282162"):
        Curve(**INVERTED_AGENTIC)


def test_from_json_validates_every_curve_and_names_the_offender():
    with pytest.raises(ValueError, match="curve 'agentic'"):
        Calibration.from_json({"default": FLAT_JSON, "types": {"agentic": INVERTED_AGENTIC}})
    with pytest.raises(ValueError, match="curve 'default'"):
        Calibration.from_json({"xk": [0.5, 0.5, 1.0], "yk": [0.0, 0.2, 1.0]})


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ({"xk": [0.1, 0.5], "yk": [0.0]}, "len"),
        ({"xk": [0.1], "yk": [0.0]}, "len"),
        ({"xk": [0.1, 0.4, 0.5], "yk": [0.0, 0.6, 0.5]}, "non-decreasing"),
        ({"xk": [0.1, float("nan")], "yk": [0.0, 1.0]}, "finite"),
    ],
    ids=["short-yk", "one-knot", "decreasing-yk", "nan"],
)
def test_from_json_refuses_malformed_curves(bad, message):
    with pytest.raises(ValueError, match=message):
        Calibration.from_json(bad)


def test_known_released_digest_loads_the_defective_file_with_a_warning(caplog):
    data = {"default": FLAT_JSON, "types": {"agentic": INVERTED_AGENTIC}}
    digest = next(iter(KNOWN_INVERTED_KNOT_DIGESTS))
    with caplog.at_level(logging.WARNING):
        calibration = Calibration.from_json(data, digest=digest)
    assert "known-defective" in caplog.text and "agentic" in caplog.text
    # Loaded as released, not repaired: the bucket output is what it always was.
    assert calibration.types["agentic"].xk == INVERTED_AGENTIC["xk"]
    with pytest.raises(ValueError, match="curve 'agentic'"):
        Calibration.from_json(data, digest="0" * 64)


def test_inverted_agentic_knots_change_scores_but_not_buckets():
    """Why the released file may load: both curves cross 0.2 at raw 0.31556 and 0.4 at 0.47738."""
    xk, yk = np.array(INVERTED_AGENTIC["xk"]), np.array(INVERTED_AGENTIC["yk"])
    raw = np.linspace(0.0, 1.0, 200_001)
    released, intended = np.interp(raw, xk, yk), np.interp(raw, np.sort(xk), yk)
    assert np.array_equal(np.digitize(released, BUCKET_EDGES), np.digitize(intended, BUCKET_EDGES))
    distorted = raw[np.abs(released - intended) > 1e-9]
    assert distorted.min() == pytest.approx(0.28216, abs=1e-4)
    assert distorted.max() == pytest.approx(0.47738, abs=1e-4)


def test_anchors_cannot_land_inside_shared_cutpoints():
    # The agentic case: shared cutpoints anchored on one type's raw range whose minimum lies
    # above the first cutpoint. The old floor (raw.min() - 1e-6) became the second-largest knot.
    shared_cuts = [0.28216151148080826, 0.477378249168396, 0.5607410967350006, 0.7496630549430847]
    raw_agentic = np.array([0.31555774807929995, 0.40, 0.55, 0.70, 0.886432])
    assert raw_agentic.min() - 1e-6 > shared_cuts[0]
    knots = anchored_knots(raw_agentic, shared_cuts)
    assert all(b > a for a, b in pairwise(knots))
    assert knots[0] < shared_cuts[0] and knots[-1] > max(raw_agentic.max(), shared_cuts[-1])


def test_global_fit_yields_increasing_knots_on_tight_clusters():
    levels = np.repeat([1, 2, 3, 4, 5], 50).astype(float)
    rng = np.random.default_rng(0)
    raw = np.clip(np.concatenate([rng.normal(c, 0.004, 50) for c in (0.40, 0.42, 0.60, 0.75, 0.90)]), 0.35, 1.0)
    assert all(b > a for a, b in pairwise(fit_calibration(raw, levels).default.xk))


def test_fitter_spreads_tied_cutpoints_into_increasing_knots():
    levels = np.repeat([1, 2, 3, 4, 5], 20).astype(float)
    raw = np.concatenate([np.full(60, 0.3), np.full(20, 0.7), np.full(20, 0.9)])
    _, cuts = fit_cutpoints(raw, levels)
    assert cuts[0] == cuts[1]  # np.maximum.accumulate ties
    knots = anchored_knots(raw, cuts)
    assert len(knots) == 6 and all(b > a for a, b in pairwise(knots))
