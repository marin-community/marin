# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Knot guards of ``calibrate.py``: a curve ``np.interp`` cannot be trusted with is refused at construction and at
load, and the fitter's anchors cannot land inside the cutpoints (the way the released ``nemotron_88k``
``calib_bme.json`` got an inverted first knot on its ``agentic`` curve)."""

import logging
from itertools import pairwise

import numpy as np
import pytest

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.cluster.quality.fast_transformer.calibrate import (
    Calibration,
    Curve,
    anchored_knots,
    fit_calibration,
    fit_cutpoints,
)

MONOTONE = {"xk": [0.05, 0.28, 0.46, 0.61, 0.75, 0.99], "yk": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]}
# The ``agentic`` curve of the released ``nemotron_88k/calib_bme.json``: the floor anchor
# (raw.min() - 1e-6 under the old fitter, taken from the type's own labels) landed above the
# first shared cutpoint.
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
        Calibration.from_json({"default": MONOTONE, "types": {"agentic": INVERTED_AGENTIC}})
    with pytest.raises(ValueError, match="curve 'default'"):
        Calibration.from_json({"xk": [0.5, 0.5, 1.0], "yk": [0.0, 0.2, 1.0]})


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        ({"xk": [0.1, 0.5], "yk": [0.0]}, "len"),
        ({"xk": [0.1], "yk": [0.0]}, "len"),
        ({"xk": [0.1, 0.4, 0.5], "yk": [0.0, 0.6, 0.5]}, "non-decreasing"),
        ({"xk": [0.1, float("nan")], "yk": [0.0, 1.0]}, "finite"),
        ({"xk": [[0.0, 1.0], [2.0, 3.0]], "yk": [[0.0, 0.5], [0.6, 1.0]]}, "one-dimensional"),
    ],
    ids=["short-yk", "one-knot", "decreasing-yk", "nan", "two-dimensional"],
)
def test_from_json_refuses_malformed_curves(bad, message):
    with pytest.raises(ValueError, match=message):
        Calibration.from_json(bad)


def test_monotone_curves_load_apply_and_round_trip():
    data = {"default": MONOTONE, "types": {"code": MONOTONE}}
    calibration = Calibration.from_json(data)
    out = calibration.apply(np.array([0.0, 0.28, 0.5, 1.0]), np.array(["code", "prose", "code", "prose"], dtype=object))
    assert out.tolist() == pytest.approx([0.0, 0.2, 0.4 + (0.5 - 0.46) / (0.61 - 0.46) * 0.2, 1.0])
    assert calibration.to_json() == data


def test_released_agentic_curve_differs_from_a_correct_fit_in_one_band():
    # What the inverted knot does: against the fit's intent (the 0.2 cut at the shared first cutpoint),
    # buckets differ exactly for raw in [cuts[0], 0.31556): 0 as released, 1 corrected. The calibrated
    # score is also off for raw in [0.31556, 0.47738] (up to 0.034), inside bucket 1 either way.
    xk, yk = np.array(INVERTED_AGENTIC["xk"]), np.array(INVERTED_AGENTIC["yk"])
    cuts = xk[1:5].tolist()
    repaired = np.array(anchored_knots(np.array([xk[0], xk[-1] - 1e-6]), cuts))
    raw = np.linspace(0.0, 1.0, 200_001)
    released, corrected = (
        np.digitize(np.interp(raw, xk, yk), BUCKET_EDGES),
        np.digitize(np.interp(raw, repaired, yk), BUCKET_EDGES),
    )
    differ = released != corrected
    assert raw[differ].min() == pytest.approx(cuts[0], abs=1e-5)
    assert raw[differ].max() == pytest.approx(0.31556, abs=1e-5)
    assert set(released[differ].tolist()) == {0} and set(corrected[differ].tolist()) == {1}


def test_anchors_cannot_land_inside_shared_cutpoints():
    # Shared cutpoints anchored on one type's raw range whose minimum lies above the first cutpoint:
    # the old floor (raw.min() - 1e-6) became the second-largest knot.
    shared_cuts = [0.28216151148080826, 0.477378249168396, 0.5607410967350006, 0.7496630549430847]
    raw_agentic = np.array([0.31555774807929995, 0.40, 0.55, 0.70, 0.886432])
    assert raw_agentic.min() - 1e-6 > shared_cuts[0]
    knots = anchored_knots(raw_agentic, shared_cuts)
    assert all(b > a for a, b in pairwise(knots))
    assert knots[0] < shared_cuts[0] and knots[-1] > max(raw_agentic.max(), shared_cuts[-1])


def test_global_fit_yields_increasing_knots_on_tight_clusters(caplog):
    levels = np.repeat([1, 2, 3, 4, 5], 50).astype(float)
    rng = np.random.default_rng(0)
    raw = np.clip(np.concatenate([rng.normal(c, 0.004, 50) for c in (0.40, 0.42, 0.60, 0.75, 0.90)]), 0.35, 1.0)
    with caplog.at_level(logging.INFO):
        curve = fit_calibration(raw, levels).default
    assert all(b > a for a, b in pairwise(curve.xk))
    assert "anchors" in caplog.text


def test_fitter_spreads_tied_cutpoints_into_increasing_knots():
    levels = np.repeat([1, 2, 3, 4, 5], 20).astype(float)
    raw = np.concatenate([np.full(60, 0.3), np.full(20, 0.7), np.full(20, 0.9)])
    _, cuts = fit_cutpoints(raw, levels)
    assert cuts[0] == cuts[1]  # np.maximum.accumulate ties
    knots = anchored_knots(raw, cuts)
    assert len(knots) == 6 and all(b > a for a, b in pairwise(knots))
