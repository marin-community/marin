# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the per-type quality calibration.

The property worth protecting is the one that decided the design: calibration
corrects a type's *offset* without forcing every group to the same shape. Bucketing
by quantile within a group would do the opposite — it equalizes mass, so a group
that is genuinely mostly junk still surrenders its top fifth to the top bucket. On
real data that ranks a quality-3.0 document above a quality-4.0 one, which is why
these tests assert both halves: the compressed type recovers, and the poor type
does not.
"""

import numpy as np

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.cluster.quality.fast_transformer.calibrate import (
    MIN_PER_LEVEL,
    apply_calibration,
    calibration_knots,
    per_type_knots,
)


def _levels(rng, n, weights):
    """``n`` oracle levels drawn 1..5 with the given relative weights."""
    p = np.array(weights, dtype=float)
    return rng.choice([1, 2, 3, 4, 5], size=n, p=p / p.sum()).astype(float)


def _raw_from(levels, rng, *, scale, offset):
    """Raw scores that track quality, on a type-specific scale."""
    return offset + scale * (levels + rng.normal(0, 0.25, size=len(levels)))


def _top_share(raw, types, knots):
    cal = apply_calibration(raw, types, knots)
    return {t: float((np.digitize(cal[types == t], BUCKET_EDGES) == 4).mean()) for t in set(types.tolist())}


def test_calibration_lifts_a_compressed_type_onto_the_shared_scale():
    """A type whose scores are squashed low still reaches the top bucket.

    This is the agentic case: its documents span the full quality range but its raw
    scores sit in a narrow, low band, so a single global remap fitted mostly on
    prose leaves even its best work out of the top bucket.
    """
    rng = np.random.default_rng(0)
    spread = [1, 1, 1, 1, 1]
    prose_levels, squashed_levels = _levels(rng, 3000, spread), _levels(rng, 3000, spread)
    raw = np.concatenate(
        [
            _raw_from(prose_levels, rng, scale=1.0, offset=0.0),
            _raw_from(squashed_levels, rng, scale=0.25, offset=-2.0),
        ]
    )
    levels = np.concatenate([prose_levels, squashed_levels])
    types = np.array(["prose"] * 3000 + ["squashed"] * 3000)

    global_only = _top_share(raw, types, calibration_knots(raw, levels))
    assert global_only["squashed"] < 0.02, "precondition: one global remap should strand the squashed type"

    per_type = _top_share(raw, types, per_type_knots(raw, levels, types, min_per_type=400))
    assert per_type["squashed"] > 0.10
    assert abs(per_type["squashed"] - per_type["prose"]) < 0.10


def test_calibration_does_not_promote_a_genuinely_poor_type():
    """A type that is mostly junk keeps a small top-bucket share.

    Quantiles inside a group would hand this type the same share as any other,
    because they equalize mass. Calibration maps by quality level, so a type without
    much good work does not acquire any.
    """
    rng = np.random.default_rng(1)
    good_levels = _levels(rng, 3000, [1, 1, 1, 3, 4])
    poor_levels = _levels(rng, 3000, [6, 4, 1, 1, 1])
    raw = np.concatenate(
        [_raw_from(good_levels, rng, scale=1.0, offset=0.0), _raw_from(poor_levels, rng, scale=1.0, offset=0.0)]
    )
    levels = np.concatenate([good_levels, poor_levels])
    types = np.array(["good"] * 3000 + ["poor"] * 3000)

    share = _top_share(raw, types, per_type_knots(raw, levels, types, min_per_type=400))
    assert share["poor"] < share["good"] / 2, "a mostly-junk type must not be lifted to parity"
    assert share["poor"] < 0.15


def test_a_type_below_the_support_floor_falls_back_to_the_default():
    """Too few labels means the global remap, not cutpoints fitted on noise."""
    rng = np.random.default_rng(2)
    many = _levels(rng, 2000, [1, 1, 1, 1, 1])
    few = _levels(rng, 50, [1, 1, 1, 1, 1])
    raw = np.concatenate([_raw_from(many, rng, scale=1.0, offset=0.0), _raw_from(few, rng, scale=1.0, offset=0.0)])
    levels = np.concatenate([many, few])
    types = np.array(["big"] * 2000 + ["tiny"] * 50)

    knots = per_type_knots(raw, levels, types, min_per_type=400)
    assert "big" in knots["types"]
    assert "tiny" not in knots["types"]

    # An unfitted type must still be scored, using the default remap.
    out = apply_calibration(raw, types, knots)
    assert np.isfinite(out).all()


def test_apply_calibration_accepts_a_global_calibration_unchanged():
    """Callers should not have to branch on which calibration shape they were given."""
    rng = np.random.default_rng(3)
    levels = _levels(rng, 500, [1, 1, 1, 1, 1])
    raw = _raw_from(levels, rng, scale=1.0, offset=0.0)
    knots = calibration_knots(raw, levels)
    assert np.allclose(apply_calibration(raw, None, knots), np.interp(raw, knots["xk"], knots["yk"]))


def test_a_thin_oracle_level_does_not_place_a_cutpoint():
    """A type can be large while one of its levels is not, and one thin level is enough.

    Math carried 5,179 labels but only 20 at level 1. Its bottom cutpoint, a median
    over those 20, landed above the 10th percentile of math's own scores and sent
    half of all math — worked geometry, algebra tutorials — to the bottom bucket.
    The type-level count never showed it, because the type was not small.
    """
    rng = np.random.default_rng(7)
    # A type skewed high, exactly like math: almost no low-quality examples.
    skewed = _levels(rng, 4000, [1, 20, 60, 200, 400])
    broad = _levels(rng, 4000, [1, 1, 1, 1, 1])
    raw = np.concatenate([_raw_from(skewed, rng, scale=1.0, offset=0.0), _raw_from(broad, rng, scale=1.0, offset=0.0)])
    levels = np.concatenate([skewed, broad])
    types = np.array(["skewed"] * 4000 + ["broad"] * 4000)

    knots = per_type_knots(raw, levels, types, min_per_type=400)
    share = _top_share(raw, types, knots)
    assert share["skewed"] > 0.30, "a type that is mostly excellent must not be dumped in low buckets"

    # The thin bottom level must borrow the global cutpoint rather than fit its own.
    thin = int((skewed == 1).sum())
    assert thin < MIN_PER_LEVEL, "precondition: level 1 is thin for the skewed type"
    assert np.isclose(knots["types"]["skewed"]["xk"][1], knots["default"]["xk"][1], atol=1e-9)


def test_bottom_bucket_is_not_where_a_high_quality_type_lands():
    """The failure as a user would see it: correctly typed excellent work scoring zero."""
    rng = np.random.default_rng(8)
    skewed = _levels(rng, 3000, [1, 20, 60, 200, 400])
    broad = _levels(rng, 3000, [1, 1, 1, 1, 1])
    raw = np.concatenate([_raw_from(skewed, rng, scale=1.0, offset=0.0), _raw_from(broad, rng, scale=1.0, offset=0.0)])
    levels = np.concatenate([skewed, broad])
    types = np.array(["skewed"] * 3000 + ["broad"] * 3000)
    cal = apply_calibration(raw, types, per_type_knots(raw, levels, types, min_per_type=400))
    buckets = np.digitize(cal, BUCKET_EDGES)
    bottom = float((buckets[(types == "skewed") & (levels >= 4)] == 0).mean())
    assert bottom < 0.02, f"{bottom:.1%} of the skewed type's good work landed in the bottom bucket"
