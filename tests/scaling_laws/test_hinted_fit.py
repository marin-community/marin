# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from marin.scaling_laws.hinted_fit import (
    ForecastKind,
    HintMethod,
    JointLawParams,
    LadderScores,
    ScoreCells,
    SigmoidParams,
    TwoStageParams,
    bootstrap_forecasts,
    fit_joint_law,
    fit_sigmoid_baseline,
    fit_two_stage_baseline,
)

CAPABILITIES = np.arange(18.0, 21.01, 0.5)
HINTS = np.round(np.arange(0.1, 1.01, 0.1), 4)
TARGET_CAPABILITY = 22.0

# The methods hint the same models, so their laws share the H = 0 slice and differ in hint response.
TRUE_LAWS = {
    HintMethod.MASKING: JointLawParams(
        capability_slope=1.2, hint_slope=4.0, intercept=-2.5, upper=0.9, lower=-0.05, floor=0.02
    ),
    HintMethod.TRUNCATION: JointLawParams(
        capability_slope=1.2, hint_slope=2.0, intercept=-2.5, upper=0.9, lower=-0.05, floor=0.02
    ),
}
TRUE_FORECAST = float(TRUE_LAWS[HintMethod.MASKING].predict(TARGET_CAPABILITY, 0.0))
TRUE_TWO_STAGE = TwoStageParams(
    bpb_scale=500.0, bpb_exponent=0.12, bpb_irreducible=0.25, accuracy_link=SigmoidParams(-12.0, 6.0)
)


def _grid(capabilities: np.ndarray, hints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c, h = np.meshgrid(capabilities, hints, indexing="ij")
    return c.ravel(), h.ravel()


def test_fit_joint_law_noise_free_grid_recovers_parameters():
    truth = TRUE_LAWS[HintMethod.MASKING]
    c, h = _grid(CAPABILITIES, np.r_[0.0, HINTS])

    fit = fit_joint_law(c, h, truth.predict(c, h))

    np.testing.assert_allclose(
        [fit.capability_slope, fit.hint_slope, fit.intercept, fit.upper, fit.lower, fit.floor],
        [truth.capability_slope, truth.hint_slope, truth.intercept, truth.upper, truth.lower, truth.floor],
        atol=1e-3,
    )
    assert fit.predict(TARGET_CAPABILITY, 0.0) == pytest.approx(truth.predict(TARGET_CAPABILITY, 0.0), abs=1e-4)


def test_fit_sigmoid_baseline_recovers_parameters_on_floor_heavy_ladder():
    # Most rungs sit near zero accuracy, the regime where single-start fits fail.
    truth = SigmoidParams(slope=2.0, intercept=-44.0)
    c = np.arange(18.0, 22.51, 0.5)

    fit = fit_sigmoid_baseline(c, truth.predict(c))

    assert fit.slope == pytest.approx(truth.slope, rel=1e-3)
    assert fit.intercept == pytest.approx(truth.intercept, rel=1e-3)


def test_fit_two_stage_baseline_recovers_extrapolated_accuracy():
    c = CAPABILITIES

    fit = fit_two_stage_baseline(c, TRUE_TWO_STAGE.predict_bits_per_byte(c), TRUE_TWO_STAGE.predict(c))

    target = np.array([TARGET_CAPABILITY])
    assert fit.predict_bits_per_byte(target) == pytest.approx(TRUE_TWO_STAGE.predict_bits_per_byte(target), abs=1e-3)
    assert fit.predict(target) == pytest.approx(TRUE_TWO_STAGE.predict(target), abs=1e-3)


BOOTSTRAP_HINTS = np.array([0.2, 0.5, 0.8])


def _sampled_ladder(n_problems: int, rng: np.random.Generator) -> LadderScores:
    """One Bernoulli outcome per (cell, problem) from the masking law, on a coarse hint grid to keep refits cheap."""

    def sample(params: JointLawParams, c: np.ndarray, h: np.ndarray) -> ScoreCells:
        p = params.predict(c, h)[:, None]
        scores = (rng.random((len(c), n_problems)) < p).astype(float)
        return ScoreCells(capability=c, hint=h, problem_scores=scores)

    law = TRUE_LAWS[HintMethod.MASKING]
    c, h = _grid(CAPABILITIES, BOOTSTRAP_HINTS)
    return LadderScores(
        unhinted=sample(law, CAPABILITIES, np.zeros_like(CAPABILITIES)),
        hinted={HintMethod.MASKING: sample(law, c, h)},
        bits_per_byte=TRUE_TWO_STAGE.predict_bits_per_byte(CAPABILITIES),
    )


def test_bootstrap_forecasts_intervals_cover_truth_and_narrow_with_more_problems():
    rng = np.random.default_rng(0)

    small = bootstrap_forecasts(_sampled_ladder(200, rng), TARGET_CAPABILITY, n_resamples=12, seed=1)
    large = bootstrap_forecasts(_sampled_ladder(3200, rng), TARGET_CAPABILITY, n_resamples=12, seed=1)

    for kind in (ForecastKind.MASKING, ForecastKind.HINTED_ENSEMBLE):
        small_lo, small_hi = small.interval(kind, 0.95)
        large_lo, large_hi = large.interval(kind, 0.95)
        assert small_lo <= small.point[kind] <= small_hi
        assert small_lo <= TRUE_FORECAST <= small_hi
        assert large_lo <= TRUE_FORECAST <= large_hi
        # 16x the problems should shrink the interval roughly 4x; require at least 2x.
        assert (large_hi - large_lo) < 0.5 * (small_hi - small_lo)
