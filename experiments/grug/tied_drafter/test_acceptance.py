# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.grug.tied_drafter.acceptance import (
    DecisionReason,
    assess_candidate,
    fit_progress_forecast,
    grouped_loss,
    local_progress_forecast,
)


def test_forecast_and_acceptance_distinguish_cost_covering_from_small_or_noisy_gain():
    steps = np.array([64, 68, 72, 80, 96, 128])
    offsets = np.arange(8) * 0.01
    control = 5 + offsets[None, :] - 0.01 * steps[:, None]
    forecast = fit_progress_forecast(steps, control, confidence=0.95)
    np.testing.assert_allclose(forecast.loss_per_step, 0.01, atol=1e-14)
    before = control[-1]
    common = dict(
        overhead_equivalent_steps=16 / 8 + 2 / 3,
        assumed_acceptance=0.8,
        repeatability_floor=1e-6,
        confidence=0.95,
        candidate_scales=2,
    )
    accepted = assess_candidate(before, before - 0.04, forecast, **common)
    assert accepted.accepted
    np.testing.assert_allclose(accepted.estimated_equivalent_steps, 4, atol=1e-12)
    too_small = assess_candidate(before, before - 0.02, forecast, **common)
    assert too_small.reason == DecisionReason.COST
    noisy = assess_candidate(before, before - 0.04 + np.tile([-0.1, 0.1], 4), forecast, **common)
    assert noisy.reason == DecisionReason.UNCERTAIN


def test_grouped_loss_uses_scored_tokens_instead_of_equal_weighting_sequences():
    losses = np.array([2, 6, 3, 9])
    counts = np.array([3, 1, 1, 3])
    np.testing.assert_array_equal(grouped_loss(losses, counts, sequences_per_group=2), [3, 7.5])


def test_decelerating_control_curve_changes_step_equivalence_at_arrival():
    steps = np.array([0, 10, 20])
    offsets = np.arange(8) * 0.02
    control = np.array([5.0, 4.0, 3.9])[:, None] + offsets[None, :]
    curve = [fit_progress_forecast(steps[i : i + 2], control[i : i + 2], confidence=0.975) for i in range(2)]
    current = local_progress_forecast(curve, step=12, lookahead=8)
    before = 3.98 + offsets
    decision = assess_candidate(
        before,
        before - 0.04,
        current,
        overhead_equivalent_steps=2,
        assumed_acceptance=1,
        repeatability_floor=1e-6,
        confidence=0.975,
        candidate_scales=1,
    )
    assert decision.accepted
    np.testing.assert_allclose(decision.estimated_equivalent_steps, 4, atol=1e-12)
    crossing = local_progress_forecast(curve, step=8, lookahead=8)
    np.testing.assert_allclose(crossing.upper_loss_per_step, 0.1, atol=1e-12)
