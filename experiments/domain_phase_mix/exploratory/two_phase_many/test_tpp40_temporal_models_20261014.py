# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_temporal_models_20261014 as temporal

SHAPE = {"rate": 0.5, "power": 0.7, "threshold": 2.0}


@pytest.mark.parametrize("model", temporal.MODEL_NAMES)
def test_tied_response_and_null_member_equal_native_mariner(model: str) -> None:
    rng = np.random.default_rng(20261014)
    alpha = 0.7995317529997072
    total = np.exp(rng.uniform(-4.0, 5.0, (21, 4)))
    large = np.array([True, True, False, False])
    head = temporal.mariner.Head(-0.3, np.arange(1, 9) * 0.07, 0.7)
    native = temporal.mariner.design_matrix(total, SHAPE)
    for parameters in temporal.parameter_grid(model):
        early, late = alpha * total, (1 - alpha) * total
        realized_native = temporal.mariner.design_matrix(early + late, SHAPE)
        matrix, offset = temporal.design_and_offset(early, late, alpha, SHAPE, model, parameters, large)
        np.testing.assert_array_equal(matrix, realized_native)
        np.testing.assert_array_equal(temporal.predict_head(head, matrix, offset), head.predict(realized_native))
    fractions = rng.uniform(0.0, 1.0, total.shape)
    early, late = fractions * total, (1 - fractions) * total
    matrix, offset = temporal.design_and_offset(
        early, late, alpha, SHAPE, model, temporal.parameter_grid(model)[0], large
    )
    np.testing.assert_allclose(matrix, native, rtol=3e-15, atol=3e-15)
    np.testing.assert_array_equal(offset, np.zeros(len(total)))


@pytest.mark.parametrize("multiplier", [0.25, 2.0, 16.0])
def test_common_clock_is_exactly_a_static_convex_mixture(multiplier: float) -> None:
    rng = np.random.default_rng(813)
    alpha = 0.8
    first = rng.dirichlet(np.ones(4), 20)
    second = rng.dirichlet(np.ones(4), 20)
    exposure_scale = np.array([0.25, 1.0, 10.0, 100.0])
    early = alpha * exposure_scale * first
    late = (1 - alpha) * exposure_scale * second
    mixture = (alpha * first + multiplier * (1 - alpha) * second) / (alpha + multiplier * (1 - alpha))
    effective = temporal.effective_exposure(early, late, alpha, multiplier)
    np.testing.assert_allclose(effective, exposure_scale * mixture, rtol=3e-15, atol=3e-15)
    np.testing.assert_allclose(mixture.sum(axis=1), 1.0, rtol=3e-15, atol=3e-15)
    head = temporal.mariner.Head(-0.4, np.array([0.3, 0.6, 0.2, 0.9, 0.4, 0.1, 0.01, 0.02]), 0.7)
    phase_matrix, offset = temporal.design_and_offset(
        early,
        late,
        alpha,
        SHAPE,
        "shared_clock",
        {"multiplier": multiplier},
        np.array([True, True, False, False]),
    )
    static = head.predict(temporal.mariner.design_matrix(mixture * exposure_scale, SHAPE))
    np.testing.assert_allclose(temporal.predict_head(head, phase_matrix, offset), static, rtol=3e-15, atol=3e-15)


def test_harm_persistence_preserves_cumulative_repetition_clock() -> None:
    early = np.array([[6.0, 2.0], [1.0, 3.0]])
    late = np.array([[4.0, 5.0], [7.0, 0.0]])
    alpha = 0.8
    loss = 0.5
    matrix, _ = temporal.design_and_offset(
        early,
        late,
        alpha,
        SHAPE,
        "harm_persistence",
        {"benefit_loss": 0.0, "harm_loss": loss},
        np.array([True, False]),
    )
    harm = temporal.mariner.harm
    total = early + late
    threshold = SHAPE["threshold"]
    # Independently accumulate early damage, late incremental damage, then
    # remove the tied-path departure. Replacing the late increment by h(E1)
    # silently resets repetitions and must fail this check.
    path = (1 - loss) * harm(early, threshold) + harm(total, threshold) - harm(early, threshold)
    tied_path = (1 - loss) * harm(alpha * total, threshold)
    tied_path += harm(total, threshold) - harm(alpha * total, threshold)
    expected = harm(total, threshold) + path - tied_path
    np.testing.assert_allclose(matrix[:, 2:], expected, rtol=3e-15, atol=3e-15)


def test_pool_order_distinguishes_which_pool_group_is_acquired_first() -> None:
    total = np.array([[4.0, 4.0], [4.0, 4.0]])
    early = np.array([[4.0, 0.0], [0.0, 4.0]])
    late = total - early
    feature = temporal.pool_order_feature(early, late, 0.5, SHAPE, np.array([True, False]))
    assert feature[0] > 0
    np.testing.assert_allclose(feature[0], -feature[1], rtol=3e-15, atol=3e-15)


@pytest.mark.timeout(180)
def test_complete_native_search_and_conditional_null_profile_match_standalone() -> None:
    rng = np.random.default_rng(413)
    alpha = 0.8
    scales = np.array([1.0, 8.0])
    early = alpha * scales * rng.dirichlet(np.ones(2), 24)
    late = (1 - alpha) * scales * rng.dirichlet(np.ones(2), 24)
    total = early + late
    observed = 0.6 + np.exp(-0.1 + temporal.mariner.design_matrix(total, SHAPE) @ np.array([0.5, 0.4, 0.02, 0.01]))
    observed += rng.normal(0, 0.001, len(observed))
    labels = np.arange(len(observed)) % 3
    fitted = temporal.fit_aggregate(early, late, observed, labels, alpha, np.array([8.0, 1.0]))
    splits = temporal.inner_splits(labels)
    with threadpool_limits(limits=1):
        native_shape, native_ridge, _ = temporal.mariner.select_shape_and_ridge(
            total, observed, temporal.mariner.FloorSpec(float("nan"), 0, 2.5), splits
        )
        native_matrix = temporal.mariner.design_matrix(total, native_shape)
        native_kappa, native_flat, native_score = temporal.mariner.search_kappa(
            native_matrix, observed, native_ridge, temporal.mariner.Anchor(float("nan"), 0), splits
        )
        native_head = temporal.mariner.fit_head(
            native_matrix, observed, native_ridge, temporal.mariner.FloorSpec(float("nan"), 0, native_kappa)
        )
    assert fitted.shape == native_shape
    assert fitted.ridge == native_ridge
    assert fitted.kappa == native_kappa
    assert fitted.flat_profile == native_flat
    np.testing.assert_array_equal(fitted.predict(early, late), native_head.predict(native_matrix))
    for model in temporal.MODEL_NAMES[1:]:
        extension = temporal.fit_extension(early, late, observed, labels, fitted, model)
        # Full-fit amplitudes or a full-fit floor reused in the held folds
        # produce a different null score, despite having identical features.
        assert extension.selection_profile[0]["conditional_inner_rmse"] == native_score
        replay = temporal.TemporalFit.from_json(json.loads(json.dumps(extension.to_json())))
        np.testing.assert_array_equal(replay.predict(early, late), extension.predict(early, late))
