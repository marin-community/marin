# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Design blocks added in round 4 of the single-phase Observatory benchmark."""

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_round4_pooled_law_20260903 as pooled,
)

BUCKETS = ("dolma3_cc/art_high", "dolma3_cc/art_low", "dolmino_synth_math", "dolma3_wikipedia")
SHAPE = {"rate": 0.25, "power": 1.0, "threshold": 3.0}


def _features() -> models.Features:
    rng = np.random.default_rng(0)
    weights = rng.dirichlet(np.ones(4), size=12)
    inventory = np.array([20.0, 60.0, 300.0, 1500.0])
    return models.features_from_panel(weights, inventory, BUCKETS, early_fraction=None, label="round4")


def _design(**overrides) -> models.Design:
    options = models.FamilyOptions(
        **{"family_signal": "none", "harm": "softplus_bucket", "benefit": "weibull", **overrides}
    )
    shape = {**SHAPE, "onset_slope": 0.75, "harm_shrink": 10.0, "interaction_shrink": 1.0}
    return models.family_design(_features(), shape, options)


def test_share_penalty_columns_are_the_bucket_weights():
    features = _features()
    design = _design(share_penalty=True)
    columns = [index for index, name in enumerate(design.names) if name.startswith("bucket_share_penalty:")]
    assert len(columns) == 4
    assert np.allclose(design.values[:, columns], features.weights)


def test_onset_covariate_shifts_the_threshold_per_bucket():
    features = _features()
    plain = _design()
    shifted = _design(harm_onset_covariate="log_inventory")
    covariate = models.onset_covariate(features, "log_inventory")
    assert covariate.shape == (4,) and abs(float(np.median(covariate))) < 1e-12
    for index in range(4):
        column = plain.names.index(f"bucket_overexposure:{index}")
        expected = models.softplus_harm(features.exposures[:, index], SHAPE["threshold"] + 0.75 * covariate[index])
        assert np.allclose(shifted.values[:, column], expected)
        if abs(covariate[index]) > 1e-9:
            assert not np.allclose(shifted.values[:, column], plain.values[:, column])


def test_quality_covariate_is_zero_for_undeclared_buckets_and_centred_over_declared_ones():
    covariate = models.onset_covariate(_features(), "quality")
    assert covariate[2] == 0.0 and covariate[3] == 0.0
    assert abs(covariate[0] + covariate[1]) < 1e-12 and covariate[0] != covariate[1]


def test_hierarchical_harm_has_a_shared_column_and_signed_shrunk_deviations():
    design = _design(harm="softplus_bucket_hierarchical")
    shared = design.names.index("shared_overexposure")
    plus = [design.names.index(f"bucket_overexposure_plus:{index}") for index in range(4)]
    minus = [design.names.index(f"bucket_overexposure_minus:{index}") for index in range(4)]
    assert np.allclose(design.values[:, shared], design.values[:, plus].sum(axis=1))
    assert np.allclose(design.values[:, minus], -design.values[:, plus])
    assert design.ridge[shared] == 1.0 and all(design.ridge[column] == 10.0 for column in plus + minus)


def test_cc_hub_interaction_uses_only_common_crawl_buckets_as_the_hub():
    features = _features()
    design = _design(interaction="cc_hub")
    signal = models.weibull_response(features.exposures, SHAPE["rate"], SHAPE["power"])
    hub = signal[:, :2].sum(axis=1, keepdims=True)
    plus = [design.names.index(f"interaction:hub_plus:{index}") for index in range(4)]
    assert np.allclose(design.values[:, plus], hub * signal)
    total = _design(interaction="total_hub")
    assert np.allclose(total.values[:, plus], signal.sum(axis=1, keepdims=True) * signal)


def test_unique_token_benefit_saturates_at_the_bucket_inventory():
    features = _features()
    unique = models.unique_token_input(features)
    assert np.all(unique <= features.weights * models.UNIQUE_TOKEN_SCALE + 1e-12)
    assert np.all(unique <= models.UNIQUE_TOKEN_SCALE / features.inventory[None, :] + 1e-12)
    design = _design(benefit_input="unique_tokens")
    expected = -models.weibull_response(unique, SHAPE["rate"], SHAPE["power"])
    columns = [design.names.index(f"bucket_signal:{index}") for index in range(4)]
    assert np.allclose(design.values[:, columns], expected)


def test_existing_options_are_unchanged_by_the_new_defaults():
    baseline = models.family_design(
        _features(), SHAPE, models.FamilyOptions(family_signal="none", harm="softplus_bucket", benefit="weibull")
    )
    assert baseline.names == tuple(f"bucket_signal:{index}" for index in range(4)) + tuple(
        f"bucket_overexposure:{index}" for index in range(4)
    )
    with pytest.raises(ValueError):
        _design(benefit_input="tokens")


def test_pooled_effective_data_model_recovers_a_separable_surface():
    rng = np.random.default_rng(1)
    weights = rng.dirichlet(np.ones(4), size=60)
    inventory = np.array([20.0, 60.0, 300.0, 1500.0])
    features = models.features_from_panel(weights, inventory, BUCKETS, early_fraction=None, label="pooled")
    # Independent oracle: the law written out directly, not through the module's predict helper.
    unique = np.minimum(weights, 1.0 / inventory[None, :]) * pooled.POOLED_UNIQUE_SCALE
    epochs = weights * inventory[None, :]
    credit = 6.0 * (1.0 - np.exp(-np.maximum(epochs - 1.0, 0.0) / 6.0))
    tau, gamma, intercept = np.array([1.0, 2.0, 0.5, 0.2]), np.array([0.0, 0.1, 0.0, 0.05]), 1.0
    effective = (unique * (1.0 + credit)) @ tau
    response = intercept + effective ** (-0.3) + weights @ gamma + rng.normal(0.0, 1e-4, size=60)
    model = pooled.PooledEffectiveDataModel("test", alphas=(0.3,), repetition_scales=(6.0,), ridge_grid=(0.0,))
    rows = np.arange(60)
    inner = ((rows[:40], rows[40:]), (rows[20:], rows[:20]))
    fitted = model.fit(features, response, rows, inner, 0)
    prediction = model.predict(fitted, features, rows)
    assert np.sqrt(np.mean((prediction - response) ** 2)) < 1e-3
    assert fitted.shape == {"alpha": 0.3, "repetition_scale": 6.0}
    assert np.all(fitted.head.coefficients >= 0.0)
    assert np.allclose(fitted.head.coefficients[:4] / fitted.head.coefficients[:4].max(), tau / tau.max(), atol=0.05)
    assert model.nonlinear_dof(features) == 2


def test_pooled_law_jacobian_matches_finite_differences():
    rng = np.random.default_rng(3)
    weights = rng.dirichlet(np.ones(4), size=20)
    inventory = np.array([20.0, 60.0, 300.0, 1500.0])
    features = models.features_from_panel(weights, inventory, BUCKETS, early_fraction=None, label="jac")
    terms = pooled.effective_terms(features, 6.0)
    response = rng.normal(1.0, 0.05, size=20)
    parameters = np.concatenate([rng.uniform(0.5, 2.0, 4), rng.uniform(0.0, 0.2, 4), [0.8]])
    _fitted, jacobian = pooled.residuals_and_jacobian(terms, weights, response, 0.3, 1e-2)
    analytic = jacobian(parameters)
    step = 1e-6
    residuals = pooled.residuals_and_jacobian(terms, weights, response, 0.3, 1e-2)[0]
    numeric = np.column_stack(
        [(residuals(parameters + step * unit) - residuals(parameters - step * unit)) / (2 * step) for unit in np.eye(9)]
    )
    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-4)
