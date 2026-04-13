# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    PENALTY_KIND_GROUP_LOG_THRESHOLD,
    PENALTY_KIND_PER_DOMAIN_LOG_THRESHOLD,
    PREMIUM_MODE_GLOBAL,
    PREMIUM_MODE_NONE,
    SIGNAL_KIND_RETAINED_TOTAL,
    SIGNAL_KIND_TOTAL_LOG,
    CCPairTotalStructuredSurrogate,
    CCPairStructuredSurrogate,
    StructuredEpochSurrogate,
    load_two_phase_many_packet,
    load_two_phase_starcoder_packet,
    optimize_cc_globalpremium_model,
    optimize_cc_pairtotal_model,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GENERIC_FAMILY_NAMES,
    GenericFamilySignalTransform,
    GenericFamilyRetainedTotalSurrogate,
    TUNED_GENERIC_FAMILY_PARAMS,
    load_generic_family_packet,
    optimize_generic_family_model,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    build_penalty_calibration_surrogate,
    penalty_calibration_variant_parameter_counts,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.intrinsic_domain_followup import (
    DEFAULT_INTRINSIC_DOMAIN_COUNT,
    IntrinsicDomainRetainedTotalSurrogate,
    IntrinsicFeatureMode,
    IntrinsicPenaltyMode,
    learn_intrinsic_group_basis,
    optimize_intrinsic_domain_model,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.phase_composition_sparse_pls import (
    load_phase_composition_packet,
    optimize_phase_composition_sparse_pls_model,
    reproduction_cv_summary,
)


def test_ccglobalpremium_retainedtotal_feature_counts_match_expected_topology():
    data = load_two_phase_many_packet()
    params = {
        "signal_kind": SIGNAL_KIND_RETAINED_TOTAL,
        "alpha": 8.0,
        "eta": 3.0,
        "lam": 1.0,
        "sig_tau": 0.0,
        "premium_mode": PREMIUM_MODE_GLOBAL,
        "pen_kind": PENALTY_KIND_GROUP_LOG_THRESHOLD,
        "tau": 1.0,
        "reg": 0.01,
    }
    model = CCPairStructuredSurrogate(data, params).fit(data.w, data.y)
    counts = model.parameter_count()

    assert len(model.singletons) == 13
    assert len(model.pairs) == 13
    assert counts.linear_coefficients == 28
    assert counts.reported_total == 29
    assert counts.total_with_shapes == 35


def test_totallog_starcoder_has_two_domain_coefficients_and_one_penalty():
    data = load_two_phase_starcoder_packet()
    params = {
        "signal_kind": SIGNAL_KIND_TOTAL_LOG,
        "pen_kind": PENALTY_KIND_PER_DOMAIN_LOG_THRESHOLD,
        "alpha": 0.316477819981146,
        "eta": 42.6237468142569,
        "lam": 1.0540148034917847,
        "tau": 6.100253846769883,
        "reg": 1.522997974471263e-08,
        "premium_mode": PREMIUM_MODE_NONE,
    }
    model = StructuredEpochSurrogate(data, params).fit(data.w, data.y)
    counts = model.parameter_count()

    assert counts.linear_coefficients == 3
    assert counts.reported_total == 4
    assert counts.total_with_shapes == 9


def test_ccglobalpremium_optimizer_returns_normalized_two_phase_schedule():
    data = load_two_phase_many_packet()
    params = {
        "signal_kind": SIGNAL_KIND_RETAINED_TOTAL,
        "alpha": 8.0,
        "eta": 3.0,
        "lam": 1.0,
        "sig_tau": 0.0,
        "premium_mode": PREMIUM_MODE_GLOBAL,
        "pen_kind": PENALTY_KIND_GROUP_LOG_THRESHOLD,
        "tau": 1.0,
        "reg": 0.01,
    }
    model = CCPairStructuredSurrogate(data, params).fit(data.w, data.y)
    _, phase0, phase1 = optimize_cc_globalpremium_model(model, data, n_random=0, seed=0)

    assert phase0.shape == (data.m,)
    assert phase1.shape == (data.m,)
    assert phase0.min() >= 0.0
    assert phase1.min() >= 0.0
    assert abs(float(phase0.sum()) - 1.0) < 1e-8
    assert abs(float(phase1.sum()) - 1.0) < 1e-8


def test_ccpairtotal_retainedtotal_feature_counts_match_expected_topology():
    data = load_two_phase_many_packet()
    params = {
        "signal_kind": SIGNAL_KIND_RETAINED_TOTAL,
        "group_signal_kind": "log_after_sum",
        "premium_mode": PREMIUM_MODE_GLOBAL,
        "diversity_mode": "none",
        "alpha": 5.693767311270728,
        "eta": 6.323564464532408,
        "lam": 0.004606280004722357,
        "tau": 1.3976070420563144,
        "reg": 2.1562923313580245e-06,
        "pen_kind": PENALTY_KIND_GROUP_LOG_THRESHOLD,
    }
    model = CCPairTotalStructuredSurrogate(data, params).fit(data.w, data.y)
    counts = model.parameter_count()

    assert len(model.singletons) == 13
    assert len(model.pairs) == 13
    assert counts.linear_coefficients == 28
    assert counts.reported_total == 29
    assert counts.total_with_shapes == 34


def test_ccpairtotal_optimizer_returns_normalized_dense_two_phase_schedule():
    data = load_two_phase_many_packet()
    params = {
        "signal_kind": SIGNAL_KIND_RETAINED_TOTAL,
        "group_signal_kind": "log_after_sum",
        "premium_mode": PREMIUM_MODE_GLOBAL,
        "diversity_mode": "none",
        "alpha": 5.693767311270728,
        "eta": 6.323564464532408,
        "lam": 0.004606280004722357,
        "tau": 1.3976070420563144,
        "reg": 2.1562923313580245e-06,
        "pen_kind": PENALTY_KIND_GROUP_LOG_THRESHOLD,
    }
    model = CCPairTotalStructuredSurrogate(data, params).fit(data.w, data.y)
    _, phase0, phase1 = optimize_cc_pairtotal_model(model, data, n_random=0, seed=0)

    assert phase0.shape == (data.m,)
    assert phase1.shape == (data.m,)
    assert phase0.min() >= 0.0
    assert phase1.min() >= 0.0
    assert abs(float(phase0.sum()) - 1.0) < 1e-8
    assert abs(float(phase1.sum()) - 1.0) < 1e-8
    assert int((phase0 < 1e-4).sum()) <= 10
    assert int((phase1 < 1e-4).sum()) <= 10


def test_generic_family_packet_and_surrogate_fit_expected_structure():
    packet = load_generic_family_packet()
    model = GenericFamilyRetainedTotalSurrogate(packet).fit(packet.base.w, packet.base.y)
    pred = model.predict(packet.base.w[:4])

    assert len(packet.singletons) == 13
    assert len(packet.pairs) == 13
    assert tuple(packet.family_map) == GENERIC_FAMILY_NAMES
    assert packet.family_map["broad_text"]
    assert packet.family_map["tech_code"]
    assert packet.family_map["reasoning"]
    assert pred.shape == (4,)


def test_generic_family_power_signal_surrogate_fit_produces_finite_predictions():
    packet = load_generic_family_packet()
    params = {
        "alpha": 0.3,
        "eta": 13.229384772843037,
        "lam": 0.035627177458741076,
        "tau": 3.2740751832677875,
        "reg": 0.0010114720923828182,
        "beta": 0.6634021668256815,
    }
    model = GenericFamilyRetainedTotalSurrogate(
        packet,
        params=params,
        signal_transform=GenericFamilySignalTransform.POWER,
    ).fit(packet.base.w, packet.base.y)
    pred = model.predict(packet.base.w[:4])

    assert pred.shape == (4,)
    assert np.isfinite(pred).all()


def test_generic_family_power_signal_optimizer_returns_normalized_schedule():
    packet = load_generic_family_packet()
    params = {
        "alpha": 0.258973,
        "eta": 6.0834,
        "lam": 0.0518,
        "tau": 1.8276,
        "reg": 1.0,
        "beta": 0.29,
    }
    model = GenericFamilyRetainedTotalSurrogate(
        packet,
        params=params,
        signal_transform=GenericFamilySignalTransform.POWER,
    ).fit(packet.base.w, packet.base.y)
    _, phase0, phase1 = optimize_generic_family_model(packet, model, n_random=0, seed=0)

    assert phase0.shape == (packet.base.m,)
    assert phase1.shape == (packet.base.m,)
    assert phase0.min() >= 0.0
    assert phase1.min() >= 0.0
    assert abs(float(phase0.sum()) - 1.0) < 1e-8
    assert abs(float(phase1.sum()) - 1.0) < 1e-8


def test_intrinsic_group_basis_has_expected_shape_and_row_normalization():
    packet = load_generic_family_packet()
    basis = learn_intrinsic_group_basis(packet, params=TUNED_GENERIC_FAMILY_PARAMS)

    assert basis.memberships.shape == (26, DEFAULT_INTRINSIC_DOMAIN_COUNT)
    assert np.allclose(basis.memberships.sum(axis=1), 1.0)
    assert len(basis.group_names) == 26


def test_intrinsic_soft_family_surrogate_fit_produces_finite_predictions():
    packet = load_generic_family_packet()
    basis = learn_intrinsic_group_basis(packet, params=TUNED_GENERIC_FAMILY_PARAMS)
    model = IntrinsicDomainRetainedTotalSurrogate(
        packet,
        basis,
        params=TUNED_GENERIC_FAMILY_PARAMS,
        feature_mode=IntrinsicFeatureMode.SOFT_FAMILY,
        penalty_mode=IntrinsicPenaltyMode.GROUP,
    ).fit(packet.base.w, packet.base.y)
    pred = model.predict(packet.base.w[:4])

    assert pred.shape == (4,)
    assert np.isfinite(pred).all()


def test_intrinsic_latent_bottleneck_optimizer_returns_normalized_schedule():
    packet = load_generic_family_packet()
    basis = learn_intrinsic_group_basis(packet, params=TUNED_GENERIC_FAMILY_PARAMS)
    model = IntrinsicDomainRetainedTotalSurrogate(
        packet,
        basis,
        params=TUNED_GENERIC_FAMILY_PARAMS,
        feature_mode=IntrinsicFeatureMode.LATENT_BOTTLENECK,
        penalty_mode=IntrinsicPenaltyMode.LATENT,
    ).fit(packet.base.w, packet.base.y)
    _, phase0, phase1 = optimize_intrinsic_domain_model(packet, model, n_random=0, seed=0)

    assert phase0.shape == (packet.base.m,)
    assert phase1.shape == (packet.base.m,)
    assert phase0.min() >= 0.0
    assert phase1.min() >= 0.0
    assert abs(float(phase0.sum()) - 1.0) < 1e-8
    assert abs(float(phase1.sum()) - 1.0) < 1e-8


def test_phase_composition_sparse_pls_reproduces_downloaded_cv_summary():
    data = load_phase_composition_packet()
    summary, model = reproduction_cv_summary(data)
    pred = model.predict(data.w[:4])

    assert summary["rows"] == 241
    assert abs(summary["cv_r2_mean"] - 0.6906) < 5e-4
    assert abs(summary["cv_spearman_mean"] - 0.8186) < 5e-4
    assert pred.shape == (4,)


def test_phase_composition_sparse_pls_optimizer_returns_normalized_schedule():
    data = load_phase_composition_packet()
    _, model = reproduction_cv_summary(data)
    _, phase0, phase1 = optimize_phase_composition_sparse_pls_model(data, model, n_random=0, seed=0)

    assert phase0.shape == (data.m,)
    assert phase1.shape == (data.m,)
    assert phase0.min() >= 0.0
    assert phase1.min() >= 0.0
    assert abs(float(phase0.sum()) - 1.0) < 1e-8
    assert abs(float(phase1.sum()) - 1.0) < 1e-8


def test_penalty_calibration_can_disable_group_penalty_when_groups_are_removed():
    packet = load_generic_family_packet()
    params = {
        "eta": 5.9558,
        "lam": 0.02468,
        "reg": 1.0e-3,
        "beta": 0.2613,
        "a_broad_text": 0.7271,
        "a_tech_code": 0.02,
        "a_reasoning": 1.5122,
        "tau_broad_text": 2.869,
        "tau_tech_code": 4.013,
        "tau_reasoning": 5.352,
    }
    counts = penalty_calibration_variant_parameter_counts(
        packet,
        "power_family_penalty",
        include_singletons=False,
        include_pairs=False,
        include_family_totals=True,
        include_family_group_penalty=False,
    )
    model = build_penalty_calibration_surrogate(
        packet,
        params=params,
        variant_name="power_family_penalty",
        include_singletons=False,
        include_pairs=False,
        include_family_totals=True,
        include_family_group_penalty=False,
    ).fit(packet.base.w, packet.base.y)
    design = model.build_design(packet.base.w[:8])
    components = model.components()

    assert counts["signal_feature_count"] == len(GENERIC_FAMILY_NAMES)
    assert counts["penalty_feature_count"] == 0
    assert counts["linear_head_param_count"] == len(GENERIC_FAMILY_NAMES) + 1
    assert counts["nonlinear_param_count"] == 7
    assert design.shape == (8, len(GENERIC_FAMILY_NAMES))
    assert set(components["family_coef"]) == set(GENERIC_FAMILY_NAMES)
    assert all(coef == 0.0 for coef in components["family_group_penalty_coef"].values())
