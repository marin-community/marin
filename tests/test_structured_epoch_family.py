# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
    GenericFamilyRetainedTotalSurrogate,
    load_generic_family_packet,
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
