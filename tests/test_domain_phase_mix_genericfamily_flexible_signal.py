# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_flexible_signal import (
    GenericFamilyFlexibleSignalSurrogate,
    pack_flexible_signal_params_observed_only,
    signal_transform,
    unpack_flexible_signal_params_observed_only,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GenericFamilyPacket,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    PacketData,
)
from experiments.domain_phase_mix import (
    two_phase_many_genericfamily_power_family_observed_only_trustblend_subset_optima as power_family_subset_optima,
)


def _toy_packet() -> GenericFamilyPacket:
    frame = pd.DataFrame({"run_name": ["run_0", "run_1", "run_2", "run_3"]})
    weights = np.asarray(
        [
            [[0.70, 0.20, 0.10], [0.10, 0.20, 0.70]],
            [[0.55, 0.35, 0.10], [0.20, 0.30, 0.50]],
            [[0.30, 0.50, 0.20], [0.40, 0.30, 0.30]],
            [[0.15, 0.65, 0.20], [0.55, 0.20, 0.25]],
        ],
        dtype=float,
    )
    packet = PacketData(
        frame=frame,
        name_col="run_name",
        y=np.asarray([1.20, 1.15, 1.11, 1.08], dtype=float),
        w=weights,
        m=3,
        c0=np.asarray([1.0, 0.9, 0.8], dtype=float),
        c1=np.asarray([1.1, 1.0, 0.9], dtype=float),
        domain_names=["broad_domain", "tech_domain", "reason_domain"],
    )
    return GenericFamilyPacket(
        base=packet,
        pairs=[],
        pair_topics=[],
        singletons=[0, 1, 2],
        family_map={"broad_text": [0], "tech_code": [1], "reasoning": [2]},
    )


def test_power_family_param_pack_round_trip() -> None:
    params = {
        "eta": 8.0,
        "lam": 0.05,
        "tau": 3.0,
        "reg": 1e-3,
        "beta": 0.7,
        "a_broad_text": 0.6,
        "a_tech_code": 0.2,
        "a_reasoning": 0.1,
    }
    packed = pack_flexible_signal_params_observed_only(params, "power_family")
    unpacked = unpack_flexible_signal_params_observed_only(packed, "power_family")
    for key, value in params.items():
        assert unpacked[key] == pytest.approx(value, rel=1e-6, abs=1e-9)


def test_boxcox_family_param_pack_round_trip() -> None:
    params = {
        "alpha": 7.0,
        "eta": 8.0,
        "lam": 0.05,
        "tau": 3.0,
        "reg": 1e-3,
        "beta": 0.7,
        "a_broad_text": 0.55,
        "a_tech_code": -0.18,
        "a_reasoning": 0.23,
    }
    packed = pack_flexible_signal_params_observed_only(params, "boxcox_family")
    unpacked = unpack_flexible_signal_params_observed_only(packed, "boxcox_family")
    for key, value in params.items():
        assert unpacked[key] == pytest.approx(value, rel=1e-6, abs=1e-9)


def test_power_boxcox_family_param_pack_round_trip() -> None:
    params = {
        "alpha": 7.0,
        "eta": 8.0,
        "lam": 0.05,
        "tau": 3.0,
        "reg": 1e-3,
        "beta": 0.7,
        "a_broad_text": 0.55,
        "a_tech_code": 0.08,
        "a_reasoning": 0.23,
    }
    packed = pack_flexible_signal_params_observed_only(params, "power_boxcox_family")
    unpacked = unpack_flexible_signal_params_observed_only(packed, "power_boxcox_family")
    for key, value in params.items():
        assert unpacked[key] == pytest.approx(value, rel=1e-6, abs=1e-9)


def test_family_curvature_changes_signal_by_family() -> None:
    params = {
        "a": 1.0,
        "a_broad_text": 0.5,
        "a_tech_code": 1.0,
        "a_reasoning": 2.0,
    }
    values = np.asarray([4.0], dtype=float)
    broad = signal_transform(values, params, "power", family_name="broad_text")[0]
    tech = signal_transform(values, params, "power", family_name="tech_code")[0]
    reasoning = signal_transform(values, params, "power", family_name="reasoning")[0]
    assert broad == pytest.approx(2.0)
    assert tech == pytest.approx(4.0)
    assert reasoning == pytest.approx(16.0)


def test_mixed_family_signal_changes_only_family_total_columns() -> None:
    packet = _toy_packet()
    params = {
        "alpha": 7.0,
        "eta": 8.0,
        "lam": 0.05,
        "tau": 3.0,
        "reg": 1e-3,
        "beta": 0.7,
        "a_broad_text": 0.6,
        "a_tech_code": 0.2,
        "a_reasoning": 0.1,
    }
    power_model = GenericFamilyFlexibleSignalSurrogate(
        packet,
        params=params,
        signal_kind="power",
        family_signal_kind="power",
        family_curvature=True,
    )
    mixed_model = GenericFamilyFlexibleSignalSurrogate(
        packet,
        params=params,
        signal_kind="power",
        family_signal_kind="boxcox",
        family_curvature=True,
    )
    power_design = power_model.build_design(packet.base.w)
    mixed_design = mixed_model.build_design(packet.base.w)
    np.testing.assert_allclose(mixed_design[:, :3], power_design[:, :3])
    assert not np.allclose(mixed_design[:, 3:6], power_design[:, 3:6])
    np.testing.assert_allclose(mixed_design[:, 6:], power_design[:, 6:])


def test_power_family_surrogate_builds_design_and_predicts() -> None:
    packet = _toy_packet()
    params = {
        "eta": 8.0,
        "lam": 0.05,
        "tau": 3.0,
        "reg": 1e-3,
        "beta": 0.7,
        "a_broad_text": 0.6,
        "a_tech_code": 0.2,
        "a_reasoning": 0.1,
    }
    model = GenericFamilyFlexibleSignalSurrogate(
        packet,
        params=params,
        signal_kind="power",
        family_curvature=True,
    )
    design = model.build_design(packet.base.w)
    assert design.shape == (4, 7)
    fitted = model.fit(packet.base.w, packet.base.y)
    pred = fitted.predict(packet.base.w)
    assert pred.shape == (4,)
    assert np.isfinite(pred).all()


def test_power_family_subset_run_name_formats_subset_size() -> None:
    assert (
        power_family_subset_optima.genericfamily_power_family_observed_only_trustblend_subset_optimum_run_name(20)
        == "baseline_genericfamily_power_family_trustblend_top8actual_cap_k020_uncheatable_bpb"
    )
