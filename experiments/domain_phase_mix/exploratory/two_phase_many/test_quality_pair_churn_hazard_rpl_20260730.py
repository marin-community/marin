# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    quality_pair_churn_hazard_rpl_20260730 as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    retained_power_law_model_20260728 as rpl,
)


def domain_names() -> tuple[str, ...]:
    return (
        "dolma3_cc/games_high",
        "dolma3_cc/games_low",
        "dolma3_cc/health_high",
        "dolma3_cc/health_low",
        "dolma3_arxiv",
    )


def geometry() -> rpl.Geometry:
    return rpl.Geometry(
        c0=np.asarray([2.0, 3.0, 5.0, 7.0, 11.0]),
        c1=np.asarray([0.5, 0.75, 1.25, 1.75, 2.75]),
        phase_0_fraction=0.8,
        family_index=np.asarray([0, 0, 0, 0, 1]),
    )


def policies() -> np.ndarray:
    return np.asarray(
        [
            [[0.3, 0.2, 0.2, 0.1, 0.2], [0.3, 0.2, 0.2, 0.1, 0.2]],
            [[0.4, 0.1, 0.1, 0.2, 0.2], [0.1, 0.4, 0.25, 0.05, 0.2]],
            [[0.4, 0.2, 0.2, 0.1, 0.1], [0.2, 0.1, 0.4, 0.2, 0.1]],
        ],
        dtype=float,
    )


def test_quality_pair_partition_is_predeclared_by_names() -> None:
    families = candidate.quality_pair_families(domain_names())
    np.testing.assert_array_equal(families, np.asarray([0, 0, 1, 1, 2]))


def test_zero_hazard_is_exactly_rpl() -> None:
    families = candidate.quality_pair_families(domain_names())
    for shape in rpl.shape_grid():
        np.testing.assert_array_equal(
            candidate.design_matrix(policies(), geometry(), shape, 0.0, families),
            rpl.design_matrix(policies(), geometry(), shape),
        )


def test_tied_policy_is_exactly_rpl() -> None:
    families = candidate.quality_pair_families(domain_names())
    tied = policies()[[0]]
    for hazard in (0.5, 1.0, 2.0, 4.0):
        for shape in rpl.shape_grid():
            np.testing.assert_array_equal(
                candidate.design_matrix(tied, geometry(), shape, hazard, families),
                rpl.design_matrix(tied, geometry(), shape),
            )


def test_singleton_partition_is_exactly_rpl_and_validates_hazard() -> None:
    families = np.arange(policies().shape[-1])
    shape = rpl.shape_grid()[0]
    np.testing.assert_array_equal(
        candidate.design_matrix(policies(), geometry(), shape, 1.0, families),
        rpl.design_matrix(policies(), geometry(), shape),
    )
    with pytest.raises(ValueError, match="nonnegative"):
        candidate.design_matrix(policies(), geometry(), shape, -0.1, families)


def test_churn_is_bounded_and_phase_swap_invariant() -> None:
    families = candidate.quality_pair_families(domain_names())
    forward = candidate.conditional_family_churn(policies(), families)
    reverse = candidate.conditional_family_churn(policies()[:, ::-1, :], families)
    assert np.all((0.0 <= forward) & (forward <= 1.0))
    np.testing.assert_allclose(forward, reverse, rtol=0.0, atol=1e-15)


def test_coherent_pair_mass_shift_has_zero_churn() -> None:
    families = candidate.quality_pair_families(domain_names())
    coherent = policies()[[2]]
    churn = candidate.conditional_family_churn(coherent, families)
    np.testing.assert_allclose(churn, np.zeros_like(churn), rtol=0.0, atol=1e-15)


def test_within_pair_quality_reallocation_has_positive_churn() -> None:
    families = candidate.quality_pair_families(domain_names())
    churn = candidate.conditional_family_churn(policies()[[1]], families)
    assert churn[0, 0] > 0.0
    assert churn[0, 1] > 0.0
    assert churn[0, 2] == 0.0


def test_bucket_churn_broadcasts_pair_value() -> None:
    families = candidate.quality_pair_families(domain_names())
    family = candidate.conditional_family_churn(policies(), families)
    bucket = candidate.bucket_churn(policies(), families)
    np.testing.assert_array_equal(bucket[:, 0], family[:, 0])
    np.testing.assert_array_equal(bucket[:, 1], family[:, 0])
    np.testing.assert_array_equal(bucket[:, 2], family[:, 1])
    np.testing.assert_array_equal(bucket[:, 3], family[:, 1])
    np.testing.assert_array_equal(bucket[:, 4], family[:, 2])


def test_only_benefit_block_changes() -> None:
    families = candidate.quality_pair_families(domain_names())
    shape = rpl.Shape(1.0, 0.1, 1.5, 0.0, 5.0, 2.0, True)
    baseline = rpl.design_matrix(policies(), geometry(), shape)
    churned = candidate.design_matrix(
        policies(),
        geometry(),
        shape,
        1.0,
        families,
    )
    benefit_columns = len(np.unique(geometry().families)) + len(geometry().excess_domains)
    np.testing.assert_array_equal(
        churned[:, benefit_columns:],
        baseline[:, benefit_columns:],
    )


def test_hazard_reduces_retained_early_state() -> None:
    families = candidate.quality_pair_families(domain_names())
    baseline = candidate.retained_share(
        policies(),
        geometry(),
        retention=5.0,
        late_multiplier=2.0,
        churn_hazard=0.0,
        churn_families=families,
    )
    churned = candidate.retained_share(
        policies(),
        geometry(),
        retention=5.0,
        late_multiplier=2.0,
        churn_hazard=1.0,
        churn_families=families,
    )
    assert np.all(churned <= baseline + 1e-15)
    assert np.any(churned < baseline - 1e-12)
