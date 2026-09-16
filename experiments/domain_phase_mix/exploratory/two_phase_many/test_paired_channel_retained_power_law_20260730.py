# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    paired_channel_retained_power_law_20260730 as paired,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    design = np.asarray(
        [
            [1.0, 2.0],
            [1.5, 1.0],
            [3.0, 4.0],
        ]
    )
    target = np.asarray([1.0, 0.8, 1.2])
    keys = np.asarray(["pair", "pair", "control"])
    weights = np.asarray(
        [
            [[0.5, 0.5], [0.5, 0.5]],
            [[0.6, 0.4], [0.4, 0.6]],
            [[0.2, 0.8], [0.2, 0.8]],
        ]
    )
    return design, target, keys, weights


def test_build_channel_system_replaces_asymmetric_equation() -> None:
    design, target, keys, weights = _fixture()
    system = paired.build_channel_system(design, target, keys, weights)

    np.testing.assert_allclose(
        system.design,
        np.asarray(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [0.5, -1.0],
            ]
        ),
    )
    np.testing.assert_allclose(system.target, np.asarray([1.0, 1.2, -0.2]))
    np.testing.assert_allclose(system.intercept_column, np.asarray([1.0, 1.0, 0.0]))
    np.testing.assert_allclose(system.column_scale, np.asarray([3.0, 4.0]))
    assert system.aggregate_count == 2
    assert system.phase_count == 1
    assert system.equation_count == len(target)
    assert system.row_weights[:2].sum() == pytest.approx(1.5)
    assert system.row_weights[2:].sum() == pytest.approx(1.5)
    assert system.row_weights.sum() == pytest.approx(3.0)


def test_build_channel_system_rejects_unmatched_asymmetric_row() -> None:
    design, target, keys, weights = _fixture()
    keys[1] = "missing"

    with pytest.raises(ValueError, match="exactly one is required"):
        paired.build_channel_system(design, target, keys, weights)


def test_build_channel_system_rejects_ambiguous_tied_counterpart() -> None:
    design, target, keys, weights = _fixture()
    design = np.vstack([design, design[0]])
    target = np.append(target, target[0])
    keys = np.append(keys, "pair")
    weights = np.concatenate([weights, weights[[0]]])

    with pytest.raises(ValueError, match="2 tied counterparts"):
        paired.build_channel_system(design, target, keys, weights)
