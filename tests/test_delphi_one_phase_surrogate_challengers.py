# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_surrogates_20260824 as single_phase,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_one_phase_surrogate_challengers_20260831 as challengers,
)


def test_aggregate_linear_design_depends_only_on_total_exposure():
    panel = single_phase.one_phase_panel("delphi_3e18")
    model = challengers.aggregate_linear_model(panel)
    shape = next(iter(model.shapes()))
    tied = panel.subset(np.arange(len(panel)) == 0)

    delta = np.zeros(len(panel.buckets))
    delta[:2] = (1e-5, -1e-5)
    ratio = float(panel.c0[0] / panel.c1[0])
    reordered = dataclasses.replace(
        tied,
        phase0=tied.phase0 + delta,
        phase1=tied.phase1 - ratio * delta,
    )

    assert np.allclose(tied.epochs, reordered.epochs, atol=1e-12)
    assert np.allclose(model.build(tied, shape).matrix, model.build(reordered, shape).matrix, atol=1e-12)


def test_runtime_refinement_reaches_quadratic_grid_optimum():
    target_counts = np.asarray([400, 700, 948], dtype=np.int64)
    initial_counts = np.asarray([500, 600, 948], dtype=np.int64)
    maximum_counts = np.full(3, challengers.MIXTURE_BLOCK_SIZE, dtype=np.int64)
    target = target_counts / challengers.MIXTURE_BLOCK_SIZE

    def predict(weights: np.ndarray) -> np.ndarray:
        return np.square(weights - target).sum(axis=1)

    refined, steps = challengers.refine_runtime_counts(predict, initial_counts, maximum_counts)

    assert np.array_equal(refined, target_counts)
    assert steps == 100
