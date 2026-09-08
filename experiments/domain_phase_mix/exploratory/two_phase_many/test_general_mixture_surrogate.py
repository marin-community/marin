# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the bucket-general mixture surrogate.

These lock in the fix for a defect that voided a whole modelling round. The free design block holds the
intercept plus one late-share column per family, and family shares sum to one, so the block is rank
deficient by construction. Partialling it out with a plain reduced QR used a basis whose surplus column
lay entirely outside the column space, which deleted real signal and made predictions depend on the
arbitrary order the families were numbered in -- by RMS 0.090 BPB against gates of 0.008.
"""

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import general_mixture_surrogate_20260809 as model


def make_panel(n_rows: int = 60, n_buckets: int = 6, n_families: int = 3, seed: int = 0) -> model.Panel:
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(n_buckets), size=(n_rows, 2))
    return model.Panel(
        weights=weights,
        epochs_early=rng.uniform(0.5, 50.0, size=n_buckets),
        epochs_late=rng.uniform(0.5, 50.0, size=n_buckets),
        family_index=np.arange(n_buckets) % n_families,
    )


def make_shape(panel: model.Panel, rng: np.random.Generator) -> tuple[model.Shape, float]:
    return model.unpack(
        np.concatenate(
            [
                [rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(-5, -0.3), rng.uniform(0.2, 10), rng.uniform(-6, 1)],
                rng.uniform(0.005, 2.0, size=panel.n_families),
                rng.uniform(-1.0, 3.5, size=panel.n_exposure_strata()),
            ]
        ),
        panel.n_families,
        panel.n_exposure_strata(),
    )


def test_free_block_is_rank_deficient_by_exactly_one():
    """The redundancy is structural, so the solver must handle it rather than the panel avoiding it."""
    panel = make_panel()
    shape, _ = make_shape(panel, np.random.default_rng(1))
    free, _ = model.design(panel, shape)
    assert np.linalg.matrix_rank(free) == free.shape[1] - 1

    sizes = np.array([int((panel.family_index == f).sum()) for f in np.unique(panel.family_index)])
    np.testing.assert_allclose(free[:, 1:] @ sizes, 1.0, atol=1e-12)


def test_column_space_spans_exactly_the_free_columns():
    panel = make_panel()
    shape, _ = make_shape(panel, np.random.default_rng(2))
    free, _ = model.design(panel, shape)
    basis = model.column_space(free)

    assert basis.shape[1] == np.linalg.matrix_rank(free)
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10)
    # every basis direction lies IN the column space; this is what the plain QR violated
    residual = basis - free @ np.linalg.lstsq(free, basis, rcond=None)[0]
    assert np.abs(residual).max() < 1e-10


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_predictions_are_invariant_to_family_relabelling(seed):
    """Renumbering families, with per-family parameters permuted to match, is the same model."""
    panel = make_panel(seed=seed)
    rng = np.random.default_rng(100 + seed)
    shape, ridge = make_shape(panel, rng)
    response = rng.normal(size=len(panel.weights))

    def predict(p: model.Panel, s: model.Shape) -> np.ndarray:
        free, constrained = model.design(p, s)
        head, amplitudes = model.fit_head(free, constrained, response, ridge, model.pooled_width(p))
        return free @ head + constrained @ amplitudes

    order = rng.permutation(panel.n_families)
    relabelled = model.Panel(panel.weights, panel.epochs_early, panel.epochs_late, np.argsort(order)[panel.family_index])
    permuted = model.Shape(
        shape.near_horizon,
        shape.damage_horizon,
        shape.offset,
        shape.damage_exponent,
        tuple(np.asarray(shape.readout_exponent)[order]),
        shape.boundary_scale,
    )
    np.testing.assert_allclose(predict(panel, shape), predict(relabelled, permuted), atol=1e-9)
