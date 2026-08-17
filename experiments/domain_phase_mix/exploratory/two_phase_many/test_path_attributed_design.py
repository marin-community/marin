# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Structural properties the path-attributed columns must have (ATOM-012).

These lock the claims the preregistration rests on, so that a later edit cannot quietly break them: the
split sums back to the unsplit law, equal amplitudes reproduce the unsplit model exactly, the split can
tell apart policies the incumbent cannot, and the readout is finite where the split evaluates it. Numbers,
not shapes -- a test that only asserted a column count would pass on a broken decomposition.
"""

import numpy as np
import pytest

from experiments.domain_phase_mix.exploratory.two_phase_many import fit_atomic_stage1_20260811 as stage1

# gamma, log10 scale, complement exponent, log10 complement scale, complement horizon, damage exponent
THETA = np.array([0.7, 0.5, 0.5, -3.0, 0.4, 1.5])
BENEFIT_EARLY, BENEFIT_LATE, COMPLEMENT, HARM_EARLY, HARM_LATE = 1, 2, 3, 4, 5


class FakePanel:
    """A two-bucket panel with this schedule's exact geometry: 80% of tokens early, epoch rate ratio 4."""

    def __init__(self, phase_0, phase_1, rate=21.2, capacity=1.29e-3):
        self.phase_0, self.phase_1 = phase_0, phase_1
        self.epochs_phase_0 = 4.0 * rate * phase_0
        self.epochs_phase_1 = rate * phase_1
        self.complement_epochs_phase_0 = 0.8 * capacity * (1.0 - phase_0)
        self.complement_epochs_phase_1 = 0.2 * capacity * (1.0 - phase_1)
        self.frame = np.empty((len(phase_0), 0))


@pytest.fixture
def panel():
    axis = np.linspace(0.0, 1.0, 21)
    g0, g1 = np.meshgrid(axis, axis, indexing="ij")
    return FakePanel(g0.ravel(), g1.ravel())


def test_split_sums_back_to_the_unsplit_law(panel):
    """The two increments must reconstruct the total-exposure readout, or the attribution is not one."""
    split = stage1.design(panel, "path-damage", THETA)
    tied = stage1.design(panel, "path-tied", THETA)
    assert np.abs(split[:, BENEFIT_EARLY] + split[:, BENEFIT_LATE] - tied[:, 1]).max() < 1e-12
    total = stage1._hill(panel.epochs_phase_0 + panel.epochs_phase_1, THETA[5])
    assert np.abs(split[:, HARM_EARLY] + split[:, HARM_LATE] - total).max() < 1e-12


def test_equal_amplitudes_reproduce_the_unsplit_model(panel):
    """`path-tied` is `path-exposure` with the one degree of freedom under test removed, not a rewrite."""
    split = stage1.design(panel, "path-exposure", THETA)
    tied = stage1.design(panel, "path-tied", THETA)
    assert np.abs(split @ np.array([0.3, 1.7, 1.7, 0.9]) - tied @ np.array([0.3, 1.7, 0.9])).max() < 1e-12


@pytest.mark.parametrize("horizon", [0.25, 0.5, 0.75, 1.0])
def test_incumbent_cannot_tell_index_matched_policies_apart_but_the_split_can(horizon):
    """The degeneracy this replaces, measured by its only consequence that matters.

    Both `two-bucket` readouts are functions of the same scalar index, so two policies sharing that index
    receive identical columns and therefore identical predictions under EVERY coefficient vector -- which
    is exactly the inability to express a two-phase preference. Comparing least-squares directions would
    not show this: two different nonlinear functions of one index have different linear approximations.
    """
    slope = 4.0 * (1.0 - horizon) / horizon
    first = FakePanel(np.array([0.10]), np.array([0.40]))
    second = FakePanel(np.array([0.10 + 0.02]), np.array([0.40 - 0.02 * slope]))
    incumbent = [stage1.design(p, "two-bucket", np.array([0.7, -3.0, horizon, 0.5]))[:, 1:] for p in (first, second)]
    assert np.abs(incumbent[0] - incumbent[1]).max() < 1e-12

    split = [stage1.design(p, "path-damage", THETA)[:, 1:] for p in (first, second)]
    assert np.abs(split[0] - split[1]).max() > 1e-3


def test_damage_increments_separate_the_phases(panel):
    """Early and late damage must load on different shares; a shared direction would be the old term."""
    split = stage1.design(panel, "path-damage", THETA)
    early, late = _directions(panel, split[:, [HARM_EARLY, HARM_LATE]])
    assert abs(early[0]) > 0.99 and abs(early[1]) < 0.1
    assert abs(late[1]) > 0.9


def test_damage_is_inert_without_repetition(panel):
    """At no-replay exposure the damage columns are identically zero, so they cannot act as flexibility."""
    quiet = FakePanel(panel.phase_0, panel.phase_1, rate=0.007)
    assert np.abs(stage1.design(quiet, "path-damage", THETA)[:, [HARM_EARLY, HARM_LATE]]).max() == 0.0


def test_readout_is_finite_at_zero_exposure_so_the_split_cannot_build_an_edge_indicator(panel):
    """The split evaluates the benefit law at zero along a whole edge, where an offset power law spikes.

    With the offset form that spike is an indicator for that edge whose height the optimiser controls
    through the fitted floor, which is a lookup wearing a mechanism's name. The retained form is exactly
    1 there for every parameter setting, so the column stays bounded in (0, 1] whatever theta does.
    """
    for gamma, log_scale in ((4.0, -2.0), (0.01, 2.0), (2.0, 0.0)):
        columns = stage1.design(panel, "path-exposure", np.array([gamma, log_scale, 0.5, -3.0, 0.4]))
        assert columns[panel.phase_0 == 0.0, BENEFIT_EARLY].max() == pytest.approx(1.0)
        assert columns[:, BENEFIT_EARLY].min() > 0.0
        assert columns[:, BENEFIT_EARLY].max() <= 1.0


def test_exposure_amplitudes_are_signed_so_that_more_exposure_helps(panel):
    """Early column positive, late column non-positive, so a non-negative amplitude means one thing."""
    columns = stage1.design(panel, "path-exposure", THETA)
    assert columns[:, BENEFIT_EARLY].min() > 0.0
    assert columns[:, BENEFIT_LATE].max() <= 1e-15


def test_only_the_replayed_bucket_is_split(panel):
    """The complement column must be identical between the split and unsplit models, not decomposed."""
    split = stage1.design(panel, "path-exposure", THETA)
    tied = stage1.design(panel, "path-tied", THETA)
    assert np.abs(split[:, COMPLEMENT] - tied[:, 2]).max() == 0.0


def _directions(panel, columns: np.ndarray) -> np.ndarray:
    """Least-squares direction of each column in the mixture square, unit-normalised."""
    design = np.column_stack([panel.phase_0 - panel.phase_0.mean(), panel.phase_1 - panel.phase_1.mean()])
    rows = []
    for index in range(columns.shape[1]):
        centred = columns[:, index] - columns[:, index].mean()
        slope, *_ = np.linalg.lstsq(design, centred, rcond=None)
        rows.append(slope / max(np.linalg.norm(slope), 1e-300))
    return np.array(rows)
