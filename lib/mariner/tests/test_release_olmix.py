# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from mariner.data import (
    MAIN_SWARM,
    OLMIX_CAP,
    TRAINED_OLMIX,
    objective_from_tasks,
    release_partition,
    release_swarm,
    release_tasks,
    released_olmix_laws,
    trained_counts,
)
from mariner.olmix import LogLinearLaw, OlmixProposal, fit_laws, propose

LAW_TOLERANCE = 1e-9  # agreed: refitted laws against fits_olmix_task_laws


def matched_proposal(snapshot: Path, objective: str, laws: Sequence[LogLinearLaw]) -> OlmixProposal:
    """The proposal of ``laws`` at the trained matched Olmix mixture's epoch cap and KL weight."""
    target = objective_from_tasks(release_tasks(snapshot), objective)
    epochs = release_partition(snapshot).epochs_per_unit_weight
    return propose(laws, target, epochs, cap=OLMIX_CAP, kl=TRAINED_OLMIX[objective][1])


def law_parameters(laws: Sequence[LogLinearLaw]) -> np.ndarray:
    return np.array([[law.log_c, law.huber_loss, *law.coefficients] for law in laws])


@pytest.mark.data_integration
@pytest.mark.timeout(120)
@pytest.mark.parametrize("objective", ["uncheatable", "obe_easy"])
def test_released_laws_reproduce_trained_olmix_mixture(snapshot, objective):
    proposal = matched_proposal(snapshot, objective, released_olmix_laws(snapshot, objective))
    np.testing.assert_array_equal(proposal.counts, trained_counts(snapshot, TRAINED_OLMIX[objective][0]))


@pytest.mark.data_integration
@pytest.mark.timeout(1200)  # the OlmoBaseEval Easy laws take about 9 minutes
@pytest.mark.parametrize("objective", ["uncheatable", "obe_easy"])
def test_refit_laws_equal_released_laws_and_trained_mixture(snapshot, objective):
    target = objective_from_tasks(release_tasks(snapshot), objective)
    laws = fit_laws(release_swarm(snapshot, MAIN_SWARM, objective), target)
    released = law_parameters(released_olmix_laws(snapshot, objective))
    np.testing.assert_allclose(law_parameters(laws), released, rtol=0, atol=LAW_TOLERANCE)
    proposal = matched_proposal(snapshot, objective, laws)
    np.testing.assert_array_equal(proposal.counts, trained_counts(snapshot, TRAINED_OLMIX[objective][0]))
