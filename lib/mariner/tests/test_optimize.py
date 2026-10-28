# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from mariner.optimize import MIXTURE_BLOCK_SIZE, propose


def test_propose_returns_grid_mixture_no_worse_than_proportional(synthetic_problem):
    swarm, _, _, truth = synthetic_problem
    proposal = propose(truth, swarm.weights)
    assert np.issubdtype(proposal.counts.dtype, np.integer)
    assert proposal.counts.min() >= 0 and proposal.counts.sum() == MIXTURE_BLOCK_SIZE
    proportional = swarm.weights[swarm.calibration]
    assert truth.predict(proposal.weights)[0] <= truth.predict(proportional)[0]


def test_propose_respects_epoch_cap(synthetic_problem):
    swarm, _, _, truth = synthetic_problem
    epochs = truth.epochs_per_unit_weight
    assert np.max(propose(truth, swarm.weights).weights * epochs) > 2.0  # the cap binds
    proposal = propose(truth, swarm.weights, cap=2.0)
    assert np.max(proposal.weights * epochs) <= 2.0 + 1e-12


def test_propose_rejects_cap_that_cannot_fill_a_mixture(synthetic_problem):
    swarm, _, _, truth = synthetic_problem
    with pytest.raises(ValueError, match="cap"):
        propose(truth, swarm.weights, cap=0.01)
