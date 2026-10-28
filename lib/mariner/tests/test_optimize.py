# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest
from mariner import optimize
from mariner.model import Head, ObjectiveFit, Shape, TaskFit
from mariner.optimize import MIXTURE_BLOCK_SIZE, propose


def test_propose_returns_grid_mixture_no_worse_than_proportional(synthetic_problem):
    swarm, _, _, truth = synthetic_problem
    proposal = propose(truth, swarm.weights)
    assert np.issubdtype(proposal.counts.dtype, np.integer)
    assert proposal.counts.min() >= 0 and proposal.counts.sum() == MIXTURE_BLOCK_SIZE
    proportional = swarm.weights[swarm.calibration]
    assert truth.predict(proposal.weights)[0] <= truth.predict(proportional)[0]


def test_propose_reports_the_model_s_predictions_under_a_kl_penalty(synthetic_problem):
    swarm, _, _, truth = synthetic_problem
    proposal = propose(truth, swarm.weights, kl=0.1)
    # The penalty steers the search; the diagnostics predict the objective at its continuous and rounded mixtures.
    assert proposal.summary["continuous_prediction"] == truth.predict(proposal.continuous)[0]
    assert proposal.summary["runtime_prediction"] == truth.predict(proposal.weights)[0]


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


def test_propose_rejects_a_cap_the_block_grid_cannot_fill_before_solving(monkeypatch):
    # At 3 epochs per unit weight, a cap of 1 epoch bounds each of three buckets at a third of the mixture: the bounds
    # sum to one, but each bucket holds at most floor(2048 / 3) = 682 blocks, 2,046 of the 2,048.
    head = Head(math.log(0.3), np.full(6, 0.2), 0.8)
    task = TaskFit("t0", Shape(1.0, 1.0, 2.0), 0.0, 1.0, False, 0.0, head)
    fit = ObjectiveFit("o", ("b0", "b1", "b2"), np.full(3, 3.0), np.ones(1), (task,), {})
    monkeypatch.setattr(optimize, "minimize", lambda *args, **kwargs: pytest.fail("SLSQP ran"))
    with pytest.raises(ValueError, match="2046 of the 2048 blocks"):
        propose(fit, np.full((4, 3), 1 / 3), cap=1.0)
