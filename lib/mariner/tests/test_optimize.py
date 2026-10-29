# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest
from mariner.model import Head, ObjectiveFit, Shape, TaskFit
from mariner.optimize import MIXTURE_BLOCK_SIZE, propose


def test_propose_returns_grid_mixture_no_worse_than_proportional(synthetic_problem):
    swarm, truth = synthetic_problem.swarm, synthetic_problem.truth
    proposal = propose(truth, swarm.weights)
    assert np.issubdtype(proposal.counts.dtype, np.integer)
    assert proposal.counts.min() >= 0 and proposal.counts.sum() == MIXTURE_BLOCK_SIZE
    proportional = swarm.weights[swarm.calibration]
    assert truth.predict(proposal.weights)[0] <= truth.predict(proportional)[0]


def test_propose_kl_penalty_pulls_the_proposal_toward_the_proportional_mixture(synthetic_problem):
    swarm, truth = synthetic_problem.swarm, synthetic_problem.truth
    proportional = swarm.weights[swarm.calibration][0]

    def distances(kl: float) -> np.ndarray:
        """Total variation distances from the proportional mixture of the continuous optimum and of its rounding."""
        proposal = propose(truth, swarm.weights, kl=kl)
        return 0.5 * np.abs(np.stack([proposal.continuous, proposal.weights]) - proportional).sum(axis=1)

    unpenalized = distances(0.0)
    assert (unpenalized > 0.5).all()  # the generating model's optimum lies far from the proportional mixture
    assert (distances(0.03) < unpenalized - 0.1).all()
    assert (distances(100.0) < 0.01).all()


def test_propose_reports_the_model_s_predictions_under_a_kl_penalty(synthetic_problem):
    swarm, truth = synthetic_problem.swarm, synthetic_problem.truth
    proposal = propose(truth, swarm.weights, kl=0.1)
    # The penalty steers the search; the diagnostics predict the objective at its continuous and rounded mixtures.
    assert proposal.summary.continuous_prediction == truth.predict(proposal.continuous)[0]
    assert proposal.summary.runtime_prediction == truth.predict(proposal.weights)[0]


def test_propose_respects_epoch_cap(synthetic_problem):
    swarm, truth = synthetic_problem.swarm, synthetic_problem.truth
    epochs = truth.epochs_per_unit_weight
    assert np.max(propose(truth, swarm.weights).weights * epochs) > 2.0  # the cap binds
    proposal = propose(truth, swarm.weights, cap=2.0)
    assert np.max(proposal.weights * epochs) <= 2.0 + 1e-12


def test_propose_rejects_cap_that_cannot_fill_a_mixture(synthetic_problem):
    swarm, truth = synthetic_problem.swarm, synthetic_problem.truth
    with pytest.raises(ValueError, match="cap"):
        propose(truth, swarm.weights, cap=0.01)


def test_propose_takes_a_cap_whose_block_maxima_just_fill_the_grid():
    # At these epochs, a cap of 0.27 epochs bounds the two buckets at 1,823 and 225 blocks, the whole grid, but the
    # weight bounds, 1823 / 2048 less an ulp and 225 / 2048, sum to one less an ulp.
    cap = 0.27
    epochs = cap * MIXTURE_BLOCK_SIZE / np.array([1823.0, 225.0])
    assert np.minimum(1.0, cap / epochs).sum() < 1
    head = Head(math.log(0.3), np.full(4, 0.2), 0.8)
    task = TaskFit("t0", Shape(1.0, 1.0, 2.0), 0.0, 1.0, False, 0.0, head)
    fit = ObjectiveFit("o", ("b0", "b1"), epochs, np.ones(1), (task,), {})
    swarm_weights = np.random.default_rng(0).dirichlet(np.ones(2), size=8)
    assert propose(fit, swarm_weights, cap=cap).counts.tolist() == [1823, 225]


def test_propose_rejects_a_cap_whose_block_maxima_fall_short_of_the_grid():
    # At 3 epochs per unit weight, a cap of 1 epoch bounds each of three buckets at a third of the mixture: the bounds
    # sum to one, but each bucket holds at most floor(2048 / 3) = 682 blocks, 2,046 of the 2,048.
    head = Head(math.log(0.3), np.full(6, 0.2), 0.8)
    task = TaskFit("t0", Shape(1.0, 1.0, 2.0), 0.0, 1.0, False, 0.0, head)
    fit = ObjectiveFit("o", ("b0", "b1", "b2"), np.full(3, 3.0), np.ones(1), (task,), {})
    with pytest.raises(ValueError, match="2046 of the 2048 blocks"):
        propose(fit, np.full((4, 3), 1 / 3), cap=1.0)
