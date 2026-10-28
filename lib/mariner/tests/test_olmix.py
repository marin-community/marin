# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from mariner import olmix
from mariner.model import Objective, Swarm
from mariner.olmix import (
    HUBER_DELTA,
    LogLinearLaw,
    fit_laws,
    fit_log_linear,
    huber_sum,
    propose,
    quantize_under_cap,
)
from mariner.optimize import MIXTURE_BLOCK_SIZE

# The generating law y = 0.7 + exp(w @ (-1, 0.5, -0.3, 0.2)) over four buckets. Its Huber loss describes a fit, and a
# generating law has none.
FLOOR = 0.7
COEFFICIENTS = np.array([-1.0, 0.5, -0.3, 0.2])
GENERATING_LAW = LogLinearLaw(math.log(FLOOR), COEFFICIENTS, math.nan)
ONE_TASK = Objective("synthetic", ("t0",), np.ones(1))
EPOCHS_PER_UNIT_WEIGHT = np.array([0.5, 1.0, 4.0, 8.0])
# The proportional mixture at those epochs, pools in proportion to 1 / epochs: 8/27 of an epoch in every bucket.
PROPORTIONAL = np.array([16.0, 8.0, 2.0, 1.0]) / 27


@pytest.mark.parametrize(
    ("weights", "epochs", "expected"),
    [([0.5, 0.3, 0.2], [4.0, 1.0, 1.0], [512, 1127, 409]), ([0.2, 0.5, 0.3], [1.0, 4.0, 1.0], [409, 512, 1127])],
    ids=["sorted", "permuted"],
)
def test_quantize_under_cap_fills_block_without_exceeding_cap(weights, epochs, expected):
    # 512 blocks are the cap of the bucket at weight 0.5; the 513 left over go to the next-largest weight, wherever it
    # sits.
    assert quantize_under_cap(np.array(weights), np.array(epochs), cap=1.0).tolist() == expected


def test_quantize_under_cap_with_infinite_cap_floors_and_gives_remainder_to_largest_weight():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        counts = quantize_under_cap(np.array([0.5, 0.3, 0.2]), np.array([4.0, 1.0, 1.0]), cap=math.inf)
    # numpy reports the invalid cast of an infinite block maximum to integers as a RuntimeWarning.
    assert not [str(warning.message) for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert counts.tolist() == [1025, 614, 409]  # floors 1024, 614 and 409; the block left over goes to weight 0.5


def test_fit_log_linear_recovers_generating_law():
    rng = np.random.default_rng(3)
    weights = rng.dirichlet(np.ones(4), size=40)
    targets = FLOOR + np.exp(weights @ COEFFICIENTS) + rng.normal(0.0, 1e-4, size=40)
    law = fit_log_linear(weights, targets)
    fresh = rng.dirichlet(np.ones(4), size=20)
    assert np.abs(law.predict(fresh) - (FLOOR + np.exp(fresh @ COEFFICIENTS))).max() < 1e-3


def test_fit_laws_fits_each_task_in_objective_order_on_every_run():
    # Run 0, the calibration run at the proportional mixture, sits 0.05 above both laws, and the objective lists the
    # tasks in the reverse of the outcome columns' order.
    rng = np.random.default_rng(5)
    weights = np.vstack([PROPORTIONAL, rng.dirichlet(np.ones(4), size=20)])
    outcomes = pd.DataFrame(
        {"t0": FLOOR + np.exp(weights @ COEFFICIENTS), "t1": 0.4 + np.exp(weights @ np.array([0.3, -0.8, 0.1, 0.6]))}
    )
    outcomes.iloc[0] += 0.05
    swarm = Swarm(
        runs=tuple(f"run_{row:02d}" for row in range(len(weights))),
        buckets=("b0", "b1", "b2", "b3"),
        weights=weights,
        epochs_per_unit_weight=EPOCHS_PER_UNIT_WEIGHT,
        pool_fractions=np.ones_like(weights),
        outcomes=outcomes,
        calibration=np.arange(len(weights)) == 0,
    )
    objective = Objective("synthetic", ("t1", "t0"), np.full(2, 0.5))
    laws = fit_laws(swarm, objective)
    # Each law's Huber loss is its loss on every run of its own task: a law fitted to the other task, or without run 0,
    # would report another loss.
    for law, task in zip(laws, objective.tasks, strict=True):
        residuals = law.predict(weights) - outcomes[task].to_numpy()
        assert law.huber_loss == pytest.approx(huber_sum(residuals, HUBER_DELTA), rel=1e-9)


def test_olmix_proposal_respects_caps():
    # The law alone would put the whole block on bucket 0, which 0.4 epochs caps at 0.8 of the mixture; buckets 2
    # and 3 end at their caps too, and rounding each bucket to its nearest block would put bucket 2 over its cap.
    cap = 0.4
    proposal = propose([GENERATING_LAW], ONE_TASK, EPOCHS_PER_UNIT_WEIGHT, cap=cap, kl=0.01)
    assert np.max(proposal.continuous * EPOCHS_PER_UNIT_WEIGHT) == pytest.approx(cap)  # the cap binds
    assert np.issubdtype(proposal.counts.dtype, np.integer) and proposal.counts.sum() == MIXTURE_BLOCK_SIZE
    assert np.max(proposal.counts / MIXTURE_BLOCK_SIZE * EPOCHS_PER_UNIT_WEIGHT) <= cap


def test_olmix_proposal_with_infinite_cap_is_unconstrained():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        proposal = propose([GENERATING_LAW], ONE_TASK, EPOCHS_PER_UNIT_WEIGHT, cap=math.inf, kl=0.01)
    # numpy reports the invalid cast of an infinite block maximum to integers as a RuntimeWarning.
    assert not [str(warning.message) for warning in caught if issubclass(warning.category, RuntimeWarning)]
    assert proposal.counts.tolist() == [2048, 0, 0, 0]  # the law alone puts the whole block on bucket 0


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate:UserWarning")  # cvxpy's notice of SCS's status
def test_olmix_proposal_refuses_solver_solutions_outside_the_caps():
    # exp(43 w0 - 16 w1) spans 23 orders of magnitude over the simplex. CLARABEL reports "optimal" 1.2e-4 above
    # bucket 1's cap of 0.4273 and SCS "optimal_inaccurate" 0.5 away from the constraints: neither is a mixture.
    law = LogLinearLaw(0.0, np.array([43.0, -16.0]), math.nan)
    with pytest.raises(RuntimeError):
        propose([law], ONE_TASK, np.array([1.2, 2.2]), cap=0.94, kl=0.01)


def test_olmix_proposal_takes_a_cap_whose_block_maxima_just_fill_the_grid():
    # At these epochs, a cap of 0.27 epochs bounds the two buckets at 1,823 and 225 blocks, the whole grid, but the
    # weight bounds, 1823 / 2048 less an ulp and 225 / 2048, sum to one less an ulp.
    cap = 0.27
    epochs = cap * MIXTURE_BLOCK_SIZE / np.array([1823.0, 225.0])
    assert np.minimum(1.0, cap / epochs).sum() < 1
    law = LogLinearLaw(math.log(FLOOR), COEFFICIENTS[:2], math.nan)
    assert propose([law], ONE_TASK, epochs, cap=cap, kl=0.01).counts.tolist() == [1823, 225]


def test_olmix_proposal_rejects_a_cap_the_block_grid_cannot_fill_before_solving(monkeypatch):
    # At 3 epochs per unit weight, a cap of 1 epoch bounds each of three buckets at a third of the mixture: the bounds
    # sum to one, but each bucket holds at most floor(2048 / 3) = 682 blocks, 2,046 of the 2,048.
    law = LogLinearLaw(math.log(FLOOR), COEFFICIENTS[:3], math.nan)
    monkeypatch.setattr(olmix, "exact_proposal", lambda *args, **kwargs: pytest.fail("the exact proposer ran"))
    with pytest.raises(ValueError, match="2046 of the 2048 blocks"):
        propose([law], ONE_TASK, np.full(3, 3.0), cap=1.0, kl=0.01)


def test_olmix_proposal_approaches_the_reference_mixture_under_large_kl():
    # At 1 epoch no cap binds near the proportional mixture.
    proposal = propose([GENERATING_LAW], ONE_TASK, EPOCHS_PER_UNIT_WEIGHT, cap=1.0, kl=1e3)
    assert np.abs(proposal.continuous - PROPORTIONAL).sum() / 2 < 1e-3
