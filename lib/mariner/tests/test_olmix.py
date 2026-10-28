# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest
from mariner.model import Objective
from mariner.olmix import LogLinearLaw, fit_log_linear, propose, quantize_under_cap
from mariner.optimize import MIXTURE_BLOCK_SIZE

# The generating law y = 0.7 + exp(w @ (-1, 0.5, -0.3, 0.2)) over four buckets. Its Huber loss describes a fit, and a
# generating law has none.
FLOOR = 0.7
COEFFICIENTS = np.array([-1.0, 0.5, -0.3, 0.2])
GENERATING_LAW = LogLinearLaw(math.log(FLOOR), COEFFICIENTS, math.nan)
ONE_TASK = Objective("synthetic", ("t0",), np.ones(1))
EPOCHS_PER_UNIT_WEIGHT = np.array([0.5, 1.0, 4.0, 8.0])


def test_quantize_under_cap_fills_block_without_exceeding_cap():
    counts = quantize_under_cap(np.array([0.5, 0.3, 0.2]), np.array([4.0, 1.0, 1.0]), cap=1.0)
    assert counts.tolist() == [512, 1127, 409]  # 512 is the cap; the 513 left over go to the next-largest weight


def test_fit_log_linear_recovers_generating_law():
    rng = np.random.default_rng(3)
    weights = rng.dirichlet(np.ones(4), size=40)
    targets = FLOOR + np.exp(weights @ COEFFICIENTS) + rng.normal(0.0, 1e-4, size=40)
    law = fit_log_linear(weights, targets)
    fresh = rng.dirichlet(np.ones(4), size=20)
    assert np.abs(law.predict(fresh) - (FLOOR + np.exp(fresh @ COEFFICIENTS))).max() < 1e-3


def test_olmix_proposal_respects_caps():
    # The law alone would put the whole block on bucket 0, which 0.4 epochs caps at 0.8 of the mixture; buckets 2
    # and 3 end at their caps too, and rounding each bucket to its nearest block would put bucket 2 over its cap.
    cap = 0.4
    proposal = propose([GENERATING_LAW], ONE_TASK, EPOCHS_PER_UNIT_WEIGHT, cap=cap, kl=0.01)
    assert np.max(proposal.continuous * EPOCHS_PER_UNIT_WEIGHT) == pytest.approx(cap)  # the cap binds
    assert np.issubdtype(proposal.counts.dtype, np.integer) and proposal.counts.sum() == MIXTURE_BLOCK_SIZE
    assert np.max(proposal.counts / MIXTURE_BLOCK_SIZE * EPOCHS_PER_UNIT_WEIGHT) <= cap


def test_olmix_proposal_approaches_the_reference_mixture_under_large_kl():
    proposal = propose([GENERATING_LAW], ONE_TASK, EPOCHS_PER_UNIT_WEIGHT, cap=1.0, kl=1e3)
    # The proportional mixture, pools in proportion to 1 / epochs, materializes 8/27 of an epoch everywhere: no cap
    # binds.
    reference = np.array([16.0, 8.0, 2.0, 1.0]) / 27
    assert np.abs(proposal.continuous - reference).sum() / 2 < 1e-3
