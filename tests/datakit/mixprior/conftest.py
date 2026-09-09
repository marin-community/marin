# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.datakit.mixprior.hf import Data


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    weights = rng.dirichlet([2, 3, 1], size=(20, 2))
    return Data(
        name="test",
        components=["a", "b", "c"],
        domains=["web", "web", "code"],
        available_tokens=np.array([10.0, 20.0, 30.0]),
        phase_budgets=np.array([40.0, 20.0]),
        weights=weights,
        labels=["reward", "guard"],
        outcomes=rng.normal(3.0, 0.1, size=(20, 2)),
        groups=["proportional_baseline"] * 2 + ["search"] * 18,
        observation_ids=[str(i) for i in range(20)],
    )
