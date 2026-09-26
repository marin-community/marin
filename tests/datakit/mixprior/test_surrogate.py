# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.datakit.mixprior.surrogate import SwarmTrainingRows, assemble_training_data


def test_training_data_standardizes_each_swarm() -> None:
    target = SwarmTrainingRows(
        swarm_id="target",
        features=np.asarray([[0.1, 0.2], [0.3, 0.4]]),
        objective_values=np.asarray([1.0, 3.0]),
        objective_variances=np.asarray([0.1, 0.2]),
    )
    source = SwarmTrainingRows(
        swarm_id="source",
        features=np.asarray([[0.5, 0.6], [0.7, 0.8]]),
        objective_values=np.asarray([10.0, 14.0]),
        objective_variances=np.asarray([0.4, 0.8]),
    )

    data = assemble_training_data([target, source])

    assert np.array_equal(
        data.features,
        np.asarray(
            [
                [0.1, 0.2, 0.0],
                [0.3, 0.4, 0.0],
                [0.5, 0.6, 1.0],
                [0.7, 0.8, 1.0],
            ]
        ),
    )
    assert np.array_equal(data.standardized_objective_values, [-1.0, 1.0, -1.0, 1.0])
    assert np.allclose(data.standardized_objective_variances, [0.1, 0.2, 0.1, 0.2])
    assert data.outcome_scales["target"] == (2.0, 1.0)
    assert data.outcome_scales["source"] == (12.0, 2.0)
