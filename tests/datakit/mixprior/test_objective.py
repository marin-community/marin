# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.datakit.mixprior.data import SwarmObservations
from experiments.datakit.mixprior.objective import (
    HINGE_TASKS,
    VarianceNormalizedObjective,
    objective_observations,
    pooled_replicate_sd,
)


def test_objective_applies_capped_hinge_and_uncapped_linear_term(
    objective: VarianceNormalizedObjective,
) -> None:
    values = objective.reference_mean.copy()
    boolq = objective.labels.index("boolq_0shot")
    values[boolq] -= 2 * objective.reference_std[boolq]
    assert objective.components(values)[2] == pytest.approx(0.0)

    values = objective.reference_mean.copy()
    humaneval = objective.labels.index("logprob_humaneval_10shot")
    values[humaneval] -= 2 * objective.reference_std[humaneval]
    assert objective.components(values)[2] == pytest.approx(-2.0)
    values[humaneval] += 4 * objective.reference_std[humaneval]
    assert objective.components(values)[2] == pytest.approx(4.0)

    tolerant = replace(objective, epsilon=1.0)
    values = objective.reference_mean.copy()
    values[boolq] -= 2 * objective.reference_std[boolq]
    assert tolerant.components(values)[2] == pytest.approx(-1.0)


def test_objective_observations_uses_only_selected_metrics() -> None:
    objective = _objective(list(HINGE_TASKS))
    values = np.zeros((1, len(objective.labels)))
    swarm = SimpleNamespace(
        swarm_id="test",
        data=SimpleNamespace(labels=objective.labels, outcomes=values),
    )
    objective_metrics = tuple(task for task in HINGE_TASKS if task != "include_mean")
    observation_sd = {label: 0.1 for label in objective.labels}
    baseline, baseline_variance = objective_observations(swarm, objective, objective_metrics, observation_sd)

    values[:, objective.labels.index("include_mean")] = 10.0
    without_include, without_include_variance = objective_observations(
        swarm, objective, objective_metrics, observation_sd
    )
    assert np.array_equal(without_include, baseline)
    assert np.array_equal(without_include_variance, baseline_variance)

    values[:, objective.labels.index("boolq_0shot")] = 10.0
    with_regression, _ = objective_observations(swarm, objective, objective_metrics, observation_sd)
    assert with_regression < without_include


def test_repeated_seed_evaluations_estimate_observation_noise(
    swarm_observations: SwarmObservations,
) -> None:
    observation_sd = pooled_replicate_sd(swarm_observations)
    assert np.allclose(observation_sd, np.sqrt(0.02))


@pytest.mark.parametrize("epsilon", [-1.0, np.nan, np.inf])
def test_objective_rejects_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises(ValueError):
        VarianceNormalizedObjective(
            labels=["boolq_0shot"],
            reference_mean=np.zeros(1),
            reference_std=np.ones(1),
            task_correlation=np.eye(1),
            reference_count=2,
            epsilon=epsilon,
        )


def _objective(labels: list[str]) -> VarianceNormalizedObjective:
    return VarianceNormalizedObjective(
        labels=labels,
        reference_mean=np.zeros(len(labels)),
        reference_std=np.ones(len(labels)),
        task_correlation=np.eye(len(labels)),
        reference_count=2,
        epsilon=0.0,
    )
