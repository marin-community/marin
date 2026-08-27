# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Define the scalar loss modeled by Bayesian optimization.

Bits-per-byte (BPB) metrics are standardized against proportional-reference
runs, so negative values are improvements and positive values are regressions.
Every modeled task contributes ``max(z, -epsilon)``: regressions are fully
penalized, while improvements beyond ``epsilon`` receive no further reward.
The explicitly listed target tasks also contribute an uncapped ``z`` term, so
the search rewards improvements on those benchmarks while every modeled task
guards against regression. Task membership is experiment policy. The GP models
the negative loss because acquisition functions maximize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np

from experiments.datakit.mixprior.data import Swarm, SwarmObservations

REFERENCE_GIST_URL = "https://gist.github.com/Helw150/9a563b9ab7b95438b8d4d689777f6f7f"
REFERENCE_GIST_REVISION = "557601ae46fce69549a06801fc882b29f1245d70"
OBJECTIVE_KIND = "variance_normalized_linear_targets_plus_capped_hinge"
PROPORTIONAL_REFERENCE_GROUP = "marin_proportional"

UNCAPPED_TARGET_TASKS = (
    "logprob_humaneval_10shot",
    "logprob_gsm8k_5shot",
    "arc_challenge_0shot",
    "openbookqa_0shot",
    "sciq_0shot",
    "mmlu_pro_5shot",
    "lb_bbh_3shot",
    "musr_0shot",
    "truthfulqa_mc1_0shot",
)

HINGE_TASKS = (
    *UNCAPPED_TARGET_TASKS,
    "boolq_0shot",
    "copa_0shot",
    "csqa_0shot",
    "gpqa_0shot",
    "hellaswag_0shot",
    "lambada_0shot",
    "medqa_0shot",
    "medmcqa_0shot",
    "piqa_0shot",
    "winogrande_0shot",
    "include_mean",
    "belebele_mean",
)


class ObjectiveObservations(NamedTuple):
    values: np.ndarray
    variances: np.ndarray


class ScalarObjective(Protocol):
    def observations(self, swarm: Swarm) -> ObjectiveObservations: ...


def _indices(labels: list[str], tasks: tuple[str, ...]) -> np.ndarray:
    missing = sorted(set(tasks) - set(labels))
    if missing:
        raise ValueError(f"Missing objective tasks: {missing}")
    return np.asarray([labels.index(task) for task in tasks], dtype=int)


def _constituents(labels: list[str], prefix: str) -> np.ndarray:
    result = [index for index, label in enumerate(labels) if label.startswith(prefix) and label != f"{prefix}mean"]
    return np.asarray(result, dtype=int)


def _uncapped_target_columns(labels: tuple[str, ...], uncapped_tasks: tuple[str, ...]) -> list[int]:
    return [index for index, task in enumerate(labels) if task in uncapped_tasks]


def _loss_standardized(
    labels: tuple[str, ...],
    values: np.ndarray,
    epsilon: float,
    uncapped_tasks: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uncapped_target = values[..., _uncapped_target_columns(labels, uncapped_tasks)].sum(axis=-1)
    hinge = np.maximum(values, -epsilon).sum(axis=-1)
    return uncapped_target, hinge, uncapped_target + hinge


def pooled_replicate_sd(data: SwarmObservations) -> np.ndarray:
    """Estimate observation-noise SD from repeated-seed evaluations."""
    flat = np.round(data.weights.reshape(len(data.weights), -1), decimals=12)
    _, groups = np.unique(flat, axis=0, return_inverse=True)
    squared_error = np.zeros(len(data.labels), dtype=np.float64)
    degrees_of_freedom = 0
    for group in range(groups.max() + 1):
        values = data.outcomes[groups == group]
        if len(values) < 2:
            continue
        squared_error += np.square(values - values.mean(axis=0)).sum(axis=0)
        degrees_of_freedom += len(values) - 1
    if degrees_of_freedom < 1:
        raise ValueError("Observation-noise estimation requires replicated designs")
    observation_sd = np.sqrt(squared_error / degrees_of_freedom)
    if np.any(observation_sd <= 0) or not np.isfinite(observation_sd).all():
        raise ValueError("Replicates do not identify positive noise for every metric")
    return observation_sd


@dataclass(frozen=True)
class VarianceNormalizedObjective:
    labels: list[str]
    reference_mean: np.ndarray
    reference_std: np.ndarray
    task_correlation: np.ndarray
    reference_count: int
    epsilon: float
    metrics: tuple[str, ...]
    hinge_tasks: tuple[str, ...]
    uncapped_tasks: tuple[str, ...]
    observation_sd: np.ndarray

    def __post_init__(self) -> None:
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("Objective epsilon must be finite and non-negative")
        if not self.metrics or len(self.metrics) != len(set(self.metrics)):
            raise ValueError("Objective metrics must be a non-empty unique tuple")
        if self.observation_sd.shape != (len(self.metrics),):
            raise ValueError("Observation standard deviations must match objective metrics")
        if np.any(self.observation_sd <= 0) or not np.isfinite(self.observation_sd).all():
            raise ValueError("Observation standard deviations must be finite and positive")
        missing = sorted(set(self.metrics) - set(self.labels))
        if missing:
            raise ValueError(f"Objective metrics are missing from the reference: {missing}")
        if not self.hinge_tasks or not set(self.uncapped_tasks).issubset(self.hinge_tasks):
            raise ValueError("Uncapped tasks must be a subset of the non-empty hinge task list")

    @classmethod
    def fit(
        cls,
        labels: list[str],
        values: np.ndarray,
        reference_mask: np.ndarray,
        epsilon: float,
        metrics: tuple[str, ...],
        hinge_tasks: tuple[str, ...],
        uncapped_tasks: tuple[str, ...],
        observation_sd: np.ndarray,
    ) -> VarianceNormalizedObjective:
        values = cls._exact_aggregates(labels, values)
        reference = values[np.asarray(reference_mask, dtype=bool)]
        if len(reference) < 2:
            raise ValueError("At least two proportional references are required")
        reference_std = reference.std(axis=0, ddof=1)
        if np.any(reference_std <= 0):
            raise ValueError("Reference sample standard deviations must be positive")
        selected = _indices(labels, hinge_tasks)
        standardized = (reference[:, selected] - reference[:, selected].mean(axis=0)) / reference_std[selected]
        return cls(
            labels=list(labels),
            reference_mean=reference.mean(axis=0),
            reference_std=reference_std,
            task_correlation=np.corrcoef(standardized, rowvar=False),
            reference_count=len(reference),
            epsilon=float(epsilon),
            metrics=metrics,
            hinge_tasks=hinge_tasks,
            uncapped_tasks=uncapped_tasks,
            observation_sd=np.asarray(observation_sd, dtype=np.float64),
        )

    @staticmethod
    def _exact_aggregates(labels: list[str], values: np.ndarray) -> np.ndarray:
        values = np.array(values, copy=True)
        for label, prefix in (
            ("include_mean", "include_"),
            ("belebele_mean", "belebele_"),
        ):
            indices = _constituents(labels, prefix)
            if len(indices):
                values[..., labels.index(label)] = values[..., indices].mean(axis=-1)
        return values

    @property
    def hinge_indices(self) -> np.ndarray:
        return _indices(self.labels, self.hinge_tasks)

    def components(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = self._exact_aggregates(self.labels, values)
        z = (values - self.reference_mean) / self.reference_std
        selected = z[..., self.hinge_indices]
        return _loss_standardized(self.hinge_tasks, selected, self.epsilon, self.uncapped_tasks)

    def observations(self, swarm: Swarm) -> ObjectiveObservations:
        """Evaluate one swarm as a higher-is-better scalar objective."""
        missing = sorted(set(self.metrics) - set(swarm.data.labels))
        if missing:
            raise ValueError(f"Swarm {swarm.swarm_id} is missing objective metrics: {missing}")
        swarm_indices = [swarm.data.labels.index(label) for label in self.metrics]
        outcomes = swarm.data.outcomes[:, swarm_indices]
        observation_sd = np.broadcast_to(self.observation_sd, outcomes.shape)
        loss, variance = self.loss_with_variance(self.metrics, outcomes, observation_sd)
        return ObjectiveObservations(-loss, variance)

    def loss_with_variance(
        self,
        labels: tuple[str, ...],
        values: np.ndarray,
        observation_sd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        reference_indices = _indices(self.labels, labels)
        mean = self.reference_mean[reference_indices]
        std = self.reference_std[reference_indices]
        standardized = (values - mean) / std
        uncapped_target_indices = _uncapped_target_columns(labels, self.uncapped_tasks)
        _, _, loss = _loss_standardized(labels, standardized, self.epsilon, self.uncapped_tasks)

        coefficients = np.ones_like(values) / std
        coefficients[standardized <= -self.epsilon] = 0.0
        coefficients[:, uncapped_target_indices] += 1.0 / std[uncapped_target_indices]
        scaled_sd = coefficients * observation_sd
        correlation_indices = [self.hinge_tasks.index(task) for task in labels]
        correlation = self.task_correlation[np.ix_(correlation_indices, correlation_indices)]
        variance = np.einsum("ni,ij,nj->n", scaled_sd, correlation, scaled_sd)
        return loss, np.maximum(variance, np.finfo(np.float64).eps)


def fit_harrier_hinge_objective(
    reference: SwarmObservations,
    noise_reference: SwarmObservations,
    metrics: tuple[str, ...],
    epsilon: float,
) -> VarianceNormalizedObjective:
    """Fit the published Harrier objective from its declared reference swarms."""
    noise_sd = pooled_replicate_sd(noise_reference)
    noise_by_label = dict(zip(noise_reference.labels, noise_sd, strict=True))
    missing = sorted(set(metrics) - set(noise_by_label))
    if missing:
        raise ValueError(f"Observation-noise estimates are missing metrics: {missing}")
    return VarianceNormalizedObjective.fit(
        reference.labels,
        reference.outcomes,
        np.asarray(reference.groups) == PROPORTIONAL_REFERENCE_GROUP,
        epsilon=epsilon,
        metrics=metrics,
        hinge_tasks=HINGE_TASKS,
        uncapped_tasks=UNCAPPED_TARGET_TASKS,
        observation_sd=np.asarray([noise_by_label[label] for label in metrics]),
    )


def objective_metadata(objective: VarianceNormalizedObjective) -> dict[str, object]:
    return {
        "kind": OBJECTIVE_KIND,
        "reference_gist_url": REFERENCE_GIST_URL,
        "reference_gist_revision": REFERENCE_GIST_REVISION,
        "labels": objective.labels,
        "epsilon": objective.epsilon,
        "objective_metrics": list(objective.metrics),
        "uncapped_target_tasks": list(objective.uncapped_tasks),
        "hinge_tasks": list(objective.hinge_tasks),
        "reference_count": objective.reference_count,
        "reference_mean": objective.reference_mean.tolist(),
        "reference_sample_std": objective.reference_std.tolist(),
        "observation_sd": objective.observation_sd.tolist(),
        "objective_task_correlation": objective.task_correlation.tolist(),
        "aggregate_policy": "language means are recomputed from constituents",
    }
