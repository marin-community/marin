# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Variance-normalized hinge-loss objective for Bayesian optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from experiments.datakit.mixprior.data import Swarm, SwarmObservations

REFERENCE_GIST_URL = "https://gist.github.com/Helw150/9a563b9ab7b95438b8d4d689777f6f7f"
REFERENCE_GIST_REVISION = "557601ae46fce69549a06801fc882b29f1245d70"
OBJECTIVE_KIND = "variance_normalized_linear_targets_plus_capped_hinge"

LINEAR_TARGET_TASKS = (
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
    *LINEAR_TARGET_TASKS,
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


class HingeObjective(Protocol):
    labels: list[str]
    epsilon: float

    def loss_with_variance(
        self,
        labels: tuple[str, ...],
        values: np.ndarray,
        observation_sd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def payload(self) -> dict[str, Any]: ...


def _indices(labels: list[str], tasks: tuple[str, ...]) -> np.ndarray:
    missing = sorted(set(tasks) - set(labels))
    if missing:
        raise ValueError(f"Missing objective tasks: {missing}")
    return np.asarray([labels.index(task) for task in tasks], dtype=int)


def _constituents(labels: list[str], prefix: str) -> np.ndarray:
    result = [index for index, label in enumerate(labels) if label.startswith(prefix) and label != f"{prefix}mean"]
    return np.asarray(result, dtype=int)


def _linear_target_columns(labels: tuple[str, ...]) -> list[int]:
    return [index for index, task in enumerate(labels) if task in LINEAR_TARGET_TASKS]


def _loss_standardized(
    labels: tuple[str, ...], values: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    linear_target = values[..., _linear_target_columns(labels)].sum(axis=-1)
    hinge = np.maximum(values, -epsilon).sum(axis=-1)
    return linear_target, hinge, linear_target + hinge


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


def objective_observations(
    swarm: Swarm,
    objective: HingeObjective,
    objective_metrics: tuple[str, ...],
    observation_sd_by_label: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return maximization-objective observations and their variances."""
    missing = sorted(set(objective_metrics) - set(swarm.data.labels))
    if missing:
        raise ValueError(f"Swarm {swarm.swarm_id} is missing objective metrics: {missing}")
    swarm_indices = [swarm.data.labels.index(label) for label in objective_metrics]
    outcomes = swarm.data.outcomes[:, swarm_indices]
    observation_sd = np.broadcast_to(
        np.asarray(
            [observation_sd_by_label[label] for label in objective_metrics],
            dtype=np.float64,
        ),
        outcomes.shape,
    )
    loss, variance = objective.loss_with_variance(objective_metrics, outcomes, observation_sd)
    return -loss, variance


@dataclass(frozen=True)
class VarianceNormalizedObjective:
    labels: list[str]
    reference_mean: np.ndarray
    reference_std: np.ndarray
    task_correlation: np.ndarray
    reference_count: int
    epsilon: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError("Objective epsilon must be finite and non-negative")

    @classmethod
    def fit(
        cls,
        labels: list[str],
        values: np.ndarray,
        reference_mask: np.ndarray,
        epsilon: float,
    ) -> VarianceNormalizedObjective:
        values = cls._exact_aggregates(labels, values)
        reference = values[np.asarray(reference_mask, dtype=bool)]
        if len(reference) < 2:
            raise ValueError("At least two proportional references are required")
        reference_std = reference.std(axis=0, ddof=1)
        if np.any(reference_std <= 0):
            raise ValueError("Reference sample standard deviations must be positive")
        selected = _indices(labels, HINGE_TASKS)
        standardized = (reference[:, selected] - reference[:, selected].mean(axis=0)) / reference_std[selected]
        return cls(
            labels=list(labels),
            reference_mean=reference.mean(axis=0),
            reference_std=reference_std,
            task_correlation=np.corrcoef(standardized, rowvar=False),
            reference_count=len(reference),
            epsilon=float(epsilon),
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
    def linear_target_indices(self) -> np.ndarray:
        return _indices(self.labels, LINEAR_TARGET_TASKS)

    @property
    def hinge_indices(self) -> np.ndarray:
        return _indices(self.labels, HINGE_TASKS)

    def components(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = self._exact_aggregates(self.labels, values)
        z = (values - self.reference_mean) / self.reference_std
        selected = z[..., self.hinge_indices]
        return _loss_standardized(HINGE_TASKS, selected, self.epsilon)

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
        linear_target_indices = _linear_target_columns(labels)
        _, _, loss = _loss_standardized(labels, standardized, self.epsilon)

        coefficients = np.ones_like(values) / std
        coefficients[standardized <= -self.epsilon] = 0.0
        coefficients[:, linear_target_indices] += 1.0 / std[linear_target_indices]
        scaled_sd = coefficients * observation_sd
        correlation_indices = [HINGE_TASKS.index(task) for task in labels]
        correlation = self.task_correlation[np.ix_(correlation_indices, correlation_indices)]
        variance = np.einsum("ni,ij,nj->n", scaled_sd, correlation, scaled_sd)
        return loss, np.maximum(variance, np.finfo(np.float64).eps)

    def payload(self) -> dict[str, Any]:
        return {
            "kind": OBJECTIVE_KIND,
            "reference_gist_url": REFERENCE_GIST_URL,
            "reference_gist_revision": REFERENCE_GIST_REVISION,
            "labels": self.labels,
            "epsilon": self.epsilon,
            "linear_target_tasks": list(LINEAR_TARGET_TASKS),
            "hinge_tasks": list(HINGE_TASKS),
            "reference_count": self.reference_count,
            "reference_mean": self.reference_mean.tolist(),
            "reference_sample_std": self.reference_std.tolist(),
            "objective_task_correlation": self.task_correlation.tolist(),
            "aggregate_policy": "language means are recomputed from constituents",
        }
