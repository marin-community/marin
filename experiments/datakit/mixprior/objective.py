# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score BPB improvements against replicated proportional runs."""

from dataclasses import dataclass

import numpy as np

from experiments.datakit.mixprior.hf import Data

PROPORTIONAL_REFERENCE_GROUP = "proportional_baseline"
STANDARDIZED_METRIC_CAP = 10.0

DROPLESS_PALOMA_PREFIX = "eval_dropless/paloma"
DROPLESS_UNCHEATABLE_PREFIX = "eval_dropless/uncheatable_eval"

TARGET_TASKS = (
    "logprob_humaneval_10shot",
    f"{DROPLESS_PALOMA_PREFIX}/dolma_100_programing_languages-llama3/bpb",
    f"{DROPLESS_UNCHEATABLE_PREFIX}/github_python-llama3/bpb",
    f"{DROPLESS_UNCHEATABLE_PREFIX}/github_cpp-llama3/bpb",
    f"{DROPLESS_UNCHEATABLE_PREFIX}/arxiv_computer_science-llama3/bpb",
    f"{DROPLESS_UNCHEATABLE_PREFIX}/arxiv_physics-llama3/bpb",
)

ZERO_SHOT_GUARDRAIL_TASKS = (
    "arc_challenge_0shot",
    "openbookqa_0shot",
    "sciq_0shot",
    "musr_0shot",
    "truthfulqa_mc1_0shot",
    "boolq_0shot",
    "copa_0shot",
    "hellaswag_0shot",
    "lambada_0shot",
    "piqa_0shot",
    "winogrande_0shot",
)

PALOMA_GUARDRAIL_TASKS = tuple(
    f"{DROPLESS_PALOMA_PREFIX}/{dataset}-llama3/bpb"
    for dataset in (
        "c4_100_domains",
        "c4_en",
        "dolma-v1_5",
        "dolma_100_subreddits",
        "falcon-refinedweb",
        "m2d2_s2orc_unsplit",
        "m2d2_wikipedia_unsplit",
        "mc4",
        "redpajama",
    )
)

UNCHEATABLE_GUARDRAIL_TASKS = tuple(
    f"{DROPLESS_UNCHEATABLE_PREFIX}/{dataset}-llama3/bpb" for dataset in ("ao3_english", "bbc_news", "wikipedia_english")
)

HINGE_TASKS = (
    *TARGET_TASKS,
    *ZERO_SHOT_GUARDRAIL_TASKS,
    *PALOMA_GUARDRAIL_TASKS,
    *UNCHEATABLE_GUARDRAIL_TASKS,
    "include_mean",
    "belebele_mean",
)


@dataclass
class Objective:
    columns: np.ndarray
    target_mask: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    noise_covariance: np.ndarray
    epsilon: float

    def __call__(self, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Score lower-is-better metric losses; return higher-is-better scores and noise variances."""
        standardized = (outcomes[:, self.columns] - self.mean) / self.scale
        within_cap = np.abs(standardized) < STANDARDIZED_METRIC_CAP
        standardized = np.clip(standardized, -STANDARDIZED_METRIC_CAP, STANDARDIZED_METRIC_CAP)

        # Average targets and guardrails separately: each group gets equal weight
        # regardless of how many evaluation metrics it contains.
        loss = np.zeros(len(standardized))
        loss_gradient = np.zeros_like(standardized)
        for mask in (self.target_mask, ~self.target_mask):
            metric_count = mask.sum()
            if metric_count == 0:
                continue
            group_values = standardized[:, mask]
            hinge_loss = np.maximum(group_values, -self.epsilon)
            loss += hinge_loss.mean(axis=1)
            loss_gradient[:, mask] = (group_values > -self.epsilon) / metric_count

        # Targets also reward improvements beyond the hinge's flat region.
        target_count = self.target_mask.sum()
        if target_count > 0:
            loss += standardized[:, self.target_mask].mean(axis=1)
            loss_gradient[:, self.target_mask] += 1 / target_count

        # Propagate metric noise through the score: variance = gradient.T @ noise @ gradient.
        # Clipped metrics have zero derivative. Convert back to raw metric units.
        loss_gradient *= within_cap / self.scale
        variance = np.einsum("ni,ij,nj->n", loss_gradient, self.noise_covariance, loss_gradient)
        variance = np.maximum(variance, np.finfo(float).eps)
        return -loss, variance


def fit_objective(
    data: Data,
    metrics: tuple[str, ...] = HINGE_TASKS,
    targets: tuple[str, ...] = TARGET_TASKS,
    epsilon: float = 0.0,
) -> Objective:
    """Estimate the fixed objective scale and noise from replicated designs."""
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError("The hinge tolerance must be finite and nonnegative")
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("Objective metrics must be nonempty and unique")
    columns = np.asarray([data.labels.index(label) for label in metrics])
    values = data.outcomes[:, columns]
    reference_rows = np.asarray(data.groups) == PROPORTIONAL_REFERENCE_GROUP
    reference = values[reference_rows]
    if len(reference) < 2 or not np.isfinite(values).all():
        raise ValueError("The objective needs finite outcomes and at least two proportional references")
    # Repeated runs of the same mixture estimate observation noise.
    flat_weights = data.weights.reshape(len(values), -1)
    rounded_weights = np.round(flat_weights, decimals=12)
    _, replicate_groups = np.unique(rounded_weights, axis=0, return_inverse=True)
    squared_error = np.zeros(len(metrics))
    degrees_of_freedom = 0
    for group in np.unique(replicate_groups):
        repeats = values[replicate_groups == group]
        deviations = repeats - repeats.mean(axis=0)
        squared_error += np.square(deviations).sum(axis=0)
        degrees_of_freedom += len(repeats) - 1
    if degrees_of_freedom == 0:
        raise ValueError("Noise estimation requires replicated designs")
    noise_sd = np.sqrt(squared_error / degrees_of_freedom)
    reference_sd = reference.std(axis=0, ddof=1)
    if np.any(noise_sd <= 0) or np.any(reference_sd <= 0):
        raise ValueError("Replicates must identify positive noise for each objective metric")
    # References supply cross-metric correlation; replicates supply noise magnitude.
    correlation = np.atleast_2d(np.corrcoef(reference, rowvar=False))
    noise_covariance = correlation * noise_sd[:, None] * noise_sd[None, :]
    return Objective(
        columns=columns,
        target_mask=np.asarray([label in targets for label in metrics]),
        mean=reference.mean(axis=0),
        scale=np.maximum(reference_sd, noise_sd),
        noise_covariance=noise_covariance,
        epsilon=epsilon,
    )
