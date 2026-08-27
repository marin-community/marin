# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Describe a proposed mixture relative to observed target mixtures."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from experiments.datakit.mixprior.data import MixtureComponentMetadata, Swarm
from experiments.datakit.mixprior.search import CandidatePosterior


def overall_content_features(
    weights: np.ndarray,
    component_content: np.ndarray,
    phase_token_fractions: np.ndarray,
) -> np.ndarray:
    """Map curricula to rooted token-weighted content distributions."""
    weights = np.asarray(weights, dtype=np.float64)
    component_content = np.asarray(component_content, dtype=np.float64)
    phase_token_fractions = np.asarray(phase_token_fractions, dtype=np.float64)
    if weights.ndim != 3 or component_content.ndim != 2:
        raise ValueError("Weights and component content must be rank-3 and rank-2")
    if weights.shape[2] != component_content.shape[0]:
        raise ValueError("Mixture weights and component content do not align")
    if phase_token_fractions.shape != (weights.shape[1],):
        raise ValueError("Phase token fractions must match the curriculum phases")
    if np.any(weights < 0) or np.any(component_content < 0):
        raise ValueError("Mixture weights and component content must be non-negative")
    if not np.allclose(weights.sum(axis=2), 1.0):
        raise ValueError("Every phase must be a simplex vector")
    if not np.allclose(component_content.sum(axis=1), 1.0):
        raise ValueError("Every component-content row must be a distribution")
    if np.any(phase_token_fractions <= 0) or not np.isclose(phase_token_fractions.sum(), 1.0):
        raise ValueError("Phase token fractions must be a positive simplex vector")
    phase_content = weights @ component_content
    return np.sqrt((phase_content * phase_token_fractions[None, :, None]).sum(axis=1))


def squared_hellinger(features: np.ndarray) -> np.ndarray:
    """Return pairwise squared Hellinger distances between rooted distributions."""
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError("Content features must be a matrix")
    squared_norm = np.sum(np.square(features), axis=1)
    squared_distance = squared_norm[:, None] + squared_norm[None, :] - 2 * features @ features.T
    return np.maximum(squared_distance, 0.0) / 2


def candidate_diagnostics(
    target: Swarm,
    weights: np.ndarray,
    posterior: CandidatePosterior,
) -> dict[str, object]:
    metadata = target.data.component_metadata
    proportional = target.data.available_tokens / target.data.available_tokens.sum()
    phase_token_fractions = target.phase_budgets / target.phase_budgets.sum()
    candidate_feature = overall_content_features(weights[None], target.content_matrix, phase_token_fractions)
    observed_features = overall_content_features(target.data.weights, target.content_matrix, phase_token_fractions)
    distance = squared_hellinger(np.concatenate([candidate_feature, observed_features]))[0, 1:]
    nearest = int(distance.argmin())
    nearest_name = target.data.run_names[nearest] or target.data.observation_ids[nearest]
    epochs = (weights * target.exposure_multipliers).sum(axis=0)
    phase = [_phase_diagnostics(values, proportional, metadata) for values in weights]
    adjacent_variation = [float(0.5 * np.abs(current - previous).sum()) for previous, current in pairwise(weights)]
    return {
        "posterior": posterior,
        "nearest_observation": {
            "observation": nearest_name,
            "overall_content_squared_hellinger": float(distance[nearest]),
        },
        "max_cumulative_epochs": float(epochs.max()),
        "adjacent_phase_total_variation": adjacent_variation,
        "phases": phase,
    }


def _phase_diagnostics(
    weights: np.ndarray,
    proportional: np.ndarray,
    metadata: list[MixtureComponentMetadata],
) -> dict[str, object]:
    quality_weights: dict[str, float] = {}
    domain_weights: dict[str, float] = {}
    proportional_domain_weights: dict[str, float] = {}
    for index, component in enumerate(metadata):
        quality = f"q{component['quality']}"
        quality_weights[quality] = quality_weights.get(quality, 0.0) + float(weights[index])
        domain = component["domain"]
        domain_weights[domain] = domain_weights.get(domain, 0.0) + float(weights[index])
        proportional_domain_weights[domain] = proportional_domain_weights.get(domain, 0.0) + float(proportional[index])
    top_domains = sorted(
        (
            {
                "domain": domain,
                "weight": domain_weights[domain],
                "proportional_weight": proportional_domain_weights[domain],
                "delta": domain_weights[domain] - proportional_domain_weights[domain],
            }
            for domain in domain_weights
        ),
        key=lambda row: row["delta"],
        reverse=True,
    )[:5]
    top_components = sorted(
        (
            {
                "component": component["cell"],
                "domain": component["domain"],
                "quality": component["quality"],
                "weight": float(weights[index]),
                "proportional_weight": float(proportional[index]),
                "delta": float(weights[index] - proportional[index]),
            }
            for index, component in enumerate(metadata)
        ),
        key=lambda row: row["delta"],
        reverse=True,
    )[:8]
    return {
        "quality_weights": quality_weights,
        "largest_domain_weight_increases": top_domains,
        "largest_component_weight_increases": top_components,
    }
