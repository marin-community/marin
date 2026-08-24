# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build, evaluate, and interpret candidate mixture sets."""

from __future__ import annotations

import logging
from itertools import pairwise
from pathlib import Path
from statistics import NormalDist
from typing import NamedTuple, TypedDict

import numpy as np
import torch

from experiments.datakit.mixprior.artifacts import ModelMetadata, write_candidate_bundle
from experiments.datakit.mixprior.campaign import Campaign, load_campaign
from experiments.datakit.mixprior.data import (
    AcquiredCandidate,
    MixtureComponentMetadata,
    PoolProvenance,
    Swarm,
)
from experiments.datakit.mixprior.model import (
    OBJECTIVE_NAME,
    AxTransferFit,
    TransferPredictor,
    curriculum_features,
    fit_additive_hellinger_model,
    prepare_hellinger_transfer_data,
    squared_hellinger,
)
from experiments.datakit.mixprior.objective import objective_observations

DEFAULT_POOL_SIZE = 65_536
DEFAULT_SEED = 20260821
POSTERIOR_MEAN_NAME = "PosteriorMean"
POSTERIOR_MEAN_SELECTION_RULE = "argmax over the finite candidate set"
PROPORTIONAL_FLOOR = 0.02
PERTURBATION_SCALE_RANGE = (0.02, 2.0)
POOL_DRAW_SIZE = 16_384
ACQUISITION_CHUNK_SIZE = 128
PROGRESS_ROWS = 4_096
logger = logging.getLogger(__name__)


class CandidatePosterior(TypedDict):
    objective_mean: float
    objective_sd: float
    incumbent_objective_value: float
    probability_of_improvement: float


class LognormalPoolInputs(NamedTuple):
    center_designs: np.ndarray
    availability_proportional_design: np.ndarray
    observed_weights: np.ndarray
    exposure_multipliers: np.ndarray
    max_cumulative_epochs: float


class PreparedCandidateFeatures(NamedTuple):
    weights: np.ndarray
    features: np.ndarray
    phase_token_fractions: np.ndarray


class CandidateSelection(NamedTuple):
    acquired: AcquiredCandidate
    weights: np.ndarray
    posterior: CandidatePosterior
    acquisition_function: str
    selection_rule: str


def campaign_lognormal_pool_inputs(campaign: Campaign) -> LognormalPoolInputs:
    target = campaign.target
    availability_proportional = target.data.available_tokens / target.data.available_tokens.sum()
    availability_proportional = np.broadcast_to(
        availability_proportional,
        (len(target.phase_budgets), len(availability_proportional)),
    )
    center_designs = [availability_proportional, *target.data.weights]
    center_designs.extend(
        weights
        for source in campaign.sources
        if source.data.mixture_components == target.data.mixture_components
        for weights in source.data.weights
    )
    return LognormalPoolInputs(
        center_designs=np.asarray(center_designs),
        availability_proportional_design=availability_proportional,
        observed_weights=target.data.weights,
        exposure_multipliers=target.exposure_multipliers,
        max_cumulative_epochs=campaign.max_cumulative_epochs,
    )


def sample_lognormal_pool(space: LognormalPoolInputs, size: int, seed: int) -> np.ndarray:
    if size < 1:
        raise ValueError("Candidate pool size must be positive")
    center_designs = unique_mixtures(space.center_designs)
    center_designs = center_designs[
        epoch_feasible_mask(center_designs, space.exposure_multipliers, space.max_cumulative_epochs)
    ]
    rng = np.random.default_rng(seed)
    pool = exclude_observed(center_designs, space.observed_weights)
    while len(pool) < size:
        selected = center_designs[rng.integers(len(center_designs), size=POOL_DRAW_SIZE)]
        scale = np.exp(
            rng.uniform(
                np.log(PERTURBATION_SCALE_RANGE[0]),
                np.log(PERTURBATION_SCALE_RANGE[1]),
                size=(
                    len(selected),
                    space.availability_proportional_design.shape[0],
                    1,
                ),
            )
        )
        base = (1.0 - PROPORTIONAL_FLOOR) * selected
        base += PROPORTIONAL_FLOOR * space.availability_proportional_design
        weights = base * np.exp(rng.normal(size=base.shape) * scale)
        weights /= weights.sum(axis=-1, keepdims=True)
        feasible = weights[
            epoch_feasible_mask(
                weights,
                space.exposure_multipliers,
                space.max_cumulative_epochs,
            )
        ]
        pool = unique_mixtures(np.concatenate([pool, feasible]))
        pool = exclude_observed(pool, space.observed_weights)
        logger.info("Sampled %s/%s feasible mixtures", f"{min(len(pool), size):,}", f"{size:,}")
    return pool[:size]


def exclude_observed(pool: np.ndarray, observed_weights: np.ndarray) -> np.ndarray:
    observed = {row.tobytes() for row in np.round(observed_weights.reshape(len(observed_weights), -1), decimals=12)}
    keep = [row.tobytes() not in observed for row in np.round(pool.reshape(len(pool), -1), decimals=12)]
    return pool[np.asarray(keep)]


def epoch_feasible_mask(
    weights: np.ndarray,
    exposure_multipliers: np.ndarray,
    max_cumulative_epochs: float,
) -> np.ndarray:
    epochs = (weights * exposure_multipliers[None]).sum(axis=1)
    return np.max(epochs, axis=1) <= max_cumulative_epochs


def unique_mixtures(weights: np.ndarray) -> np.ndarray:
    flat = np.round(weights.reshape(len(weights), -1), decimals=12)
    _, indices = np.unique(flat, axis=0, return_index=True)
    return weights[np.sort(indices)]


def acquire_posterior_mean(
    model: TransferPredictor,
    candidate_features: np.ndarray,
    swarm_index: int,
) -> AcquiredCandidate:
    """Maximize the PosteriorMean acquisition function over a candidate set."""
    values = []
    for start in range(0, len(candidate_features), ACQUISITION_CHUNK_SIZE):
        stop = min(start + ACQUISITION_CHUNK_SIZE, len(candidate_features))
        values.append(model.predict(candidate_features[start:stop], swarm_index).mean)
        if stop == len(candidate_features) or stop % PROGRESS_ROWS == 0:
            logger.info("Scored %s/%s pool rows", f"{stop:,}", f"{len(candidate_features):,}")
    acquisition_values = np.concatenate(values)
    pool_index = int(np.argmax(acquisition_values))
    return AcquiredCandidate(
        pool_index=pool_index,
        acquisition_value=float(acquisition_values[pool_index]),
        acquisition_values=acquisition_values,
    )


def prepare_candidate_features(target: Swarm, weights: np.ndarray) -> PreparedCandidateFeatures:
    """Project a fixed mixture pool into the target swarm's model space."""
    phase_token_fractions = target.phase_budgets / target.phase_budgets.sum()
    return PreparedCandidateFeatures(
        weights=weights,
        features=curriculum_features(weights, target.content_matrix, phase_token_fractions),
        phase_token_fractions=phase_token_fractions,
    )


def select_posterior_mean(
    campaign: Campaign,
    model: TransferPredictor,
    candidates: PreparedCandidateFeatures,
) -> CandidateSelection:
    """Select a candidate by maximizing the posterior-mean objective."""
    acquired = acquire_posterior_mean(model, candidates.features, model.target_swarm_index)
    return build_candidate_selection(
        campaign,
        model,
        candidates,
        acquired,
        acquisition_function=POSTERIOR_MEAN_NAME,
        selection_rule=POSTERIOR_MEAN_SELECTION_RULE,
    )


def build_candidate_selection(
    campaign: Campaign,
    model: TransferPredictor,
    candidates: PreparedCandidateFeatures,
    acquired: AcquiredCandidate,
    *,
    acquisition_function: str,
    selection_rule: str,
) -> CandidateSelection:
    """Interpret one acquisition result without assuming how it was selected."""
    weights = candidates.weights[acquired.pool_index]
    moments = model.predict(
        candidates.features[acquired.pool_index : acquired.pool_index + 1],
        model.target_swarm_index,
    )
    objective_mean = float(moments.mean[0])
    objective_sd = float(np.sqrt(max(moments.variance[0], 0.0)))
    observed_objective, _ = objective_observations(
        campaign.target,
        campaign.objective,
        campaign.objective_metrics,
        campaign.observation_sd,
    )
    incumbent = float(observed_objective.max())
    probability = (
        float(objective_mean > incumbent)
        if objective_sd == 0.0
        else 1.0 - NormalDist(mu=objective_mean, sigma=objective_sd).cdf(incumbent)
    )
    posterior: CandidatePosterior = {
        "objective_mean": objective_mean,
        "objective_sd": objective_sd,
        "incumbent_objective_value": incumbent,
        "probability_of_improvement": probability,
    }
    return CandidateSelection(
        acquired=acquired,
        weights=weights,
        posterior=posterior,
        acquisition_function=acquisition_function,
        selection_rule=selection_rule,
    )


def generate_candidate(
    *,
    campaign_manifest: Path,
    output_dir: Path,
    pool_size: int,
    seed: int,
    device: torch.device,
    dependency_lock: Path,
) -> dict:
    """Run the default fit, pool, acquisition, interpretation, and write stages."""
    torch.manual_seed(seed)
    campaign = load_campaign(campaign_manifest)
    model = fit_additive_hellinger_model(prepare_hellinger_transfer_data(campaign), device)
    pool = sample_lognormal_pool(campaign_lognormal_pool_inputs(campaign), size=pool_size, seed=seed)
    candidates = prepare_candidate_features(campaign.target, pool)
    selection = select_posterior_mean(campaign, model, candidates)
    diagnostics = candidate_diagnostics(
        campaign.target,
        selection.weights,
        selection.posterior,
        objective_name=OBJECTIVE_NAME,
        hinge_tolerance=campaign.objective.epsilon,
        acquisition_function=selection.acquisition_function,
        selection_rule=selection.selection_rule,
    )
    proposal: PoolProvenance = {
        "kind": "lognormal_perturbation",
        "parameters": {
            "center_designs": (
                "availability-proportional target design and observations with " "matching mixture components"
            ),
            "availability_proportional_floor": PROPORTIONAL_FLOOR,
            "lognormal_scale_range": list(PERTURBATION_SCALE_RANGE),
        },
    }
    return write_candidate_bundle(
        campaign_manifest=campaign_manifest,
        campaign=campaign,
        model_payload=model.model_state(),
        model_metadata=default_model_metadata(campaign, model),
        pool=pool,
        acquired=selection.acquired,
        selected_weights=selection.weights,
        diagnostics=diagnostics,
        phase_token_fractions=candidates.phase_token_fractions,
        output_dir=output_dir,
        seed=seed,
        proposal=proposal,
        acquisition_function=selection.acquisition_function,
        selection_rule=selection.selection_rule,
        dependency_lock=dependency_lock,
    )


def candidate_diagnostics(
    target: Swarm,
    weights: np.ndarray,
    posterior: CandidatePosterior,
    *,
    objective_name: str,
    hinge_tolerance: float,
    acquisition_function: str,
    selection_rule: str,
) -> dict:
    metadata = target.data.component_metadata
    proportional = target.data.available_tokens / target.data.available_tokens.sum()
    phase_token_fractions = target.phase_budgets / target.phase_budgets.sum()
    candidate_feature = curriculum_features(weights[None], target.content_matrix, phase_token_fractions)
    observed_features = curriculum_features(target.data.weights, target.content_matrix, phase_token_fractions)
    distance = squared_hellinger(np.concatenate([candidate_feature, observed_features]))[0, 1:]
    nearest = int(distance.argmin())
    nearest_row = target.data.frame.iloc[nearest]
    nearest_name = nearest_row.run_name or nearest_row.observation_id
    epochs = (weights * target.exposure_multipliers).sum(axis=0)
    phase = [_phase_diagnostics(values, proportional, metadata) for values in weights]
    adjacent_variation = [float(0.5 * np.abs(current - previous).sum()) for previous, current in pairwise(weights)]
    return {
        "posterior": posterior,
        "nearest_observation": {
            "observation": nearest_name,
            "phase_weighted_squared_hellinger": float(distance[nearest]),
        },
        "max_cumulative_epochs": float(epochs.max()),
        "adjacent_phase_total_variation": adjacent_variation,
        "phases": phase,
        "summary": {
            "objective_name": objective_name,
            "objective_direction": "maximize",
            "hinge_tolerance": hinge_tolerance,
            "acquisition_function": acquisition_function,
            "selection_rule": selection_rule,
            "nearest_observation": nearest_name,
        },
    }


def _phase_diagnostics(
    weights: np.ndarray,
    proportional: np.ndarray,
    metadata: list[MixtureComponentMetadata],
) -> dict:
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
    top_cells = sorted(
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
        "largest_component_weight_increases": top_cells,
    }


def default_model_metadata(campaign: Campaign, model: AxTransferFit) -> ModelMetadata:
    """Describe the default fitted model for artifact persistence."""
    return {
        "kind": "shared_hellinger_plus_categorical_swarm_residual_gp",
        "device": str(model.device),
        "details": {
            "objective": OBJECTIVE_NAME,
            "kernel_reference_swarm": campaign.kernel_reference_swarm,
            "kernel_lengthscale": model.lengthscale,
            "fixed_context_role": "swarm provenance; not a GP input",
            "observation_counts": model.observation_counts,
            "fit_seconds": model.elapsed,
        },
    }
