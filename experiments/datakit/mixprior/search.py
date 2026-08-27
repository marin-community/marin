# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample lattice mixtures and select one from a fitted surrogate."""

from __future__ import annotations

import logging
from collections.abc import Callable
from statistics import NormalDist
from typing import NamedTuple, TypedDict

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

from experiments.datakit.mixprior.campaign import Campaign
from experiments.datakit.mixprior.data import Swarm
from experiments.datakit.mixprior.surrogate import (
    BotorchMixturePredictor,
    MixturePredictor,
)

DEFAULT_POOL_SIZE = 65_536
DEFAULT_POOL_SEEDS = (111, 222, 333)
DEFAULT_ACQUISITION_SEED = 7
POSTERIOR_MEAN_NAME = "PosteriorMean"
POSTERIOR_MEAN_SELECTION_RULE = "argmax over the finite candidate set"
LOG_NEI_NAME = "qLogNoisyExpectedImprovement"
LOG_NEI_SAMPLES = 1_024
LOG_NEI_SELECTION_RULE = f"argmax log noisy expected improvement with {LOG_NEI_SAMPLES:,} Sobol samples"
PROPORTIONAL_FLOOR = 0.02
PERTURBATION_SCALE_RANGE = (0.02, 2.0)
GLOBAL_PROPOSAL_FRACTION = 0.5
GLOBAL_CONCENTRATION_RANGE = (0.02, 20.0)
POOL_DRAW_SIZE = 16_384
MAX_POOL_DRAWS = 256
ACQUISITION_CHUNK_SIZE = 128
PROGRESS_ROWS = 4_096
MIXTURE_DENOMINATOR = 49_152
LATENT_FUNCTION_UNCERTAINTY = "latent_function"
logger = logging.getLogger(__name__)


class CandidatePosterior(TypedDict):
    objective_mean: float
    objective_sd: float
    uncertainty_kind: str
    incumbent_objective_value: float
    probability_of_improvement: float


class AcquiredCandidate(NamedTuple):
    pool_index: int
    acquisition_value: float
    acquisition_values: np.ndarray


AcquisitionScorer = Callable[
    [BotorchMixturePredictor, Swarm, np.ndarray, int | None, np.ndarray | None],
    AcquiredCandidate,
]


class Acquisition(NamedTuple):
    name: str
    selection_rule: str
    seed: int | None
    score: AcquisitionScorer


class PoolProvenance(TypedDict):
    kind: str
    parameters: dict[str, object]


class CandidatePoolInputs(NamedTuple):
    center_designs: np.ndarray
    availability_proportional_design: np.ndarray
    observed_weights: np.ndarray


class CandidatePool(NamedTuple):
    weights: np.ndarray
    seeds: tuple[int, ...]
    provenance: PoolProvenance


class CandidateSelection(NamedTuple):
    acquired: AcquiredCandidate
    weights: np.ndarray
    posterior: CandidatePosterior
    acquisition_function: str
    selection_rule: str
    acquisition_seed: int | None


def campaign_pool_inputs(campaign: Campaign) -> CandidatePoolInputs:
    """Center proposals on proportional and observed compatible mixtures."""
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
    return CandidatePoolInputs(
        center_designs=np.asarray(center_designs),
        availability_proportional_design=availability_proportional,
        observed_weights=target.data.weights,
    )


def sample_candidate_pool(space: CandidatePoolInputs, size: int, seed: int) -> np.ndarray:
    """Sample global simplex designs and perturbations around known designs.

    Half of each draw comes from symmetric Dirichlet distributions spanning
    sparse to dense simplexes. The other half uses local-to-broad lognormal
    perturbations. The 2% proportional floor lets zero-weight components enter
    the local proposals. Candidate validation does not enforce either proposal
    distribution.
    """
    if size < 1:
        raise ValueError("Candidate pool size must be positive")
    center_designs = unique_mixtures(quantize_mixtures(space.center_designs))
    observed_weights = quantize_mixtures(space.observed_weights)
    rng = np.random.default_rng(seed)
    pool = exclude_observed(center_designs, observed_weights)
    for _ in range(MAX_POOL_DRAWS):
        if len(pool) >= size:
            return pool[:size]
        global_count = round(POOL_DRAW_SIZE * GLOBAL_PROPOSAL_FRACTION)
        local_count = POOL_DRAW_SIZE - global_count
        selected = center_designs[rng.integers(len(center_designs), size=local_count)]
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
        local = base * np.exp(rng.normal(size=base.shape) * scale)
        local /= local.sum(axis=-1, keepdims=True)
        concentration = np.exp(
            rng.uniform(
                np.log(GLOBAL_CONCENTRATION_RANGE[0]),
                np.log(GLOBAL_CONCENTRATION_RANGE[1]),
                size=(global_count, selected.shape[1], 1),
            )
        )
        global_weights = rng.gamma(
            shape=np.broadcast_to(concentration, (global_count, selected.shape[1], selected.shape[2]))
        )
        global_weights /= global_weights.sum(axis=-1, keepdims=True)
        weights = np.concatenate([local, global_weights])
        rng.shuffle(weights, axis=0)
        weights = quantize_mixtures(weights)
        pool = unique_mixtures(np.concatenate([pool, weights]))
        pool = exclude_observed(pool, observed_weights)
        logger.info("Sampled %s/%s mixtures", f"{min(len(pool), size):,}", f"{size:,}")
    raise ValueError(
        f"Could only sample {len(pool):,} of {size:,} mixtures after {MAX_POOL_DRAWS * POOL_DRAW_SIZE:,} draws"
    )


def sample_candidate_pool_union(
    space: CandidatePoolInputs,
    size_per_seed: int,
    seeds: tuple[int, ...],
) -> np.ndarray:
    """Combine independently sampled pools into one deduplicated candidate set."""
    if not seeds:
        raise ValueError("At least one pool seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Pool seeds must be distinct")
    pools = [sample_candidate_pool(space, size_per_seed, seed) for seed in seeds]
    pool = unique_mixtures(np.concatenate(pools))
    logger.info(
        "Combined %s independently sampled pools into %s unique mixtures",
        len(seeds),
        f"{len(pool):,}",
    )
    return pool


def sample_standard_candidate_pool(
    space: CandidatePoolInputs,
    size_per_seed: int,
    seeds: tuple[int, ...],
) -> CandidatePool:
    weights = sample_candidate_pool_union(space, size_per_seed, seeds)
    provenance: PoolProvenance = {
        "kind": "global_and_local_simplex_sampling",
        "parameters": {
            "size_per_seed": size_per_seed,
            "global_fraction": GLOBAL_PROPOSAL_FRACTION,
            "global_dirichlet_concentration_range": list(GLOBAL_CONCENTRATION_RANGE),
            "center_designs": (
                "availability-proportional target design and observations with matching mixture components"
            ),
            "availability_proportional_floor": PROPORTIONAL_FLOOR,
            "lognormal_scale_range": list(PERTURBATION_SCALE_RANGE),
        },
    }
    return CandidatePool(weights, seeds, provenance)


def exclude_observed(pool: np.ndarray, observed_weights: np.ndarray) -> np.ndarray:
    observed = {row.tobytes() for row in np.round(observed_weights.reshape(len(observed_weights), -1), decimals=12)}
    keep = [row.tobytes() not in observed for row in np.round(pool.reshape(len(pool), -1), decimals=12)]
    return pool[np.asarray(keep)]


def quantize_mixtures(weights: np.ndarray) -> np.ndarray:
    """Project each phase onto the 49,152-count simplex lattice."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 3:
        raise ValueError("Mixture weights must have candidate, phase, and component axes")
    if np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError("Mixture weights must be finite and non-negative")
    if not np.allclose(weights.sum(axis=-1), 1.0):
        raise ValueError("Every phase must be a simplex vector")
    scaled = weights * MIXTURE_DENOMINATOR
    counts = np.floor(scaled).astype(np.int64)
    fractions = scaled - counts
    flat_counts = counts.reshape(-1, counts.shape[-1])
    flat_fractions = fractions.reshape(-1, fractions.shape[-1])
    remaining = MIXTURE_DENOMINATOR - flat_counts.sum(axis=1)
    order = np.argsort(-flat_fractions, axis=1, kind="stable")
    for offset in range(weights.shape[-1]):
        rows = np.flatnonzero(remaining > offset)
        flat_counts[rows, order[rows, offset]] += 1
    return counts / MIXTURE_DENOMINATOR


def validate_candidate_pool(weights: np.ndarray, expected_shape: tuple[int, int]) -> np.ndarray:
    """Require a candidate pool on the configured simplex lattice."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape[1:] != expected_shape:
        raise ValueError(f"Candidate phases and components must have shape {expected_shape}")
    quantized = quantize_mixtures(weights)
    if not np.allclose(weights, quantized, rtol=0.0, atol=1e-12):
        raise ValueError(f"Candidate weights must be multiples of 1/{MIXTURE_DENOMINATOR}")
    return weights


def unique_mixtures(weights: np.ndarray) -> np.ndarray:
    flat = np.round(weights.reshape(len(weights), -1), decimals=12)
    _, indices = np.unique(flat, axis=0, return_index=True)
    return weights[np.sort(indices)]


def acquire_posterior_mean(
    model: MixturePredictor,
    target: Swarm,
    weights: np.ndarray,
) -> AcquiredCandidate:
    """Maximize the PosteriorMean acquisition function over a candidate set."""
    values = []
    for start in range(0, len(weights), ACQUISITION_CHUNK_SIZE):
        stop = min(start + ACQUISITION_CHUNK_SIZE, len(weights))
        values.append(model.predict(target, weights[start:stop]).mean)
        if stop == len(weights) or stop % PROGRESS_ROWS == 0:
            logger.info("Scored %s/%s pool rows", f"{stop:,}", f"{len(weights):,}")
    acquisition_values = np.concatenate(values)
    pool_index = int(np.argmax(acquisition_values))
    return AcquiredCandidate(
        pool_index=pool_index,
        acquisition_value=float(acquisition_values[pool_index]),
        acquisition_values=acquisition_values,
    )


def acquire_log_nei(
    model: BotorchMixturePredictor,
    target: Swarm,
    weights: np.ndarray,
    seed: int,
    pending_weights: np.ndarray | None = None,
) -> AcquiredCandidate:
    """Evaluate one-point log noisy expected improvement over a finite pool."""
    baseline = model.candidate_tensor(target, target.data.weights)
    pending = None if pending_weights is None else model.candidate_tensor(target, pending_weights)
    sampler = SobolQMCNormalSampler(torch.Size([LOG_NEI_SAMPLES]), seed=seed)
    acquisition = qLogNoisyExpectedImprovement(
        model=model.botorch_model,
        X_baseline=baseline,
        X_pending=pending,
        sampler=sampler,
        prune_baseline=True,
        incremental=True,
    )
    values = []
    with torch.no_grad():
        for start in range(0, len(weights), ACQUISITION_CHUNK_SIZE):
            stop = min(start + ACQUISITION_CHUNK_SIZE, len(weights))
            candidates = model.candidate_tensor(target, weights[start:stop])
            values.append(acquisition(candidates.unsqueeze(-2)).detach().cpu().numpy().reshape(-1))
            if stop == len(weights) or stop % PROGRESS_ROWS == 0:
                logger.info("Scored %s/%s pool rows with log NEI", f"{stop:,}", f"{len(weights):,}")
    acquisition_values = np.concatenate(values)
    pool_index = int(np.argmax(acquisition_values))
    return AcquiredCandidate(pool_index, float(acquisition_values[pool_index]), acquisition_values)


def _posterior_mean_score(
    model: BotorchMixturePredictor,
    target: Swarm,
    weights: np.ndarray,
    _seed: int | None,
    _pending_weights: np.ndarray | None,
) -> AcquiredCandidate:
    return acquire_posterior_mean(model, target, weights)


def _log_nei_score(
    model: BotorchMixturePredictor,
    target: Swarm,
    weights: np.ndarray,
    seed: int | None,
    pending_weights: np.ndarray | None,
) -> AcquiredCandidate:
    if seed is None:
        raise ValueError("Log NEI requires an acquisition seed")
    return acquire_log_nei(model, target, weights, seed, pending_weights)


POSTERIOR_MEAN = Acquisition(
    POSTERIOR_MEAN_NAME,
    POSTERIOR_MEAN_SELECTION_RULE,
    None,
    _posterior_mean_score,
)


def log_nei(seed: int) -> Acquisition:
    return Acquisition(LOG_NEI_NAME, LOG_NEI_SELECTION_RULE, seed, _log_nei_score)


def acquire_candidate(
    acquisition: Acquisition,
    model: BotorchMixturePredictor,
    target: Swarm,
    weights: np.ndarray,
    pending_weights: np.ndarray | None = None,
) -> AcquiredCandidate:
    return acquisition.score(model, target, weights, acquisition.seed, pending_weights)


def select_candidate(
    campaign: Campaign,
    model: BotorchMixturePredictor,
    weights: np.ndarray,
    acquisition: Acquisition,
) -> CandidateSelection:
    weights = validate_candidate_pool(weights, campaign.target.data.weights.shape[1:])
    acquired = acquire_candidate(acquisition, model, campaign.target, weights)
    return build_candidate_selection(
        campaign,
        model,
        weights,
        acquired,
        acquisition=acquisition,
    )


def build_candidate_selection(
    campaign: Campaign,
    model: MixturePredictor,
    candidate_weights: np.ndarray,
    acquired: AcquiredCandidate,
    *,
    acquisition: Acquisition,
) -> CandidateSelection:
    """Interpret one acquisition result without assuming how it was selected."""
    candidate_weights = validate_candidate_pool(candidate_weights, campaign.target.data.weights.shape[1:])
    weights = candidate_weights[acquired.pool_index]
    moments = model.predict(campaign.target, candidate_weights[acquired.pool_index : acquired.pool_index + 1])
    objective_mean = float(moments.mean[0])
    objective_sd = float(np.sqrt(max(moments.latent_variance[0], 0.0)))
    incumbent = float(campaign.objective.observations(campaign.target).values.max())
    probability = (
        float(objective_mean > incumbent)
        if objective_sd == 0.0
        else 1.0 - NormalDist(mu=objective_mean, sigma=objective_sd).cdf(incumbent)
    )
    posterior: CandidatePosterior = {
        "objective_mean": objective_mean,
        "objective_sd": objective_sd,
        "uncertainty_kind": LATENT_FUNCTION_UNCERTAINTY,
        "incumbent_objective_value": incumbent,
        "probability_of_improvement": probability,
    }
    return CandidateSelection(
        acquired=acquired,
        weights=weights,
        posterior=posterior,
        acquisition_function=acquisition.name,
        selection_rule=acquisition.selection_rule,
        acquisition_seed=acquisition.seed,
    )
